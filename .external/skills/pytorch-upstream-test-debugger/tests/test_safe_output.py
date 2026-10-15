import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "safe_output.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("safe_output", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_install_files_replaces_a_complete_set(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("old\n")
    fd, identity = MODULE.open_output_dir(output)
    try:
        MODULE.install_files(fd, identity, {"old.txt": b"new\n", "added.txt": b"added\n"})
    finally:
        os.close(fd)
    assert (output / "old.txt").read_bytes() == b"new\n"
    assert (output / "added.txt").read_bytes() == b"added\n"
    assert sorted(path.name for path in output.iterdir()) == ["added.txt", "old.txt"]


@pytest.mark.parametrize("swap_component", ["ancestor", "final"])
def test_path_swap_after_preflight_fails_without_artifacts(tmp_path, monkeypatch, swap_component):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = parent / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    displaced = parent.with_name("parent.displaced")
    displaced_output = displaced / "output"

    def swap(stage, _fd):
        if stage != "after_preflight":
            return
        if swap_component == "ancestor":
            parent.rename(displaced)
            parent.symlink_to(outside, target_is_directory=True)
        else:
            output.rename(output.with_name("output.displaced"))
            output.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(MODULE, "_TEST_HOOK", swap)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match="output.*identity changed|path changed"):
            MODULE.install_files(fd, identity, {"result.txt": b"safe\n"})
    finally:
        os.close(fd)

    if swap_component == "ancestor":
        actual_displaced = displaced_output
    else:
        actual_displaced = output.with_name("output.displaced")
    assert list(outside.iterdir()) == []
    assert list(actual_displaced.iterdir()) == []


def test_late_failure_restores_all_existing_and_absent_outputs(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "first.txt").write_text("first old\n")
    (output / "second.txt").write_text("second old\n")
    monkeypatch.setenv("SAFE_OUTPUT_TEST_FAIL_INSTALL", "3")
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match="injected late"):
            MODULE.install_files(
                fd,
                identity,
                {
                    "first.txt": b"first new\n",
                    "second.txt": b"second new\n",
                    "third.txt": b"third new\n",
                },
            )
    finally:
        os.close(fd)
    assert (output / "first.txt").read_text() == "first old\n"
    assert (output / "second.txt").read_text() == "second old\n"
    assert not (output / "third.txt").exists()
    assert sorted(path.name for path in output.iterdir()) == ["first.txt", "second.txt"]


def test_rejects_symlink_directory_and_nonregular_final(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        MODULE.open_output_dir(link)

    output = tmp_path / "output"
    output.mkdir()
    (output / "bad").mkdir()
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match="not a regular file"):
            MODULE.install_files(fd, identity, {"bad": b"data"})
    finally:
        os.close(fd)
    assert (output / "bad").is_dir()


def test_cli_install_and_help(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "result.txt").write_text("result\n")
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "install", "--dir", str(output), "--from-dir", str(staging)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "result.txt").read_text() == "result\n"

    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    assert help_result.returncode == 0
    assert "usage:" in help_result.stdout.lower()


def test_write_fsync_failure_removes_hidden_staging_file(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    fd, identity = MODULE.open_output_dir(output)

    def fail_fsync(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(MODULE.os, "fsync", fail_fsync)
    try:
        with pytest.raises(OSError, match="injected fsync failure"):
            MODULE.install_files(fd, identity, {"result.txt": b"result\n"})
    finally:
        os.close(fd)
    assert list(output.iterdir()) == []


def test_backup_placeholder_is_removed_when_close_fails(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    fd, _ = MODULE.open_output_dir(output)
    real_close = MODULE.os.close
    failed = False

    def fail_one_close(descriptor):
        nonlocal failed
        if not failed and descriptor != fd:
            failed = True
            real_close(descriptor)
            raise OSError("injected backup close failure")
        real_close(descriptor)

    monkeypatch.setattr(MODULE.os, "close", fail_one_close)
    try:
        with pytest.raises(OSError, match="injected backup close failure"):
            MODULE._reserve_backup(fd, "result.txt")
    finally:
        real_close(fd)
    assert list(output.iterdir()) == []


def test_cli_run_redacts_json_and_text_before_install(tmp_path):
    output = tmp_path / "output"
    command = tmp_path / "write_artifacts.py"
    sensitive = "/" + "home/person/private.txt"
    account = "123456" * 2
    secret_assignment = " " + "to" + "ken=hidden"
    command.write_text(
        "import json, os, pathlib\n"
        "root = pathlib.Path(os.environ['SAFE_OUTPUT_STAGE'])\n"
        f"value = {sensitive!r} + ' ' + {account!r} + {secret_assignment!r}\n"
        "(root / 'report.json').write_text(json.dumps({'nested': [value]}))\n"
        "(root / 'pytest.log').write_text(value + '\\n')\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--dir",
            str(output),
            "--name",
            "report.json",
            "--name",
            "pytest.log",
            "--redact-all",
            "--",
            sys.executable,
            str(command),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = __import__("json").loads((output / "report.json").read_text())
    installed = report["nested"][0] + (output / "pytest.log").read_text()
    for value in (sensitive, account, "hidden"):
        assert value not in installed


def test_directory_fsync_failure_restores_backup_without_hidden_files(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.txt").write_text("old\n")
    fd, identity = MODULE.open_output_dir(output)
    real_fsync = MODULE.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "fsync", fail_directory_fsync)
    try:
        with pytest.raises(OSError, match="injected directory fsync failure"):
            MODULE.install_files(fd, identity, {"result.txt": b"new\n"})
    finally:
        os.close(fd)
    assert (output / "result.txt").read_text() == "old\n"
    assert [path.name for path in output.iterdir()] == ["result.txt"]


def _make_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    originals = {
        "first.txt": b"first old\n",
        "second.txt": b"second old\n",
        "third.txt": b"third old\n",
    }
    for name, data in originals.items():
        (output / name).write_bytes(data)
    replacements = {name: data.replace(b"old", b"new") for name, data in originals.items()}
    return output, originals, replacements


def _assert_originals_and_no_hidden(output, originals):
    assert {path.name for path in output.iterdir()} == set(originals)
    for name, data in originals.items():
        assert (output / name).read_bytes() == data


def _rename_phase(source, destination):
    if ".backup." in destination:
        return "backup"
    if ".new." in source:
        return "install"
    if ".backup." in source:
        return "rollback"
    return "other"


@pytest.mark.parametrize("phase", ["backup", "install"])
@pytest.mark.parametrize("failure_index", [1, 2, 3])
def test_each_backup_and_install_rename_failure_restores_everything(
    tmp_path, monkeypatch, phase, failure_index
):
    output, originals, replacements = _make_existing_output(tmp_path)
    real_replace = MODULE.os.replace
    phase_calls = 0

    def fail_selected_replace(source, destination, **kwargs):
        nonlocal phase_calls
        if _rename_phase(source, destination) == phase:
            phase_calls += 1
            if phase_calls == failure_index:
                raise OSError(f"injected {phase} rename failure {failure_index}")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(MODULE.os, "replace", fail_selected_replace)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match=f"injected {phase} rename failure"):
            MODULE.install_files(fd, identity, replacements)
    finally:
        os.close(fd)
    _assert_originals_and_no_hidden(output, originals)


@pytest.mark.parametrize("failure_index", [1, 2, 3])
def test_each_rollback_rename_retries_and_restores_everything(tmp_path, monkeypatch, failure_index):
    output, originals, replacements = _make_existing_output(tmp_path)
    real_replace = MODULE.os.replace
    real_fsync = MODULE.os.fsync
    rollback_calls = 0

    def fail_selected_restore(source, destination, **kwargs):
        nonlocal rollback_calls
        if _rename_phase(source, destination) == "rollback":
            rollback_calls += 1
            if rollback_calls == failure_index:
                raise OSError(f"injected rollback rename failure {failure_index}")
        return real_replace(source, destination, **kwargs)

    file_fsyncs = 0

    def fail_commit_fsync(descriptor):
        nonlocal file_fsyncs
        file_fsyncs += 1
        if file_fsyncs == 4:
            raise OSError("injected commit fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "replace", fail_selected_restore)
    monkeypatch.setattr(MODULE.os, "fsync", fail_commit_fsync)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match="injected commit fsync failure"):
            MODULE.install_files(fd, identity, replacements)
    finally:
        os.close(fd)
    _assert_originals_and_no_hidden(output, originals)


@pytest.mark.parametrize("failure_index", [1, 2, 3, 4])
def test_each_staging_or_commit_fsync_failure_restores_everything(
    tmp_path, monkeypatch, failure_index
):
    output, originals, replacements = _make_existing_output(tmp_path)
    real_fsync = MODULE.os.fsync
    calls = 0

    def fail_selected_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_index:
            raise OSError(f"injected fsync failure {failure_index}")
        return real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "fsync", fail_selected_fsync)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match=f"injected fsync failure {failure_index}"):
            MODULE.install_files(fd, identity, replacements)
    finally:
        os.close(fd)
    _assert_originals_and_no_hidden(output, originals)


def test_rollback_fsync_failure_preserves_original_error_and_visible_files(tmp_path, monkeypatch):
    output, originals, replacements = _make_existing_output(tmp_path)
    real_fsync = MODULE.os.fsync
    calls = 0

    def fail_rollback_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected rollback fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setenv("SAFE_OUTPUT_TEST_FAIL_INSTALL", "1")
    monkeypatch.setattr(MODULE.os, "fsync", fail_rollback_fsync)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError) as raised:
            MODULE.install_files(fd, identity, replacements)
    finally:
        os.close(fd)
    assert "injected late install failure" in str(raised.value)
    assert "injected rollback fsync failure" in str(raised.value)
    _assert_originals_and_no_hidden(output, originals)


def test_unrecoverable_backup_names_file_and_restores_other_entries(tmp_path, monkeypatch):
    output, originals, replacements = _make_existing_output(tmp_path)
    real_replace = MODULE.os.replace

    def fail_second_restore(source, destination, **kwargs):
        if _rename_phase(source, destination) == "rollback" and destination == "second.txt":
            raise OSError("injected permanent restore failure")
        return real_replace(source, destination, **kwargs)

    real_fsync = MODULE.os.fsync
    calls = 0

    def fail_commit_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected primary commit failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "replace", fail_second_restore)
    monkeypatch.setattr(MODULE.os, "fsync", fail_commit_fsync)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError) as raised:
            MODULE.install_files(fd, identity, replacements)
    finally:
        os.close(fd)
    message = str(raised.value)
    assert "injected primary commit failure" in message
    assert "unrecovered files: second.txt" in message
    assert (output / "first.txt").read_bytes() == originals["first.txt"]
    assert (output / "third.txt").read_bytes() == originals["third.txt"]
    assert not (output / "second.txt").exists()
    hidden = [path for path in output.iterdir() if path.name.startswith(".second.txt.backup.")]
    assert len(hidden) == 1
    assert hidden[0].read_bytes() == originals["second.txt"]


def test_swap_between_revalidations_fails_before_first_commit_rename(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.txt").write_text("old\n")
    displaced = tmp_path / "output.displaced"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    real_revalidate = MODULE._revalidate_requested_output
    calls = 0

    def swap_after_first_check(fd, identity, output_path):
        nonlocal calls
        calls += 1
        real_revalidate(fd, identity, output_path)
        if calls == 1:
            output.rename(displaced)
            replacement.rename(output)

    monkeypatch.setattr(MODULE, "_revalidate_requested_output", swap_after_first_check)
    fd, identity = MODULE.open_output_dir(output)
    try:
        with pytest.raises(OSError, match="identity changed"):
            MODULE.install_files(fd, identity, {"result.txt": b"new\n"})
    finally:
        os.close(fd)
    assert list(output.iterdir()) == []
    assert (displaced / "result.txt").read_text() == "old\n"
    assert [path.name for path in displaced.iterdir()] == ["result.txt"]


@pytest.mark.parametrize("phase", ["backup", "install"])
@pytest.mark.parametrize("swap_component", ["ancestor", "final"])
def test_cli_path_swap_during_commit_rolls_back(
    tmp_path, monkeypatch, capsys, phase, swap_component
):
    ancestor = tmp_path / "requested"
    ancestor.mkdir()
    output = ancestor / "output"
    output.mkdir()
    originals = {"first.txt": b"first old\n", "second.txt": b"second old\n"}
    for name, data in originals.items():
        (output / name).write_bytes(data)

    staging = tmp_path / "staging"
    staging.mkdir()
    for name, data in originals.items():
        (staging / name).write_bytes(data.replace(b"old", b"new"))

    external = tmp_path / "external"
    external.mkdir()
    real_replace = MODULE.os.replace
    swapped = False
    displaced = tmp_path / "requested.displaced"

    def swap_before_first_phase_rename(source, destination, **kwargs):
        nonlocal swapped, displaced
        if not swapped and _rename_phase(source, destination) == phase:
            swapped = True
            if swap_component == "ancestor":
                ancestor.rename(displaced)
                ancestor.symlink_to(external, target_is_directory=True)
            else:
                displaced = output.with_name("output.displaced")
                output.rename(displaced)
                output.symlink_to(external, target_is_directory=True)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(MODULE.os, "replace", swap_before_first_phase_rename)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "install", "--dir", str(output), "--from-dir", str(staging)],
    )

    assert MODULE.main() != 0
    assert "path changed" in capsys.readouterr().err
    assert swapped
    displaced_output = displaced / "output" if swap_component == "ancestor" else displaced
    _assert_originals_and_no_hidden(displaced_output, originals)
    assert list(external.iterdir()) == []
