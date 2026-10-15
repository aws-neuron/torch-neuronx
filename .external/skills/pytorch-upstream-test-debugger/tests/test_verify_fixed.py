import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "verify_fixed.sh"
NODEID = "tests/test_synthetic.py::test_failure"


def make_record(outcome, message=""):
    record = {"nodeid": NODEID, "outcome": outcome}
    if message:
        record["call"] = {"crash": {"message": message}}
    return record


def run_verify(
    tmp_path,
    baseline_record,
    current_record,
    import_exit=0,
    pytest_exit=None,
    stale_report=None,
    baseline_text=None,
    verify_kind="",
    swap_component="",
    fail_mkdir=False,
    log_text="",
    temp_root=None,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    runner = repo / "tests" / "pytorch_tests_native" / "runner"
    runner.mkdir(parents=True)
    (runner / "env_lock.c").write_text("void env_lock_placeholder(void) {}\n")
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(NODEID + "\n")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        baseline_text if baseline_text is not None else json.dumps({"tests": [baseline_record]})
    )
    output_parent = tmp_path
    if swap_component == "ancestor":
        output_parent = tmp_path / "output-parent"
        output_parent.mkdir()
    output = output_parent / "output"
    if stale_report is not None or verify_kind:
        output.mkdir()
    if stale_report is not None:
        (output / "report.json").write_text(json.dumps(stale_report))
    if verify_kind:
        destination = output / "verify.md"
        if verify_kind == "symlink":
            sentinel = tmp_path / "sentinel.txt"
            sentinel.write_text("keep\n")
            destination.symlink_to(sentinel)
        elif verify_kind == "directory":
            destination.mkdir()
        else:
            os.mkfifo(destination)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os, pathlib, sys
if sys.argv[1:2] == ['-c']:
    sys.exit(int(os.environ.get('FAKE_IMPORT_EXIT', '0')))
if sys.argv[1:3] == ['-m', 'pytest']:
    report_arg = next(a for a in sys.argv if a.startswith('--json-report-file='))
    pathlib.Path(report_arg.split('=', 1)[1]).write_text(os.environ['FAKE_REPORT'])
    configured = os.environ.get('FAKE_PYTEST_EXIT', '')
    print(os.environ.get('FAKE_LOG_TEXT', 'synthetic pytest execution'))
    if configured:
        sys.exit(int(configured))
    sys.exit(0 if '\"outcome\": \"passed\"' in os.environ['FAKE_REPORT'] else 1)
if sys.argv[1:2] == ['-']:
    os.execv(os.environ['REAL_PYTHON'], [os.environ['REAL_PYTHON'], *sys.argv[1:]])
sys.exit(99)
"""
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "FAKE_REPORT": json.dumps({"tests": [current_record]}),
            "FAKE_IMPORT_EXIT": str(import_exit),
            "FAKE_PYTEST_EXIT": "" if pytest_exit is None else str(pytest_exit),
            "FAKE_LOG_TEXT": log_text,
        }
    )
    if swap_component:
        outside = tmp_path / "outside-output"
        outside.mkdir()
        swap_path = output_parent if swap_component == "ancestor" else output
        env["SAFE_OUTPUT_TEST_SWAP_ANCESTOR"] = str(swap_path)
        env["SAFE_OUTPUT_TEST_SWAP_TARGET"] = str(outside)
    if temp_root is not None:
        env["TMPDIR"] = str(temp_root)
    if fail_mkdir:
        env["FAKE_MKDIR_PARENT"] = str(tmp_path / "mkdir-parent.txt")
        fake_mkdir = bin_dir / "mkdir"
        fake_mkdir.write_text(
            '#!/bin/sh\n[ "$1" = "--" ] && shift\n'
            'printf \'%s\n\' "${1%/*}" > "$FAKE_MKDIR_PARENT"\n'
            "exit 1\n"
        )
        fake_mkdir.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(repo), str(tests_file), str(output), str(baseline)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    verify_path = output / "verify.md"
    verify_text = verify_path.read_text() if verify_path.is_file() else ""
    return result, verify_text


def test_fixed_failure_exits_zero(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: bad shape"),
        make_record("passed"),
    )
    assert result.returncode == 0, result.stderr
    assert "- fixed: 1" in verify_text


def test_same_signature_remains_failure_and_is_distinguished(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: shape 128 is invalid"),
        make_record("failed", "ValueError: shape 256 is invalid"),
    )
    assert result.returncode == 2
    assert "- still_bad_same_signature: 1" in verify_text
    assert "- still_bad_different_signature: 0" in verify_text


def test_different_signature_is_changed_not_fixed(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: bad shape"),
        make_record("failed", "RuntimeError: compiler failure"),
    )
    assert result.returncode == 2
    assert "- fixed: 0" in verify_text
    assert "- still_bad_same_signature: 0" in verify_text
    assert "- still_bad_different_signature: 1" in verify_text
    assert "baseline=`ValueError: bad shape`" in verify_text
    assert "current=`RuntimeError: compiler failure`" in verify_text


def test_report_nodeid_mismatch_fails(tmp_path):
    current = {"nodeid": "tests/test_other.py::test_other", "outcome": "passed"}
    result, _ = run_verify(tmp_path, make_record("failed", "boom"), current)
    assert result.returncode == 11
    assert "report node ID set differs" in result.stderr
    assert "rerun failed" in result.stderr


def test_rerun_dependency_failure_is_propagated(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "boom"),
        make_record("passed"),
        import_exit=1,
    )
    assert result.returncode == 10
    assert not verify_text
    assert "rerun failed" in result.stderr


def test_help_exits_zero_without_device_environment():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_pass_to_failure_is_reported_as_regression_and_exits_one(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("passed"),
        make_record("failed", "RuntimeError: new defect"),
    )
    assert result.returncode == 1
    assert "- regressed: 1" in verify_text
    assert "# verify: fixed=0 regressed=1" in result.stderr


def test_stale_report_cannot_create_verify_fixed_false_success(tmp_path):
    stale = {"tests": [make_record("passed")]}
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: baseline defect"),
        make_record("passed"),
        pytest_exit=1,
        stale_report=stale,
    )
    assert result.returncode == 11
    assert not verify_text
    assert "fresh report contains no failures" in result.stderr
    assert "rerun failed" in result.stderr


def test_worker_crash_stage_diagnostics_are_compared(tmp_path):
    baseline = make_record("failed")
    baseline["???"] = {"longrepr": "worker exited: device runtime crash"}
    current = make_record("failed")
    current["???"] = {"longrepr": "worker exited: host allocation crash"}
    result, verify_text = run_verify(tmp_path, baseline, current)
    assert result.returncode == 2
    assert "- still_bad_different_signature: 1" in verify_text
    assert "device runtime crash" in verify_text
    assert "host allocation crash" in verify_text


def test_truly_malformed_baseline_json_exits_eleven_without_traceback(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "boom"),
        make_record("passed"),
        baseline_text='{"tests": [}',
    )
    assert result.returncode == 11
    assert not verify_text
    assert "invalid JSON report" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_nonregular_verify_report_is_rejected_without_mutation(tmp_path, kind):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "boom"),
        make_record("passed"),
        verify_kind=kind,
    )
    destination = tmp_path / "output" / "verify.md"
    assert result.returncode == 11
    if kind == "directory":
        assert destination.is_dir()
    elif kind == "symlink":
        assert (tmp_path / "sentinel.txt").read_text() == "keep\n"
        assert verify_text == "keep\n"
        assert destination.is_symlink()
    else:
        assert stat.S_ISFIFO(destination.lstat().st_mode)
    assert not (tmp_path / "output" / "report.json").exists()


@pytest.mark.parametrize("argc", range(4))
def test_missing_positional_arguments_print_stable_usage_and_exit_two(argc):
    arguments = ["repo", "tests.txt", "output"][:argc]
    result = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("Usage: verify_fixed.sh ")
    assert "unbound variable" not in result.stderr


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo"])
def test_non_directory_output_path_is_rejected_without_mutation(tmp_path, kind):
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(NODEID + "\n")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"tests": [make_record("failed", "boom")]}))
    output = tmp_path / "output"
    if kind == "regular":
        output.write_text("keep\n")
    elif kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        output.symlink_to(outside, target_is_directory=True)
    else:
        os.mkfifo(output)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path), str(tests_file), str(output), str(baseline)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 11
    if kind == "regular":
        assert output.read_text() == "keep\n"
    elif kind == "symlink":
        assert output.is_symlink()
        assert list(outside.iterdir()) == []
    else:
        assert stat.S_ISFIFO(output.lstat().st_mode)


def test_verify_signatures_redact_before_normalization_on_both_reports(tmp_path):
    home_before = "/" + "home/person/one.py"
    home_after = "/local/" + "home/other/two.py"
    account_before = "123456" * 2
    account_after = "654321" * 2
    key_before = "AKIA" + "E" * 16
    key_after = "ASIA" + "F" * 16
    before_message = (
        f"ValueError: failed at {home_before} {account_before} {key_before} token=before"
    )
    after_message = f"ValueError: failed at {home_after} {account_after} {key_after} token=after"
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", before_message),
        make_record("failed", after_message),
    )
    assert result.returncode == 2, result.stderr
    assert "- still_bad_same_signature: 1" in verify_text
    for sensitive in (
        home_before,
        home_after,
        account_before,
        account_after,
        key_before,
        key_after,
        "before",
        "after",
    ):
        assert sensitive not in verify_text
    assert "<redacted-home>" in verify_text
    assert "<redacted-account>" in verify_text
    assert "<redacted-key-id>" in verify_text
    assert "<redacted-secret>" in verify_text


@pytest.mark.parametrize("swap_component", ["ancestor", "final"])
def test_verify_path_swap_fails_closed_without_external_writes(tmp_path, swap_component):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: bad shape"),
        make_record("passed"),
        swap_component=swap_component,
    )
    assert result.returncode == 11
    assert verify_text == ""
    assert "output-directory" in result.stderr
    assert list((tmp_path / "outside-output").iterdir()) == []
    if swap_component == "ancestor":
        displaced_output = tmp_path / "output-parent.displaced" / "output"
    else:
        displaced_output = tmp_path / "output.displaced"
    assert list(displaced_output.iterdir()) == []


def test_mkdir_failure_removes_exact_verify_temp_directory(tmp_path):
    temp_root = tmp_path / "temp-root"
    temp_root.mkdir()
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "ValueError: bad shape"),
        make_record("passed"),
        fail_mkdir=True,
        temp_root=temp_root,
    )
    created = Path((tmp_path / "mkdir-parent.txt").read_text().strip())
    assert result.returncode == 11
    assert verify_text == ""
    assert created.parent == temp_root
    assert created.name.startswith("pytorch-upstream-verify.")
    assert not created.exists()
    assert list(temp_root.iterdir()) == []


def test_every_installed_verify_artifact_is_redacted(tmp_path):
    import socket

    home = "/" + "Users/person/project.py"
    account = "654321" * 2
    key_id = "ASIA" + "H" * 16
    sensitive_value = "verification-sensitive"
    hostname = socket.gethostname()
    secret_assignment = "pass" + "word=" + sensitive_value
    message = f"failed at {home} {account} {key_id} {secret_assignment} host={hostname}"
    current = make_record("failed", message)
    current["metadata"] = {"nested": [message, {"value": message}]}
    result, _ = run_verify(
        tmp_path,
        make_record("failed", message),
        current,
        log_text=message,
    )
    assert result.returncode == 2, result.stderr
    output = tmp_path / "output"
    assert {path.name for path in output.iterdir()} == {
        "pytest.log",
        "report.json",
        "summary.txt",
        "verify.md",
    }
    json.loads((output / "report.json").read_text())
    installed = "\n".join(path.read_text() for path in output.iterdir())
    for sensitive in (home, account, key_id, sensitive_value, hostname):
        assert sensitive not in installed


def test_pytest_exit_three_is_propagated_without_helper_collision(tmp_path):
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "boom"),
        make_record("passed"),
        pytest_exit=3,
    )
    assert result.returncode == 3
    assert verify_text == ""
    assert "rerun failed" in result.stderr


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "symlink"])
def test_verify_rejects_invalid_tmpdir(tmp_path, kind):
    if kind == "relative":
        temp_root = Path("relative-temp")
    elif kind == "missing":
        temp_root = tmp_path / "missing"
    elif kind == "file":
        temp_root = tmp_path / "temp-file"
        temp_root.write_text("not a directory\n")
    else:
        real_temp = tmp_path / "real-temp"
        real_temp.mkdir()
        temp_root = tmp_path / "temp-link"
        temp_root.symlink_to(real_temp, target_is_directory=True)
    result, verify_text = run_verify(
        tmp_path,
        make_record("failed", "boom"),
        make_record("passed"),
        temp_root=temp_root,
    )
    assert result.returncode == 10
    assert verify_text == ""
    assert "TMPDIR must be an absolute path to an existing real directory" in result.stderr
