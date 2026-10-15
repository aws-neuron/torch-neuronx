"""Test shared support used by the ModuScope shard runners."""

from __future__ import annotations

import os
import shutil
import signal
from types import SimpleNamespace

import moduscope_runner_support as runner_support
import pytest


def set_installed_config(monkeypatch, tmp_path, task_config) -> None:
    package_dir = tmp_path / "cli_scripts"
    config_dir = package_dir / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "test.yaml").write_text("task_config:\n  _target_: fake_config.TaskConfig\n")
    package_spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    config_module = SimpleNamespace(TaskConfig=task_config)
    monkeypatch.setattr(runner_support.importlib.util, "find_spec", lambda name: package_spec)
    monkeypatch.setattr(runner_support.importlib, "import_module", lambda name: config_module)


@pytest.mark.parametrize(
    "package_spec",
    [None, SimpleNamespace(submodule_search_locations=())],
    ids=["missing", "not-a-package"],
)
def test_installed_config_field_rejects_a_missing_or_nonpackage_config_module(
    monkeypatch, package_spec
):
    monkeypatch.setattr(runner_support.importlib.util, "find_spec", lambda name: package_spec)

    with pytest.raises(ModuleNotFoundError, match="cli_scripts is not installed"):
        runner_support.installed_config_field("test", "members")


@pytest.mark.parametrize("field_source", ["model-fields", "class-attribute"])
def test_installed_config_field_reads_pydantic_or_plain_class_defaults(
    monkeypatch, tmp_path, field_source
):
    expected_members = ("first", "second")
    calls = []

    class Field:
        def get_default(self, *, call_default_factory):
            calls.append(call_default_factory)
            return expected_members

    task_config = (
        SimpleNamespace(model_fields={"members": Field()})
        if field_source == "model-fields"
        else SimpleNamespace(members=expected_members)
    )
    set_installed_config(monkeypatch, tmp_path, task_config)

    assert runner_support.installed_config_field("test", "members") == expected_members
    assert calls == ([True] if field_source == "model-fields" else [])


def test_clear_compile_caches_removes_only_compile_cache_directories(compile_cache_directories):
    cache_names = (
        "hlo_cache",
        "neff_cache",
        "torchinductor_root",
        "torchinductor_ci",
        "nki_artifacts",
    )
    cache_directories = [compile_cache_directories / name for name in cache_names]
    for cache_directory in cache_directories:
        cache_directory.mkdir()

    assert set(runner_support.clear_compile_caches()) == {
        str(directory) for directory in cache_directories[:4]
    }
    assert [directory.exists() for directory in cache_directories] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert runner_support.clear_compile_caches() == ()


def test_run_member_process_prints_and_executes(monkeypatch, capsys):
    invocation = SimpleNamespace(argv=("python", "test.py"))
    reason = "killed by the runner after 7s (exit code 124)"
    result = runner_support.process.InvocationResult(124, reason, 7)
    monkeypatch.setattr(runner_support.process, "run_test_invocation", lambda *args: result)
    seen = runner_support.run_member_process(
        invocation, 7, "Running: ", "Component suite exceeded 7s and was killed."
    )
    assert seen is result
    assert capsys.readouterr().out == (
        "Running: python test.py\nComponent suite exceeded 7s and was killed.\n"
    )


def fake_clock():
    """Advance the fake clock during sleep.

    Fail if a loop reads the clock 10,000 times without sleeping.
    """
    fake = SimpleNamespace(now=1000.0, reads=0)

    def monotonic():
        fake.reads += 1
        assert fake.reads <= 10_000, "10,000 clock reads without a sleep: a loop that never sleeps"
        return fake.now

    def sleep(seconds):
        fake.now, fake.reads = fake.now + seconds, 0

    fake.monotonic, fake.sleep = monotonic, sleep
    return fake


def write_process_tree(root, processes):
    """Create device-file links for each process ID."""
    for process_id, targets in processes.items():
        (root / str(process_id) / "fd").mkdir(parents=True)
        for file_descriptor, target in enumerate(targets, start=3):
            (root / str(process_id) / "fd" / str(file_descriptor)).symlink_to(target)


@pytest.fixture(autouse=True)
def fake_process_root(tmp_path, monkeypatch):
    process_root = tmp_path / "proc"
    monkeypatch.setattr(runner_support, "PROC_ROOT", str(process_root))
    return process_root


def test_device_sweep_waits_for_a_holder_through_the_grace_period(fake_process_root, monkeypatch):
    fake_time = fake_clock()
    monkeypatch.setattr(runner_support, "time", fake_time)
    monkeypatch.setattr(runner_support, "send_sigkill", lambda _process_id: None)
    write_process_tree(fake_process_root, {4242: ["/dev/neuron0"]})

    assert runner_support.stop_new_device_holders(set()) == ([4242], [], [4242])
    assert fake_time.now == 1030.0

    shutil.rmtree(fake_process_root / "4242")
    write_process_tree(fake_process_root, {4243: ["/dev/neuron0"]})
    monkeypatch.setattr(
        runner_support,
        "send_sigkill",
        lambda process_id: shutil.rmtree(fake_process_root / str(process_id)),
    )

    assert runner_support.stop_new_device_holders(set()) == ([4243], [], [])
    assert fake_time.now == 1030.0


def test_device_sweep_polls_until_sigkill_releases_the_holder(monkeypatch):
    # Keep the holder visible for two polls after SIGKILL.
    fake_time, kills, listings = fake_clock(), [], []
    monkeypatch.setattr(runner_support, "time", fake_time)
    monkeypatch.setattr(
        runner_support.os,
        "kill",
        lambda process_id, sent_signal: kills.append((process_id, sent_signal, len(listings))),
    )

    def holders(_process_root, _runner_process_id):
        listings.append(fake_time.now)
        return [] if kills and len(listings) >= kills[0][2] + 3 else [4242]

    monkeypatch.setattr(runner_support, "neuron_device_holders", holders)

    assert runner_support.stop_new_device_holders(set()) == ([4242], [], [])
    assert kills == [(4242, signal.SIGKILL, 1)]
    assert listings == [1000.0, 1000.0, 1000.5, 1001.0]


def test_device_sweep_kills_a_late_holder(monkeypatch):
    snapshots = iter(([4242], [4242, 4243], [4243], []))
    fake_time, killed = fake_clock(), []
    monkeypatch.setattr(runner_support, "time", fake_time)
    monkeypatch.setattr(
        runner_support,
        "neuron_device_holders",
        lambda _process_root, _runner_process_id: list(next(snapshots)),
    )
    monkeypatch.setattr(runner_support, "send_sigkill", killed.append)

    assert runner_support.stop_new_device_holders(set()) == ([4242, 4243], [], [])
    assert killed == [4242, 4243]


@pytest.mark.parametrize("error", [PermissionError, ProcessLookupError], ids=["not-ours", "gone"])
def test_sigkill_ignores_missing_or_unowned_processes(monkeypatch, error):
    attempts = []

    def raising(process_id, sent_signal):
        attempts.append((process_id, sent_signal))
        raise error()

    monkeypatch.setattr(runner_support.os, "kill", raising)

    runner_support.send_sigkill(4242)

    assert attempts == [(4242, signal.SIGKILL)]


def test_device_holder_scan_finds_reparented_orphans_but_skips_runner(
    fake_process_root, monkeypatch
):
    # The orphan belongs to process 1. A parent filter would miss the device holder.
    orphan_process_id = 4242
    killed = []
    write_process_tree(
        fake_process_root,
        {os.getpid(): ["/dev/neuron0"], orphan_process_id: ["/dev/neuron1"], 4243: ["/dev/null"]},
    )
    (fake_process_root / str(orphan_process_id) / "status").write_text("PPid:\t1\n")
    (fake_process_root / "4244").mkdir()
    (fake_process_root / "cpuinfo").write_text("")

    def record_and_reap(process_id):
        killed.append(process_id)
        shutil.rmtree(fake_process_root / str(process_id))

    monkeypatch.setattr(runner_support, "time", fake_clock())
    monkeypatch.setattr(runner_support, "send_sigkill", record_and_reap)

    assert runner_support.neuron_device_holders(str(fake_process_root), os.getpid()) == [
        orphan_process_id
    ]
    assert runner_support.device_holder_baseline() == {orphan_process_id}
    assert runner_support.stop_new_device_holders(set()) == ([orphan_process_id], [], [])
    assert killed == [orphan_process_id]
