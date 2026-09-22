"""Test the process helpers for one entry command."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import moduscope_process as process
import pytest

# Use at least 3 seconds because --forked tests can share a busy host.
PROCESS_TIMEOUT_SECONDS, TERMINATION_GRACE_SECONDS, TIMING_SLACK_SECONDS = 3.0, 1.0, 2.0
# The alarm stops this helper if the timeout test leaves it running.
UNRESPONSIVE_PROCESS_SCRIPT = (
    "import os, signal, sys, time; signal.alarm(15)\n"
    + "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    + "while True: open(sys.argv[2], 'a').write('x\\n'); time.sleep(0.005)\n"
)
# The worker can share the launcher group or use its own session.
WORKER_SCRIPT = (
    "import signal, sys, time; signal.alarm(60)\n"
    + "if sys.argv[2] == 'deaf': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    + "while True: open(sys.argv[1], 'a').write('x\\n'); time.sleep(0.005)\n"
)
LAUNCHER_SCRIPT = f"""\
import os, signal, subprocess, sys
heartbeat, pid_file, mode = sys.argv[1:]
own_session = mode == "own-session"
worker = subprocess.Popen(
    [sys.executable, "-c", {WORKER_SCRIPT!r}, heartbeat, mode], start_new_session=own_session
)
def teardown(signum, frame):
    (os.killpg if own_session else os.kill)(worker.pid, signal.SIGTERM)
    worker.wait()
    sys.exit(1)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "deaf" else teardown)
open(pid_file, "w").write(str(worker.pid))
if mode == "exit":
    sys.exit(0)
signal.alarm(60)
while True:
    signal.pause()
"""


def test_emitted_token_lines_are_printed_and_written_in_order(tmp_path, capsys):
    process.emit_token_line("RUN target=t kind=hf shard=0 run=1")
    print("Entry 0 (x/y), 900s bound: python -m cli_scripts.task_runner")
    process.emit_token_line("::warning::model x_y status=fail error=bad")
    process.write_token_lines(str(tmp_path))
    assert capsys.readouterr().out.splitlines() == [
        "RUN target=t kind=hf shard=0 run=1",
        "Entry 0 (x/y), 900s bound: python -m cli_scripts.task_runner",
        "::warning::model x_y status=fail error=bad",
    ]
    assert (tmp_path / "tokens.txt").read_text() == (
        "RUN target=t kind=hf shard=0 run=1\n::warning::model x_y status=fail error=bad\n"
    )
    process.write_token_lines(str(tmp_path / "missing"))
    assert not (tmp_path / "missing").exists()


def test_run_test_invocation_merges_overrides_and_unsets(monkeypatch):
    monkeypatch.setenv("KEEP", "inherited")
    monkeypatch.setenv("DEVICES", "neuron")
    monkeypatch.setenv("TEST_FILES", "test_00_rms_norm.py")
    invocation = process.TestInvocation(
        argv=("python", "-m", "suite"),
        cwd="/suite",
        env={"MODEL_CONFIG": "x"},
        unset_env=("DEVICES", "TEST_FILES"),
    )
    calls = []

    def run_with_timeout(argv, timeout_seconds, **kwargs):
        calls.append((argv, timeout_seconds, kwargs))
        return 137, "ModuScope exited with code 137"

    monkeypatch.setattr(process, "run_with_timeout", run_with_timeout)

    result = process.run_test_invocation(invocation, 30)

    assert (result.returncode, result.reason) == (137, "ModuScope exited with code 137")
    assert result.seconds >= 0
    assert calls[0][:2] == (invocation.argv, 30)
    assert calls[0][2]["cwd"] == "/suite"
    child_env = calls[0][2]["env"]
    assert child_env["KEEP"] == "inherited"
    assert child_env["MODEL_CONFIG"] == "x"
    assert "DEVICES" not in child_env
    assert "TEST_FILES" not in child_env


def test_run_test_invocation_measures_child_runtime():
    invocation = process.TestInvocation(argv=(sys.executable, "-c", "import time; time.sleep(0.2)"))

    result = process.run_test_invocation(invocation, 5)

    assert result.returncode == 0
    assert 0.2 <= result.seconds < 5


def test_child_process_group_normal_exit_ignores_a_missing_group(monkeypatch):
    sent_signals = []

    def raise_missing_group(_process_id, signal_number):
        sent_signals.append(signal_number)
        raise ProcessLookupError

    monkeypatch.setattr(process.os, "killpg", raise_missing_group)
    group_check_script = "import os, sys; sys.exit(0 if os.getpgid(0) == os.getpid() else 7)"
    assert (
        process.run_in_process_group([sys.executable, "-c", group_check_script], timeout_seconds=30)
        == 0
    )
    assert sent_signals == [signal.SIGKILL]


def test_process_group_timeout_stops_waiting_for_an_unreachable_leader(
    tmp_path, child_process_id_file, monkeypatch
):
    # Skip signal delivery to simulate a process stuck in the kernel.
    sent_signals = []

    def raise_missing_group(_process_id, signal_number):
        sent_signals.append(signal_number)
        raise ProcessLookupError

    monkeypatch.setattr(process.os, "killpg", raise_missing_group)
    heartbeat_path = tmp_path / "heartbeat"
    argv = [
        sys.executable,
        "-c",
        UNRESPONSIVE_PROCESS_SCRIPT,
        str(child_process_id_file),
        str(heartbeat_path),
    ]
    start_time = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        process.run_in_process_group(
            argv,
            PROCESS_TIMEOUT_SECONDS,
            grace_seconds=TERMINATION_GRACE_SECONDS,
        )
    elapsed_seconds = time.monotonic() - start_time
    assert sent_signals == [signal.SIGTERM, signal.SIGKILL]
    assert (
        PROCESS_TIMEOUT_SECONDS + 2 * TERMINATION_GRACE_SECONDS
        <= elapsed_seconds
        < PROCESS_TIMEOUT_SECONDS + 2 * TERMINATION_GRACE_SECONDS + TIMING_SLACK_SECONDS
    ), f"{elapsed_seconds:.1f}s"
    initial_heartbeat_size = heartbeat_path.stat().st_size
    time.sleep(0.5)
    assert (
        heartbeat_path.stat().st_size > initial_heartbeat_size
    ), f"leader {child_process_id_file.read_text()} stopped writing"


@pytest.mark.parametrize(
    ("mode", "outcome", "minimum_seconds", "maximum_seconds"),
    [
        ("exit", 0, 0, PROCESS_TIMEOUT_SECONDS),
        (
            "own-session",
            "timeout",
            PROCESS_TIMEOUT_SECONDS,
            PROCESS_TIMEOUT_SECONDS + TERMINATION_GRACE_SECONDS + TIMING_SLACK_SECONDS,
        ),
        (
            "deaf",
            "timeout",
            PROCESS_TIMEOUT_SECONDS + TERMINATION_GRACE_SECONDS,
            PROCESS_TIMEOUT_SECONDS + 2 * TERMINATION_GRACE_SECONDS + TIMING_SLACK_SECONDS,
        ),
    ],
    ids=["exit", "own-session", "deaf"],
)
def test_process_group_cleanup_stops_worker_on_exit_or_timeout(
    tmp_path, child_process_id_file, mode, outcome, minimum_seconds, maximum_seconds
):
    # A live launcher must stop worker sessions. A signal-resistant group requires SIGKILL.
    heartbeat = tmp_path / "heartbeat"
    argv = [sys.executable, "-c", LAUNCHER_SCRIPT, str(heartbeat), str(child_process_id_file), mode]
    started = time.monotonic()
    try:
        seen = process.run_in_process_group(
            argv, PROCESS_TIMEOUT_SECONDS, grace_seconds=TERMINATION_GRACE_SECONDS
        )
    except subprocess.TimeoutExpired:
        seen = "timeout"
    elapsed = time.monotonic() - started
    assert seen == outcome
    assert minimum_seconds <= elapsed < maximum_seconds, f"{elapsed:.1f}s"
    heartbeat_size = heartbeat.stat().st_size if heartbeat.exists() else 0
    time.sleep(0.5)
    assert (
        heartbeat.stat().st_size if heartbeat.exists() else 0
    ) == heartbeat_size, f"the {mode} worker {child_process_id_file.read_text()} kept writing"
