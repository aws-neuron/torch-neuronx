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
