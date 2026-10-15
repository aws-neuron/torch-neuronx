"""Hardware-free tests for with-hw-lock."""

import os
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "with-hw-lock"


def run(lock, *args):
    env = dict(os.environ, AGENT_HW_LOCK=str(lock))
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)


def test_no_command_exits_two(tmp_path):
    assert run(tmp_path / "hw.lock").returncode == 2
    assert run(tmp_path / "hw.lock", "--").returncode == 2


def test_runs_the_command_and_keeps_its_exit_code(tmp_path):
    lock = tmp_path / "nested" / "hw.lock"
    assert run(lock, "--", "sh", "-c", "exit 3").returncode == 3
    assert lock.exists()


def test_second_command_waits_for_the_first(tmp_path):
    lock = tmp_path / "hw.lock"
    log = tmp_path / "log"
    env = dict(os.environ, AGENT_HW_LOCK=str(lock))
    cmd = f"echo start >> {log}; sleep 1; echo end >> {log}"
    first = subprocess.Popen([str(SCRIPT), "--", "sh", "-c", cmd], env=env)
    time.sleep(0.3)
    second = subprocess.Popen([str(SCRIPT), "--", "sh", "-c", cmd], env=env)
    assert first.wait(timeout=30) == 0
    assert second.wait(timeout=30) == 0
    assert log.read_text().split() == ["start", "end", "start", "end"]
