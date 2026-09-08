"""Run a model command in its own process group with a timeout, and print the token lines the \
workflow reads."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess

TIMEOUT_EXIT_CODE = 124


def shell_exit_code(returncode: int) -> int:
    """Convert a negative subprocess signal code to the shell's ``128 + signal`` exit code."""
    return 128 - returncode if returncode < 0 else returncode


_token_lines: list[str] = []


def emit_token_line(line: str) -> None:
    """Print and record one token line.

    A token line is a ``KEY=value`` line that the runner prints and later steps parse.
    """
    print(line)
    _token_lines.append(line)


def write_token_lines(outputs_root: str) -> None:
    """Write recorded token lines to ``tokens.txt`` in print order.

    Skip the write when ``outputs_root`` does not exist.
    """
    if not os.path.isdir(outputs_root):
        return
    with open(os.path.join(outputs_root, "tokens.txt"), "w", encoding="utf-8") as token_file:
        token_file.writelines(f"{token_line}\n" for token_line in _token_lines)


def run_in_process_group(
    argv: list[str],
    timeout_seconds: float,
    grace_seconds: float = 45.0,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command in a process group.

    After timeout, send SIGTERM and SIGKILL, then finish after both grace waits even if it survives.
    """
    process = subprocess.Popen(argv, start_new_session=True, cwd=cwd, env=env)
    try:
        return process.wait(timeout=timeout_seconds)
    finally:
        try:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=grace_seconds)
        finally:
            # macOS can return EPERM for a dead process group that Python has not reaped.
            # Ignore it to preserve the child exit code.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace_seconds)


def run_with_timeout(argv: list[str], bound: float, **popen_kwargs) -> tuple[int, str]:
    """Run ``argv`` for ``bound`` seconds before cleanup, which can add two grace periods.

    Return its shell exit code and reason, or ``TIMEOUT_EXIT_CODE`` and a kill reason after timeout.
    """
    try:
        returncode = shell_exit_code(run_in_process_group(argv, bound, **popen_kwargs))
    except subprocess.TimeoutExpired:
        failure_reason = f"killed by the runner after {bound:.0f}s (exit code {TIMEOUT_EXIT_CODE})"
        return TIMEOUT_EXIT_CODE, failure_reason
    return returncode, f"ModuScope exited with code {returncode}"
