#!/usr/bin/env python3
"""Merge-queue coverage gate runner for the vertical ATen op suites.

Run the eager and compile coverage suites under NEURON_COVERAGE_GATE=1, then
retry the failed ops once. This mirrors the upstream run_upstream_tests.py
--last-failed retry model: a failure counts only if it repeats on the retry.

NEURON_COVERAGE_GATE=1 turns each op test into an assertion. A regressed
supported op fails. A known gap (the allowlist) reports xfail. The retry pass
re-runs only the failed ops, so a flaky failure clears and a real regression
repeats. The retry exit code gates.

The gate hardcodes one shape per op and 64 workers by design: a fast,
deterministic merge signal. The caller does not tune these. The op tests are
single-core, so trn2 runs one worker per logical NeuronCore.

Usage: python testing/vertical/run_gate.py [csv_dir]
"""

from __future__ import annotations

import os
import subprocess
import sys

SUITES = [
    "tests/vertical/test_target_ops_coverage.py",
    "tests/vertical/test_target_ops_compile_coverage.py",
]
# --timeout=720 is the per-op backstop. compile_time_limit (600s) bounds the
# compile; the timeout bounds a hang in the exec loop or the correctness oracle,
# which nothing else caps. A hung op must not stall the merge queue. Use the
# signal method, not thread, and no --forked: --forked segfaults a worker that
# holds a live Neuron runtime (see run_upstream_tests.py).
GATE_OPTS = [
    "--max-shapes-per-op",
    "1",
    "-n",
    "64",
    "--timeout=720",
    "--timeout-method=signal",
]

# pytest exit code for "tests failed with a complete cache". Only this code is
# safe to funnel into a --last-failed retry.
PYTEST_TESTS_FAILED = 1
# pytest exit code for "no tests collected".
PYTEST_NO_TESTS = 5


def should_retry(main_exit: int) -> bool:
    """Return whether to run the --last-failed retry pass.

    Retry only on a clean per-test failure (pytest exit 1). A timeout or an
    internal, usage, or collection error leaves the --last-failed cache
    incomplete, so a retry replays nothing useful.
    """
    return main_exit == PYTEST_TESTS_FAILED


def resolve_exit(retry_exit: int) -> int:
    """Return the gate exit code from the retry-pass exit code.

    A retry that collects no test (exit 5) means every main-pass failure was
    flaky and cleared, so treat it as a pass. This is reachable only after a
    clean exit-1 main pass.
    """
    return 0 if retry_exit == PYTEST_NO_TESTS else retry_exit


def _run_pytest(report_flags: list[str], extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", *SUITES, *GATE_OPTS, *report_flags, *extra]
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main(argv: list[str]) -> int:
    csv_dir = argv[1] if len(argv) > 1 else "outputs/vertical_coverage"
    os.environ["NEURON_COVERAGE_GATE"] = "1"
    os.environ["NEURON_COVERAGE_CSV_DIR"] = csv_dir

    # Main pass. Records every op status and marks a regression as a failure.
    main_exit = _run_pytest(["--tb=no", "-q"], [])
    if main_exit == 0:
        print("Main pass clean: no regression, no retry.")
        return 0

    if not should_retry(main_exit):
        print(
            f"::error::Main pass exited {main_exit} (not a clean test failure). "
            "The run did not complete. Fail the gate."
        )
        return main_exit

    # Retry only the failed ops once. A flaky failure that clears here no longer
    # gates; a real regression reproduces and fails again.
    print("Main pass reported test failures. Retry the failed ops once.")
    retry_exit = _run_pytest(
        ["--tb=short", "-rf"], ["--last-failed", "--last-failed-no-failures=none"]
    )
    if retry_exit == PYTEST_NO_TESTS:
        print("Retry collected no test to re-run. Treat as pass.")
    return resolve_exit(retry_exit)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
