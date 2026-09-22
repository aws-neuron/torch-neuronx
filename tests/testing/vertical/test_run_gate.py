"""Unit tests for the vertical merge-queue gate runner (exit-code policy)."""

from __future__ import annotations

import pytest
import run_gate


@pytest.mark.parametrize(
    "main_exit, expected",
    [
        (1, True),  # clean per-test failure: cache is complete, retry is useful
        (0, False),  # no failure: caller handles this before should_retry
        (2, False),  # interrupted: cache incomplete
        (3, False),  # internal error
        (4, False),  # usage error
        (5, False),  # no tests collected
        (124, False),  # timeout
        (137, False),  # SIGKILL
    ],
)
def test_should_retry_only_on_clean_test_failure(main_exit, expected):
    assert run_gate.should_retry(main_exit) is expected


@pytest.mark.parametrize(
    "retry_exit, expected",
    [
        (5, 0),  # retry collected nothing: every main failure was flaky -> pass
        (0, 0),  # retry passed
        (1, 1),  # regression reproduced -> gate fails
        (2, 2),  # retry itself broke -> surface the code
    ],
)
def test_resolve_exit(retry_exit, expected):
    assert run_gate.resolve_exit(retry_exit) == expected
