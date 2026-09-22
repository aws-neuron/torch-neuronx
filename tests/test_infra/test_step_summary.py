"""Tests for ``step_summary.py`` — per-node GitHub Actions summary writer.

Three layers tested in isolation:

  - ``compute_node_summary``: status counts after retry merge, failures list,
    pass-rate math, duration sum.
  - ``render_step_summary_markdown``: structure of the markdown output,
    headers, table rows, failure section, pipe escaping.
  - ``write_node_step_summary``: env-var gating (no-op when
    ``GITHUB_STEP_SUMMARY`` is unset), append behavior, exception swallow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from step_summary import (
    _MAX_FAILURES_IN_SUMMARY,
    compute_node_summary,
    render_step_summary_markdown,
    write_node_step_summary,
)


def _write_pytest_json(path: Path, tests: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tests": tests}))


def _test_entry(
    nodeid: str,
    outcome: str,
    *,
    duration: float = 0.1,
    crash_message: str | None = None,
) -> dict:
    entry: dict = {
        "nodeid": nodeid,
        "outcome": outcome,
        "setup": {"duration": 0.0},
        "call": {"duration": duration},
        "teardown": {"duration": 0.0},
    }
    if crash_message:
        entry["call"]["crash"] = {"message": crash_message}
    return entry


# ---------------------------------------------------------------------------
# compute_node_summary
# ---------------------------------------------------------------------------
def test_compute_summary_basic_counts(tmp_path):
    main = tmp_path / "test_results_X.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_a", "passed"),
            _test_entry("t/x.py::TestX::test_b", "passed"),
            _test_entry("t/x.py::TestX::test_c", "failed", crash_message="boom"),
            _test_entry("t/x.py::TestX::test_d", "skipped"),
            _test_entry("t/x.py::TestX::test_e", "xfailed"),
            _test_entry("t/x.py::TestX::test_f", "xpassed"),
        ],
    )
    summary = compute_node_summary(main, retry_report=None)
    assert summary is not None
    assert summary["counts"] == {
        "passed": 2,
        "failed": 1,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
    }
    assert summary["total"] == 6
    # Pass rate = (passed + xpassed) / (passed + xpassed + failed + xfailed + error)
    # = (2 + 1) / (2 + 1 + 1 + 1) = 3/5 = 60%
    assert summary["pass_rate"] == "60.0%"


def test_compute_summary_missing_main_returns_none(tmp_path):
    assert compute_node_summary(tmp_path / "nope.json", retry_report=None) is None


def test_compute_summary_empty_main_returns_none(tmp_path):
    p = tmp_path / "empty.json"
    _write_pytest_json(p, [])
    assert compute_node_summary(p, retry_report=None) is None


def test_compute_summary_retry_promotes_failed_to_passed(tmp_path):
    # The key retry semantics test: a test that failed in main but passed
    # on retry must be counted as `passed` in the per-node totals (even
    # though the row's status becomes `passed_on_retry` for the CSV).
    main = tmp_path / "main.json"
    retry = tmp_path / "main_retry.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_a", "passed"),
            _test_entry("t/x.py::TestX::test_b", "failed", crash_message="flake"),
        ],
    )
    _write_pytest_json(retry, [_test_entry("t/x.py::TestX::test_b", "passed")])

    summary = compute_node_summary(main, retry_report=retry)
    assert summary is not None
    # test_b is now counted as passed, not failed.
    assert summary["counts"] == {"passed": 2}
    assert summary["pass_rate"] == "100.0%"
    # And no failure shows up in the failures list.
    assert summary["failures"] == []


def test_compute_summary_retry_skipped_counts_as_skipped(tmp_path):
    # Retry semantics: failed-in-main, skipped-on-retry → counts as skipped.
    main = tmp_path / "main.json"
    retry = tmp_path / "main_retry.json"
    _write_pytest_json(main, [_test_entry("t/x.py::TestX::test_a", "failed", crash_message="x")])
    _write_pytest_json(retry, [_test_entry("t/x.py::TestX::test_a", "skipped")])

    summary = compute_node_summary(main, retry_report=retry)
    assert summary["counts"] == {"skipped": 1}
    # No failure.
    assert summary["failures"] == []


def test_compute_summary_retry_xfailed_counts_as_xfailed(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "main_retry.json"
    _write_pytest_json(main, [_test_entry("t/x.py::TestX::test_a", "failed", crash_message="x")])
    _write_pytest_json(retry, [_test_entry("t/x.py::TestX::test_a", "xfailed")])

    summary = compute_node_summary(main, retry_report=retry)
    assert summary["counts"] == {"xfailed": 1}
    assert summary["failures"] == []


def test_compute_summary_retry_xpassed_counts_as_xpassed(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "main_retry.json"
    _write_pytest_json(main, [_test_entry("t/x.py::TestX::test_a", "failed", crash_message="x")])
    _write_pytest_json(retry, [_test_entry("t/x.py::TestX::test_a", "xpassed")])

    summary = compute_node_summary(main, retry_report=retry)
    assert summary["counts"] == {"xpassed": 1}
    assert summary["failures"] == []


def test_compute_summary_retry_still_failing_kept_as_failure(tmp_path):
    # Genuine failure: failed in main, failed in retry. apply_retry_outcomes
    # leaves the row's status as "failed" (not promoted), so it should
    # still appear in the failures list with its original error message.
    main = tmp_path / "main.json"
    retry = tmp_path / "main_retry.json"
    _write_pytest_json(
        main,
        [_test_entry("t/x.py::TestX::test_a", "failed", crash_message="real bug")],
    )
    _write_pytest_json(
        retry,
        [_test_entry("t/x.py::TestX::test_a", "failed", crash_message="still broken")],
    )

    summary = compute_node_summary(main, retry_report=retry)
    assert summary["counts"] == {"failed": 1}
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["status"] == "failed"
    # Retains the original main-run failure_reason (apply_retry_outcomes
    # only promotes; it doesn't overwrite genuine failures).
    assert "real bug" in summary["failures"][0]["failure_reason"]


def test_compute_summary_failures_list_includes_test_id_and_reason(tmp_path):
    main = tmp_path / "x.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_pass", "passed"),
            _test_entry(
                "t/x.py::TestX::test_fail",
                "failed",
                crash_message="AssertionError: something broke",
            ),
        ],
    )
    summary = compute_node_summary(main, retry_report=None)
    assert len(summary["failures"]) == 1
    f = summary["failures"][0]
    assert "test_fail" in f["test_id"]
    assert f["status"] == "failed"
    assert "something broke" in f["failure_reason"]


def test_compute_summary_duration_sums_across_phases(tmp_path):
    main = tmp_path / "x.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_a", "passed", duration=0.5),
            _test_entry("t/x.py::TestX::test_b", "passed", duration=1.0),
        ],
    )
    summary = compute_node_summary(main, retry_report=None)
    # extract_test_data_from_json sums setup + call + teardown; for our
    # fixtures only `call` has a non-zero duration.
    assert summary["duration"] == pytest.approx(1.5, abs=0.01)


def test_compute_summary_pass_rate_with_only_skipped_is_na(tmp_path):
    # All-skipped suite: denominator is 0. Pass rate should be N/A.
    main = tmp_path / "x.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_a", "skipped"),
            _test_entry("t/x.py::TestX::test_b", "skipped"),
        ],
    )
    summary = compute_node_summary(main, retry_report=None)
    assert summary["pass_rate"] == "N/A"


# ---------------------------------------------------------------------------
# render_step_summary_markdown
# ---------------------------------------------------------------------------
def test_render_markdown_no_summary_shows_no_report_message():
    md = render_step_summary_markdown(["test_ops.py"], summary=None)
    assert "## PyTorch Upstream — test_ops.py" in md
    assert "No pytest JSON report" in md


def test_render_markdown_basic_structure():
    summary = {
        "counts": {"passed": 100, "failed": 3, "skipped": 5},
        "total": 108,
        "pass_rate": "97.1%",
        "duration": 1842.3,
        "failures": [],
    }
    md = render_step_summary_markdown(["test_ops.py"], summary)
    # Has the header.
    assert "## PyTorch Upstream — test_ops.py" in md
    # Stats line.
    assert "**Total:** 108" in md
    assert "**Pass rate:** 97.1%" in md
    assert "**Duration:** 1842.3s" in md
    # Status table.
    assert "| passed | failed | skipped" in md
    assert "| 100 | 3 | 5 | 0 | 0 | 0 |" in md
    # No failures section because list is empty.
    assert "### Failures" not in md


def test_render_markdown_includes_failures_section():
    summary = {
        "counts": {"passed": 1, "failed": 2},
        "total": 3,
        "pass_rate": "33.3%",
        "duration": 12.0,
        "failures": [
            {
                "test_id": "test_ops.py::TestOps::test_add_neuron_float32",
                "status": "failed",
                "failure_reason": "RuntimeError: NRT init failed",
            },
            {
                "test_id": "test_ops.py::TestOps::test_mul_neuron_int64",
                "status": "error",
                "failure_reason": "Timeout >800.0s",
            },
        ],
    }
    md = render_step_summary_markdown(["test_ops.py"], summary)
    assert "### Failures (2 total)" in md
    assert "| test_id | status | error |" in md
    assert "test_add_neuron_float32" in md
    assert "RuntimeError: NRT init failed" in md
    assert "Timeout >800.0s" in md


def _make_failures(n: int) -> list[dict]:
    return [
        {
            "test_id": f"test_ops.py::TestOps::test_{i}",
            "status": "failed",
            "failure_reason": f"err {i}",
        }
        for i in range(n)
    ]


def _summary_with_failures(failures: list[dict]) -> dict:
    return {
        "counts": {"failed": len(failures)},
        "total": len(failures),
        "pass_rate": "0.0%",
        "duration": 0.0,
        "failures": failures,
    }


def test_render_markdown_renders_all_rows_below_cap():
    # Below the cap, every failure must be in the rendered table — no
    # silent truncation. Guards against an off-by-one accidentally hiding
    # the last row.
    n = _MAX_FAILURES_IN_SUMMARY - 5
    md = render_step_summary_markdown(["test_ops.py"], _summary_with_failures(_make_failures(n)))
    assert f"### Failures ({n} total)" in md
    assert "_Showing first" not in md  # no truncation note
    assert md.count("test_ops.py::TestOps::test_") == n


def test_render_markdown_renders_all_rows_at_exact_cap():
    # At exactly the cap, all rows are rendered without the truncation
    # note (truncation triggers on `> cap`, not `>=`). Pinning this
    # behavior so future tweaks don't change it accidentally.
    n = _MAX_FAILURES_IN_SUMMARY
    md = render_step_summary_markdown(["test_ops.py"], _summary_with_failures(_make_failures(n)))
    assert f"### Failures ({n} total)" in md
    assert "_Showing first" not in md
    assert md.count("test_ops.py::TestOps::test_") == n


def test_render_markdown_truncates_one_above_cap():
    # First case where truncation kicks in: cap + 1.
    n = _MAX_FAILURES_IN_SUMMARY + 1
    md = render_step_summary_markdown(["test_ops.py"], _summary_with_failures(_make_failures(n)))
    assert f"### Failures ({n} total)" in md
    assert f"_Showing first {_MAX_FAILURES_IN_SUMMARY};" in md
    assert md.count("test_ops.py::TestOps::test_") == _MAX_FAILURES_IN_SUMMARY


def test_render_markdown_truncates_huge_failure_lists():
    n = _MAX_FAILURES_IN_SUMMARY + 25
    md = render_step_summary_markdown(["test_ops.py"], _summary_with_failures(_make_failures(n)))
    assert f"### Failures ({n} total)" in md
    assert f"_Showing first {_MAX_FAILURES_IN_SUMMARY};" in md
    # The truncation note appears once; only the first N rows appear.
    assert md.count("test_ops.py::TestOps::test_") == _MAX_FAILURES_IN_SUMMARY


def test_render_markdown_escapes_pipes_in_test_id_and_error():
    # Pipes in cell content would break the markdown table grid. They
    # must be backslash-escaped.
    summary = {
        "counts": {"failed": 1},
        "total": 1,
        "pass_rate": "0.0%",
        "duration": 0.0,
        "failures": [
            {
                "test_id": "weird::test|with|pipes",
                "status": "failed",
                "failure_reason": "stderr | stdout dump",
            }
        ],
    }
    md = render_step_summary_markdown(["test_x.py"], summary)
    assert "test\\|with\\|pipes" in md
    assert "stderr \\| stdout dump" in md


def test_render_markdown_first_line_only_for_multiline_errors():
    # Failures with multi-line error messages keep the first line only,
    # because markdown table cells can't span newlines.
    summary = {
        "counts": {"failed": 1},
        "total": 1,
        "pass_rate": "0.0%",
        "duration": 0.0,
        "failures": [
            {
                "test_id": "t::test",
                "status": "failed",
                "failure_reason": "first line\nsecond line\nthird",
            }
        ],
    }
    md = render_step_summary_markdown(["t.py"], summary)
    assert "first line" in md
    assert "second line" not in md
    assert "third" not in md


# ---------------------------------------------------------------------------
# write_node_step_summary
# ---------------------------------------------------------------------------
def test_write_step_summary_no_op_when_env_unset_and_no_output_dir(tmp_path, monkeypatch):
    # Outside CI with no output_dir, both paths are inactive — helper
    # silently no-ops so local invocations don't crash.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    main = tmp_path / "x.json"
    _write_pytest_json(main, [_test_entry("t::test_a", "passed")])
    # Doesn't raise; doesn't write anywhere.
    write_node_step_summary(["t.py"], main, None)


def test_write_step_summary_persists_to_artifact_dir_when_env_unset(tmp_path, monkeypatch):
    # Inside the test workload GITHUB_STEP_SUMMARY isn't set, but we
    # still want the markdown captured in the per-node artifact so the
    # GH-runner-side step can pick it up.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    artifact_dir = tmp_path / "outputs"
    main = tmp_path / "x.json"
    _write_pytest_json(main, [_test_entry("t::test_a", "passed")])

    write_node_step_summary(["t.py"], main, None, output_dir=artifact_dir)

    summary_path = artifact_dir / "step_summary.md"
    assert summary_path.is_file()
    content = summary_path.read_text()
    assert "## PyTorch Upstream — t.py" in content


def test_write_step_summary_writes_to_both_when_env_set_and_output_dir_given(tmp_path, monkeypatch):
    # On hosts that DO have the env var set (e.g. ubuntu-latest aggregator
    # job, local dev), the helper writes both places — append directly AND
    # persist the file.
    env_summary = tmp_path / "env_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(env_summary))
    artifact_dir = tmp_path / "outputs"
    main = tmp_path / "x.json"
    _write_pytest_json(main, [_test_entry("t::test_a", "passed")])

    write_node_step_summary(["t.py"], main, None, output_dir=artifact_dir)

    assert env_summary.is_file()
    assert (artifact_dir / "step_summary.md").is_file()
    # Same content in both places.
    assert env_summary.read_text() == (artifact_dir / "step_summary.md").read_text()


def test_write_step_summary_appends_to_env_file(tmp_path, monkeypatch):
    summary_file = tmp_path / "step_summary.md"
    summary_file.write_text("preexisting\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    main = tmp_path / "x.json"
    _write_pytest_json(
        main,
        [
            _test_entry("t/x.py::TestX::test_a", "passed"),
            _test_entry("t/x.py::TestX::test_b", "failed", crash_message="oops"),
        ],
    )

    write_node_step_summary(["test_x.py"], main, None)

    content = summary_file.read_text()
    # Preserves what was already there (append, not overwrite).
    assert content.startswith("preexisting\n")
    # Added our markdown.
    assert "## PyTorch Upstream — test_x.py" in content
    assert "| 1 | 1 |" in content  # passed=1, failed=1


def test_write_step_summary_handles_missing_main_gracefully(tmp_path, monkeypatch):
    # Pytest crashed before producing a JSON. We still want a panel
    # message rather than nothing or a stack trace.
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    write_node_step_summary(["test_x.py"], tmp_path / "does_not_exist.json", None)

    content = summary_file.read_text()
    assert "## PyTorch Upstream — test_x.py" in content
    assert "No pytest JSON report" in content


def test_write_step_summary_swallows_unwritable_env_file(tmp_path, monkeypatch, capsys):
    # If the env var points at an unwritable location, the helper must
    # not crash — post-process is non-fatal by convention.
    bad_main = tmp_path / "corrupt.json"
    bad_main.write_text("{not valid json")

    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/nonexistent_dir/summary.md")
    # No output_dir — only the env-var path runs and fails.
    write_node_step_summary(["test_x.py"], bad_main, None)

    captured = capsys.readouterr()
    # The helper logs a warning instead of raising.
    assert "WARNING: failed to write GITHUB_STEP_SUMMARY" in captured.out


def test_write_step_summary_with_retry_report(tmp_path, monkeypatch):
    # End-to-end with retry: failed in main + passed in retry → totals
    # show 1 passed, 0 failed. Failure list is empty.
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    main = tmp_path / "x.json"
    retry = tmp_path / "x_retry.json"
    _write_pytest_json(main, [_test_entry("t::test_flake", "failed", crash_message="x")])
    _write_pytest_json(retry, [_test_entry("t::test_flake", "passed")])

    write_node_step_summary(["test_x.py"], main, retry)
    content = summary_file.read_text()
    # 1 passed (promoted from failed_on_retry). Failure section absent.
    assert "| 1 | 0 | 0 | 0 | 0 | 0 |" in content
    assert "### Failures" not in content
