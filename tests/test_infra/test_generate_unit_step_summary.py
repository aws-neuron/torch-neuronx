"""Tests for ``generate_unit_step_summary.py`` — the unit-test job summary writer.

The tool scans ``<reports-dir>/json_reports`` for one pytest-json report per
suite, merges the ``*-retry.json`` outcome, sums the suites, and renders a
markdown panel. The tests cover:

  - ``_suite_reports``: report discovery, coverage/retry exclusion, retry
    pairing.
  - ``aggregate_reports``: summed counts across suites, per-suite breakdown,
    failure list with the suite name, retry promotion, empty-dir handling.
  - ``render_markdown``: panel structure, the "no report" fallback, pipe
    escaping.
  - ``write_summary``: the file write and the ``$GITHUB_STEP_SUMMARY`` append.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from generate_unit_step_summary import (
    _suite_reports,
    aggregate_reports,
    render_markdown,
    write_summary,
)


def _write_pytest_json(path: Path, tests: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tests": tests}))


def _entry(nodeid: str, outcome: str, *, duration: float = 0.1, crash: str | None = None) -> dict:
    entry: dict = {
        "nodeid": nodeid,
        "outcome": outcome,
        "setup": {"duration": 0.0},
        "call": {"duration": duration},
        "teardown": {"duration": 0.0},
    }
    if crash:
        entry["call"]["crash"] = {"message": crash}
    return entry


# ---------------------------------------------------------------------------
# _suite_reports
# ---------------------------------------------------------------------------
def test_suite_reports_excludes_coverage_and_pairs_retry(tmp_path):
    json_dir = tmp_path / "json_reports"
    _write_pytest_json(json_dir / "report.json", [])
    _write_pytest_json(json_dir / "report-retry.json", [])
    _write_pytest_json(json_dir / "vertical-report.json", [])
    _write_pytest_json(json_dir / "coverage.json", [])

    got = _suite_reports(json_dir)
    names = [suite for suite, _, _ in got]
    assert names == ["report", "vertical-report"]

    by_suite = {suite: (main, retry) for suite, main, retry in got}
    assert by_suite["report"][1] == json_dir / "report-retry.json"
    assert by_suite["vertical-report"][1] is None


# ---------------------------------------------------------------------------
# aggregate_reports
# ---------------------------------------------------------------------------
def test_aggregate_sums_across_suites(tmp_path):
    json_dir = tmp_path / "json_reports"
    _write_pytest_json(
        json_dir / "report.json",
        [
            _entry("t/a.py::test_a", "passed"),
            _entry("t/a.py::test_b", "failed", crash="boom-a"),
        ],
    )
    _write_pytest_json(
        json_dir / "vertical-report.json",
        [
            _entry("t/b.py::test_c", "passed"),
            _entry("t/b.py::test_d", "passed"),
        ],
    )

    summary = aggregate_reports(json_dir)
    assert summary is not None
    assert summary["counts"] == {"passed": 3, "failed": 1}
    assert summary["total"] == 4
    # (passed + xpassed) / (passed + xpassed + failed + xfailed + error) = 3/4.
    assert summary["pass_rate"] == "75.0%"
    assert {s["name"] for s in summary["suites"]} == {"report", "vertical-report"}

    failures = summary["failures"]
    assert len(failures) == 1
    assert failures[0]["suite"] == "report"
    assert failures[0]["test_id"] == "t/a.py::test_b"
    assert "boom-a" in failures[0]["failure_reason"]


def test_aggregate_retry_promotes_failed_to_passed(tmp_path):
    json_dir = tmp_path / "json_reports"
    _write_pytest_json(
        json_dir / "report.json", [_entry("t/a.py::test_x", "failed", crash="flake")]
    )
    _write_pytest_json(json_dir / "report-retry.json", [_entry("t/a.py::test_x", "passed")])

    summary = aggregate_reports(json_dir)
    assert summary is not None
    # Retry made it pass. It must not appear as a failure.
    assert summary["counts"].get("failed", 0) == 0
    assert summary["failures"] == []


def test_aggregate_empty_dir_returns_none(tmp_path):
    (tmp_path / "json_reports").mkdir()
    assert aggregate_reports(tmp_path / "json_reports") is None


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------
def test_render_no_report_fallback():
    md = render_markdown("ATen Unit Tests", None)
    assert "## ATen Unit Tests" in md
    assert "No pytest JSON report" in md


def test_render_has_totals_suites_and_failures(tmp_path):
    summary = {
        "counts": {"passed": 3, "failed": 1},
        "total": 4,
        "pass_rate": "75.0%",
        "duration": 12.3,
        "suites": [
            {"name": "report", "counts": {"passed": 1, "failed": 1}, "total": 2},
            {"name": "vertical-report", "counts": {"passed": 2}, "total": 2},
        ],
        "failures": [
            {
                "suite": "report",
                "test_id": "t/a.py::test_b",
                "status": "failed",
                "failure_reason": "boom | pipe",
            },
        ],
    }
    md = render_markdown("Unit Test Results", summary)
    assert "## Unit Test Results" in md
    assert "**Total:** 4" in md
    assert "**Pass rate:** 75.0%" in md
    assert "### Suites" in md
    # The failing suite sorts to the top of the suite table.
    suite_table = md.split("### Suites")[1]
    assert suite_table.index("| report |") < suite_table.index("| vertical-report |")
    assert "### Failures (1 total)" in md
    # The pipe in the error text is escaped so it does not break the cell.
    assert "boom \\| pipe" in md


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------
def test_write_summary_writes_file_and_github_summary(tmp_path, monkeypatch):
    gh_summary = tmp_path / "gh_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh_summary))
    reports_dir = tmp_path / "reports"

    write_summary("## panel\n", reports_dir)

    assert (reports_dir / "step_summary.md").read_text() == "## panel\n"
    assert gh_summary.read_text() == "## panel\n"


def test_write_summary_no_github_env_still_writes_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    reports_dir = tmp_path / "reports"

    write_summary("## panel\n", reports_dir)

    assert (reports_dir / "step_summary.md").read_text() == "## panel\n"
