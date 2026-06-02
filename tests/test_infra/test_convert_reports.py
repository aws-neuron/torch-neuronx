"""Tests for ``convert_reports.py`` — the per-node JSON-to-CSV writer.

The runner generates per-node CSVs from pytest JSON reports, and the
aggregation pass groups those CSVs by ``test_class`` to produce the
summary CSV. A previous refactor stripped the ``test_class`` and
``test_method`` columns from the per-class CSVs, which made the
aggregator's ``row.get("test_class", "unknown")`` lookup return
``"unknown"`` for every row — bucketing all results under a single
``unknown`` row in the summary.

These tests guard against that round-trip regression: if anyone removes
``test_class`` from the CSV columns again, these assertions fire before
the bug ships.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from convert_reports import (
    extract_test_data_from_json,
    json_to_csv,
)


def _write_pytest_json(path: Path, tests: list[dict]) -> None:
    """Write a minimal pytest-json-report payload at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tests": tests}))


def _real_nodeid(test_class: str, test_method: str) -> str:
    """Match what real pytest emits for upstream tests."""
    return f"tests/pytorch_tests_native/test/test_ops.py::{test_class}::{test_method}"


def _make_test(test_class: str, test_method: str, outcome: str = "passed") -> dict:
    return {
        "nodeid": _real_nodeid(test_class, test_method),
        "outcome": outcome,
        "setup": {"duration": 0.0},
        "call": {"duration": 0.1},
        "teardown": {"duration": 0.0},
    }


# ---------------------------------------------------------------------------
# extract_test_data_from_json — class extraction must work
# ---------------------------------------------------------------------------
def test_extract_test_data_populates_test_class(tmp_path):
    # Real-world nodeid format with full path prefix. The aggregator
    # depends on test_class being non-empty for every row.
    json_path = tmp_path / "test_results_test_ops.json"
    _write_pytest_json(json_path, [_make_test("TestCommonNEURON", "test_add_neuron_float32")])

    rows = extract_test_data_from_json(str(json_path))
    assert len(rows) == 1
    assert rows[0]["test_class"] == "TestCommonNEURON"
    assert rows[0]["test_method"] == "test_add_neuron_float32"
    assert rows[0]["test_class"] != ""  # the previous regression


def test_extract_test_data_handles_two_part_nodeid(tmp_path):
    # When pytest emits a nodeid without a class (e.g. module-level test),
    # the extractor falls back to using the file stem as test_class.
    json_path = tmp_path / "x.json"
    payload = [
        {
            "nodeid": "tests/pytorch_tests_native/test/test_module_level.py::test_top_level",
            "outcome": "passed",
            "setup": {"duration": 0.0},
            "call": {"duration": 0.1},
            "teardown": {"duration": 0.0},
        }
    ]
    _write_pytest_json(json_path, payload)
    rows = extract_test_data_from_json(str(json_path))
    assert rows[0]["test_class"] == "test_module_level"
    assert rows[0]["test_method"] == "test_top_level"


# ---------------------------------------------------------------------------
# json_to_csv — CSV must contain test_class column (the regression)
# ---------------------------------------------------------------------------
def test_csv_round_trip_preserves_test_class_column(tmp_path):
    # The bug: json_to_csv used to strip "test_class" and "test_method"
    # from the fieldnames list, claiming they were redundant with test_id.
    # The downstream aggregator's row.get("test_class", "unknown") then
    # returned "unknown" for every row.
    #
    # This test fails if anyone reintroduces that strip.
    json_path = tmp_path / "test_results_TestX.json"
    csv_path = tmp_path / "test_results_TestX.csv"
    _write_pytest_json(
        json_path,
        [
            _make_test("TestCommonNEURON", "test_add_neuron_float32"),
            _make_test("TestFakeTensorNEURON", "test_mul_neuron_int64", outcome="skipped"),
        ],
    )
    assert json_to_csv(str(json_path), str(csv_path)) is True
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    # Critical regression guards.
    assert "test_class" in cols, (
        f"test_class column was stripped from CSV! Columns: {cols}. "
        "This causes the aggregator to bucket every row as 'unknown'."
    )
    assert "test_method" in cols, f"test_method column was stripped. Columns: {cols}"
    assert "nodeid" not in cols, "nodeid is internal-only and should be dropped"

    # Values are populated, not blank.
    classes = [row["test_class"] for row in rows]
    assert classes == ["TestCommonNEURON", "TestFakeTensorNEURON"]


def test_csv_aggregator_lookup_returns_real_class_not_unknown(tmp_path):
    # Mimics the aggregation pattern that caused the original "all unknown"
    # bug:
    #
    #     cls = row.get("test_class", "unknown")
    #     stats[cls][status] += 1
    #
    # If test_class is missing or empty, every row gets bucketed as
    # "unknown". This test fails if that ever happens again.
    json_path = tmp_path / "test_results_TestX.json"
    csv_path = tmp_path / "test_results_TestX.csv"
    _write_pytest_json(
        json_path,
        [_make_test("TestCommonNEURON", f"test_method_{i}") for i in range(5)]
        + [_make_test("TestFakeTensorNEURON", f"test_method_{i}") for i in range(3)],
    )
    json_to_csv(str(json_path), str(csv_path))

    from collections import Counter

    with open(csv_path) as f:
        classes = Counter(row.get("test_class", "unknown") for row in csv.DictReader(f))

    assert "unknown" not in classes, (
        f"Aggregator would bucket {classes.get('unknown', 0)} rows as 'unknown' — "
        "test_class is missing or empty in the CSV."
    )
    assert classes["TestCommonNEURON"] == 5
    assert classes["TestFakeTensorNEURON"] == 3


def test_csv_columns_match_extract_output(tmp_path):
    # Defensive: the CSV column set should be exactly what extract_test_data
    # produces, minus the internal-only "nodeid" column. Catches accidental
    # column-ordering changes that could break downstream consumers.
    json_path = tmp_path / "x.json"
    csv_path = tmp_path / "x.csv"
    _write_pytest_json(json_path, [_make_test("TestX", "test_y")])
    json_to_csv(str(json_path), str(csv_path))

    extracted = extract_test_data_from_json(str(json_path))[0]
    expected_cols = {k for k in extracted if k != "nodeid"}

    with open(csv_path) as f:
        actual_cols = set(csv.DictReader(f).fieldnames or [])

    assert actual_cols == expected_cols, (
        f"CSV columns drifted from extract_test_data_from_json output. "
        f"Expected (minus nodeid): {sorted(expected_cols)}, got: {sorted(actual_cols)}"
    )


# ---------------------------------------------------------------------------
# extract_test_data_from_json — failure-reason extraction across stages
# ---------------------------------------------------------------------------


def _failed_test_with_call_crash(message: str) -> dict:
    """Standard failure: pytest captured a crash in the call stage."""
    return {
        "nodeid": _real_nodeid("TestX", "test_y"),
        "outcome": "failed",
        "setup": {"duration": 0.0, "outcome": "passed"},
        "call": {
            "duration": 0.1,
            "outcome": "failed",
            "crash": {"path": "x.py", "lineno": 10, "message": message},
        },
        "teardown": {"duration": 0.0, "outcome": "passed"},
    }


def test_extract_reason_uses_call_crash_message(tmp_path):
    json_path = tmp_path / "x.json"
    _write_pytest_json(json_path, [_failed_test_with_call_crash("RuntimeError: boom\n  details")])

    rows = extract_test_data_from_json(str(json_path))
    assert rows[0]["failure_reason"] == "RuntimeError: boom\n  details"


def test_extract_reason_recovers_worker_crash_victim(tmp_path):
    """xdist-worker-crash victims have empty ``call`` and the diagnostic
    under a literal ``???`` key — common for NRT segfaults. Without this
    branch, dashboard categorization buckets miss them entirely."""
    json_path = tmp_path / "x.json"
    payload = [
        {
            "nodeid": _real_nodeid("TestCommonNEURON", "test_dtypes_constant_pad_nd_neuron"),
            "outcome": "failed",
            "setup": {"duration": 0.0, "outcome": "passed"},
            "call": {},
            "teardown": {},
            "???": {
                "duration": 0,
                "outcome": "failed",
                "longrepr": (
                    "[gw26] linux -- Python 3.12.3\n"
                    "worker 'gw26' crashed while running 'tests/x.py::T::a'"
                ),
            },
        }
    ]
    _write_pytest_json(json_path, payload)

    rows = extract_test_data_from_json(str(json_path))
    assert "worker 'gw26' crashed" in rows[0]["failure_reason"]


def test_extract_reason_falls_back_to_call_longrepr(tmp_path):
    """When ``call.crash`` is absent, use last non-empty line of ``call.longrepr``."""
    json_path = tmp_path / "x.json"
    payload = [
        {
            "nodeid": _real_nodeid("TestX", "test_y"),
            "outcome": "failed",
            "setup": {"duration": 0.0, "outcome": "passed"},
            "call": {
                "duration": 0.1,
                "outcome": "failed",
                "longrepr": "tests/x.py:42: in test_y\n    foo()\nE   AssertionError: nope",
            },
            "teardown": {"duration": 0.0, "outcome": "passed"},
        }
    ]
    _write_pytest_json(json_path, payload)

    rows = extract_test_data_from_json(str(json_path))
    assert "AssertionError: nope" in rows[0]["failure_reason"]


def test_extract_reason_recovers_setup_stage_crash(tmp_path):
    """Setup-stage failures get a ``[setup]`` prefix so dashboards can
    distinguish fixture errors from test-body failures."""
    json_path = tmp_path / "x.json"
    payload = [
        {
            "nodeid": _real_nodeid("TestX", "test_y"),
            "outcome": "failed",
            "setup": {
                "duration": 0.0,
                "outcome": "failed",
                "crash": {"path": "fix.py", "lineno": 5, "message": "FixtureError: missing"},
            },
            "call": {},
            "teardown": {},
        }
    ]
    _write_pytest_json(json_path, payload)

    reason = extract_test_data_from_json(str(json_path))[0]["failure_reason"]
    assert reason.startswith("[setup]") and "FixtureError" in reason


def test_extract_reason_passed_test_has_empty_reason(tmp_path):
    """Only failed/error/xfailed get a reason; everything else is empty
    so panels filtering on ``failure_reason != ''`` aren't polluted."""
    json_path = tmp_path / "x.json"
    payload = [
        _make_test("TestX", "test_pass", outcome="passed"),
        _make_test("TestX", "test_skip", outcome="skipped"),
        _make_test("TestX", "test_xpass", outcome="xpassed"),
    ]
    _write_pytest_json(json_path, payload)

    rows = extract_test_data_from_json(str(json_path))
    assert all(r["failure_reason"] == "" for r in rows)


def test_extract_reason_truncates_very_long_messages(tmp_path):
    """2000-char cap so a runaway longrepr can't bloat parquet files."""
    long_msg = "RuntimeError: " + ("x" * 5000)
    json_path = tmp_path / "x.json"
    _write_pytest_json(json_path, [_failed_test_with_call_crash(long_msg)])

    reason = extract_test_data_from_json(str(json_path))[0]["failure_reason"]
    assert len(reason) <= 2000
    assert reason.startswith("RuntimeError: ")
