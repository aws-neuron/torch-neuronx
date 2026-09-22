"""Regression tests for the retry-pass merge logic.

The ``run_upstream_tests.sh`` retry block writes ``*_retry.json`` reports
alongside the main pytest JSON reports. The aggregator merges them so that:

* tests that passed on retry no longer count as failures;
* worker-crash misclassified skip/xfail-marked tests surface with their
  true marker-based status instead of a spurious ``failed``.

These tests cover the ``apply_retry_outcomes`` helper (the core merge
rule), the per-node pairing done by ``convert_output_dir``, and the
failures-only CSV emitted by the aggregator. They run under pytest with
no Neuron-specific dependencies.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from aggregate_upstream_results import (
    FAILURE_STATUSES,
    _calc_pass_rate,
    flatten_reports,
    write_failures_only_csv,
    write_flaky_csv,
)
from convert_reports import (
    PASSED_ON_RETRY_STATUS,
    apply_retry_outcomes,
    convert_output_dir,
    extract_test_data_from_json,
)


def _main_row(status: str, nodeid: str = "test_x.py::C::x", reason: str = "") -> dict:
    return {
        "nodeid": nodeid,
        "test_id": nodeid.split("/test/")[-1] if "/test/" in nodeid else nodeid,
        "test_class": "C",
        "test_method": nodeid.split("::")[-1],
        "status": status,
        "duration(sec)": 0.01,
        "failure_reason": reason,
        "timestamp": "",
    }


def _retry_row(status: str, nodeid: str = "test_x.py::C::x", reason: str = "") -> dict:
    return {"nodeid": nodeid, "status": status, "failure_reason": reason}


def _write_pytest_json(path: Path, tests: list[tuple[str, str]]) -> None:
    """Write a minimal pytest-json-report payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": nodeid,
                        "outcome": outcome,
                        "setup": {"duration": 0.0},
                        "call": {"duration": 0.01},
                        "teardown": {"duration": 0.0},
                    }
                    for nodeid, outcome in tests
                ]
            }
        )
    )


# -----------------------------------------------------------------------------
# apply_retry_outcomes — golden matrix
# -----------------------------------------------------------------------------
# Every (main_status, retry_status) pair we care about → final_status.
GOLDEN_MATRIX = [
    # Flakiness: retry recovered a true pass
    ("failed", "passed", "passed_on_retry"),
    ("failed", "xpassed", "xpassed_on_retry"),
    ("error", "passed", "passed_on_retry"),
    ("error", "xpassed", "xpassed_on_retry"),
    # Worker-crash misclassification: retry shows the test's true marker
    ("failed", "skipped", "skipped_on_retry"),
    ("failed", "xfailed", "xfailed_on_retry"),
    ("error", "skipped", "skipped_on_retry"),
    ("error", "xfailed", "xfailed_on_retry"),
    # Genuine failure — retry confirms; main row kept unchanged
    ("failed", "failed", "failed"),
    ("failed", "error", "failed"),
    ("error", "failed", "error"),
    ("error", "error", "error"),
    # Retry must never override a non-failing main
    ("passed", "failed", "passed"),
    ("xpassed", "failed", "xpassed"),
    ("skipped", "failed", "skipped"),
    ("xfailed", "passed", "xfailed"),
]


@pytest.mark.parametrize(("main_status", "retry_status", "expected"), GOLDEN_MATRIX)
def test_apply_retry_outcomes_matrix(main_status: str, retry_status: str, expected: str) -> None:
    got = apply_retry_outcomes([_main_row(main_status)], [_retry_row(retry_status)])
    assert got[0]["status"] == expected


def test_apply_retry_outcomes_empty_retry_is_noop() -> None:
    main = [_main_row("failed", reason="boom")]
    assert apply_retry_outcomes(main, []) == main


def test_apply_retry_outcomes_missing_nodeid_leaves_main_unchanged() -> None:
    """Retry may re-run only a subset (e.g. one shard's failures)."""
    main = [_main_row("failed", "t::C::a"), _main_row("failed", "t::C::b")]
    retry = [_retry_row("passed", "t::C::a")]
    got = apply_retry_outcomes(main, retry)
    by = {r["nodeid"]: r["status"] for r in got}
    assert by == {"t::C::a": PASSED_ON_RETRY_STATUS, "t::C::b": "failed"}


def test_apply_retry_outcomes_preserves_original_reason_for_forensics() -> None:
    main = [_main_row("failed", reason="INTERNALERROR worker crashed")]
    got = apply_retry_outcomes(main, [_retry_row("passed")])
    assert "INTERNALERROR worker crashed" in got[0]["failure_reason"]
    assert "[retry passed]" in got[0]["failure_reason"]


def test_apply_retry_outcomes_does_not_mutate_inputs() -> None:
    main = [_main_row("failed", reason="orig")]
    original = dict(main[0])
    apply_retry_outcomes(main, [_retry_row("passed")])
    assert main[0] == original


def test_apply_retry_outcomes_last_retry_observation_wins() -> None:
    """Safety: retry JSON should not contain duplicate nodeids, but if it does,
    the last observation wins — deterministic."""
    main = [_main_row("failed")]
    retry = [_retry_row("failed"), _retry_row("passed")]
    assert apply_retry_outcomes(main, retry)[0]["status"] == "passed_on_retry"


# -----------------------------------------------------------------------------
# convert_output_dir — retry pairing and no double-counting
# -----------------------------------------------------------------------------
def test_convert_output_dir_merges_retry_sibling(tmp_path: Path) -> None:
    _write_pytest_json(
        tmp_path / "test_results_Foo.json",
        [("test_example.py::Foo::ok", "passed"), ("test_example.py::Foo::flake", "failed")],
    )
    _write_pytest_json(
        tmp_path / "test_results_Foo_retry.json",
        [("test_example.py::Foo::flake", "passed")],
    )

    assert convert_output_dir(str(tmp_path)) == 0

    # Reports land flat next to their source JSONs — no subdirectories.
    assert (tmp_path / "test_results_Foo.csv").exists()
    # Retry JSON must NOT produce its own CSV (would double-count).
    assert not (tmp_path / "test_results_Foo_retry.csv").exists()

    rows = list(csv.DictReader((tmp_path / "test_results_Foo.csv").open()))
    by = {r["test_id"]: r["status"] for r in rows}
    assert by == {
        "test_example.py::Foo::ok": "passed",
        "test_example.py::Foo::flake": "passed_on_retry",
    }
    # nodeid is internal — not emitted to CSV. test_class and test_method
    # ARE emitted (the aggregator groups by test_class).
    assert "nodeid" not in rows[0]
    assert "test_class" in rows[0]
    assert "test_method" in rows[0]


def test_convert_output_dir_handles_sharded_names(tmp_path: Path) -> None:
    for shard in (0, 1):
        _write_pytest_json(
            tmp_path / f"test_results_TestMeta_shard{shard}.json",
            [(f"test_example.py::TestMeta::s{shard}", "failed")],
        )
        _write_pytest_json(
            tmp_path / f"test_results_TestMeta_shard{shard}_retry.json",
            [(f"test_example.py::TestMeta::s{shard}", "passed")],
        )

    convert_output_dir(str(tmp_path))
    csvs = sorted(p.name for p in tmp_path.glob("test_results_*.csv"))
    assert csvs == [
        "test_results_TestMeta_shard0.csv",
        "test_results_TestMeta_shard1.csv",
    ]


def test_convert_output_dir_without_retry_is_unchanged(tmp_path: Path) -> None:
    """Backwards-compat: artifacts missing _retry.json must still aggregate."""
    _write_pytest_json(tmp_path / "test_results_X.json", [("test_example.py::X::a", "passed")])
    assert convert_output_dir(str(tmp_path)) == 0
    rows = list(csv.DictReader((tmp_path / "test_results_X.csv").open()))
    assert len(rows) == 1 and rows[0]["status"] == "passed"


def test_extract_test_data_includes_nodeid(tmp_path: Path) -> None:
    """Internal contract relied on by apply_retry_outcomes."""
    path = tmp_path / "r.json"
    _write_pytest_json(path, [("test_example.py::C::a", "passed")])
    rows = extract_test_data_from_json(str(path))
    assert rows[0]["nodeid"] == "test_example.py::C::a"


# -----------------------------------------------------------------------------
# Failures-only CSV + pass-rate
# -----------------------------------------------------------------------------
def test_write_failures_only_csv_excludes_recoveries(tmp_path: Path) -> None:
    """passed_on_retry and other *_on_retry statuses must not appear."""
    rows = [
        _main_row("passed", "t::C::ok"),
        _main_row("failed", "t::C::real_bug", reason="assert"),
        _main_row("passed_on_retry", "t::C::flake"),
        _main_row("skipped_on_retry", "t::C::worker_crash_skip"),
        _main_row("xfailed_on_retry", "t::C::worker_crash_xfail"),
        _main_row("error", "t::C::fixture_err", reason="fixture"),
    ]
    out = tmp_path / "failures.csv"
    assert write_failures_only_csv(rows, out) == 2

    written = list(csv.DictReader(out.open()))
    assert sorted(r["test_id"] for r in written) == ["t::C::fixture_err", "t::C::real_bug"]
    for r in written:
        assert r["status"] in FAILURE_STATUSES


def test_write_failures_only_csv_sort_is_stable(tmp_path: Path) -> None:
    """Sorted by (test_class, test_method) for diffable run-to-run output."""
    rows = [
        _main_row("failed", "t::Z::b"),
        _main_row("failed", "t::A::a"),
        _main_row("failed", "t::A::b"),
    ]
    # Overwrite test_class since _main_row hardcodes "C"
    rows[0]["test_class"] = "Z"
    rows[1]["test_class"] = "A"
    rows[2]["test_class"] = "A"

    out = tmp_path / "failures.csv"
    write_failures_only_csv(rows, out)
    got = [r["test_id"] for r in csv.DictReader(out.open())]
    assert got == ["t::A::a", "t::A::b", "t::Z::b"]


def test_write_failures_only_csv_writes_empty_file_when_no_failures(tmp_path: Path) -> None:
    """Downstream consumers rely on the file existing even when there are 0 rows."""
    out = tmp_path / "failures.csv"
    assert write_failures_only_csv([_main_row("passed"), _main_row("skipped")], out) == 0
    assert out.exists()
    assert "test_id,status" in out.read_text()


def test_write_flaky_csv_captures_all_on_retry_statuses(tmp_path: Path) -> None:
    """Flaky CSV captures every *_on_retry status and excludes everything else."""
    rows = [
        _main_row("passed", "t::C::ok"),
        _main_row("failed", "t::C::real_bug"),
        _main_row("passed_on_retry", "t::C::flake_pass"),
        _main_row("xpassed_on_retry", "t::C::flake_xpass"),
        _main_row("skipped_on_retry", "t::C::crash_skip"),
        _main_row("xfailed_on_retry", "t::C::crash_xfail"),
        _main_row("skipped", "t::C::normal_skip"),
        _main_row("xfailed", "t::C::normal_xfail"),
    ]
    out = tmp_path / "flaky.csv"
    n = write_flaky_csv(rows, out)
    assert n == 4

    written = list(csv.DictReader(out.open()))
    ids = sorted(r["test_id"] for r in written)
    assert ids == ["t::C::crash_skip", "t::C::crash_xfail", "t::C::flake_pass", "t::C::flake_xpass"]
    for r in written:
        assert r["status"].endswith("_on_retry")


def test_write_flaky_csv_empty_when_no_retries(tmp_path: Path) -> None:
    out = tmp_path / "flaky.csv"
    assert write_flaky_csv([_main_row("passed"), _main_row("failed")], out) == 0
    assert out.exists()


# -----------------------------------------------------------------------------
# flatten_reports
# -----------------------------------------------------------------------------
def test_flatten_reports_copies_all_types(tmp_path: Path) -> None:
    """Per-node artifacts (flat layout) are collected into central dirs per type."""
    # Two "nodes". Each per-node dir is flat — test_results_*.json /
    # <name>.html / test_results_*.csv / <name>.log live directly inside it.
    for node in ("test_autograd", "test_ops-shard0"):
        base = tmp_path / f"upstream-pytorch-{node}"
        base.mkdir()
        (base / f"test_results_{node}.json").write_text("{}")
        (base / f"{node}.html").write_text("<html/>")
        (base / f"test_results_{node}.csv").write_text("test_id,status\n")
        (base / f"{node}.log").write_text("log line")

    counts = flatten_reports(tmp_path)
    assert counts == {
        "json_reports": 2,
        "html_reports": 2,
        "csv_reports": 2,
        "log_reports": 2,
    }

    for sub in ("json_reports", "html_reports", "csv_reports", "log_reports"):
        files = sorted(p.name for p in (tmp_path / sub).iterdir())
        assert len(files) == 2, f"{sub}: {files}"


def test_flatten_reports_skips_duplicate_names(tmp_path: Path) -> None:
    """Duplicate filenames (shouldn't happen but...) are skipped, not overwritten."""
    for node in ("node_a", "node_b"):
        base = tmp_path / f"upstream-pytorch-{node}"
        base.mkdir()
        (base / "test_results_shared.json").write_text(f"from {node}")

    counts = flatten_reports(tmp_path)
    assert counts["json_reports"] == 1  # one copied, one skipped
    files = list((tmp_path / "json_reports").iterdir())
    assert len(files) == 1


def test_flatten_reports_empty_when_no_nodes(tmp_path: Path) -> None:
    counts = flatten_reports(tmp_path)
    assert counts == {
        "json_reports": 0,
        "html_reports": 0,
        "csv_reports": 0,
        "log_reports": 0,
    }
    # Central dirs still created (downstream consumers may check for them).
    for sub in ("json_reports", "html_reports", "csv_reports", "log_reports"):
        assert (tmp_path / sub).is_dir()


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"passed": 9, "failed": 1}, "90.0"),
        # passed_on_retry counts as pass — same rate
        ({"passed": 8, "passed_on_retry": 1, "failed": 1}, "90.0"),
        # xpassed_on_retry also counts as pass
        ({"passed": 8, "xpassed_on_retry": 1, "failed": 1}, "90.0"),
        ({"passed_on_retry": 10}, "100.0"),
        # skipped and skipped_on_retry excluded from denominator
        ({"skipped": 100}, "N/A"),
        ({"skipped_on_retry": 100}, "N/A"),
        ({"passed": 5, "skipped": 50, "skipped_on_retry": 45}, "100.0"),
        # xfailed and xfailed_on_retry count toward denom (not pass)
        ({"passed": 5, "xfailed": 5}, "50.0"),
        ({"passed": 5, "xfailed_on_retry": 5}, "50.0"),
        ({"passed": 5, "xfailed": 3, "xfailed_on_retry": 2}, "50.0"),
        ({"passed": 5, "error": 5}, "50.0"),
        ({}, "N/A"),
    ],
)
def test_calc_pass_rate(counts: dict, expected: str) -> None:
    assert _calc_pass_rate(counts) == expected


def test_pass_rate_end_to_end_with_partial_retry_recovery(tmp_path: Path) -> None:
    """8/10 pass in main, retry recovers 1 of 2 failures → 90% pass rate."""
    # Main: 8 passed + 2 failed
    _write_pytest_json(
        tmp_path / "test_results_Foo.json",
        [
            *[(f"test_x.py::Foo::pass_{i}", "passed") for i in range(8)],
            ("test_x.py::Foo::flake", "failed"),
            ("test_x.py::Foo::real_bug", "failed"),
        ],
    )
    # Retry: flake passes, real_bug still fails
    _write_pytest_json(
        tmp_path / "test_results_Foo_retry.json",
        [
            ("test_x.py::Foo::flake", "passed"),
            ("test_x.py::Foo::real_bug", "failed"),
        ],
    )

    convert_output_dir(str(tmp_path))

    rows = list(csv.DictReader((tmp_path / "test_results_Foo.csv").open()))
    from collections import Counter

    status_counts = Counter(r["status"] for r in rows)
    assert status_counts == {"passed": 8, "passed_on_retry": 1, "failed": 1}
    assert _calc_pass_rate(status_counts) == "90.0"
