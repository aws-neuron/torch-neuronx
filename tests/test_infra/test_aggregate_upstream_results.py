"""Regression tests for ``aggregate_upstream_results``.

The aggregator globs pytest JSON reports directly inside each per-node
artifact directory (``upstream-pytorch-*/``) — the runner writes reports
flat so no nested ``json_reports/`` or ``outputs/`` subdirs are expected.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from aggregate_upstream_results import (
    _get_run_metadata,
    aggregate_summary,
    flatten_reports,
    write_github_summary,
    write_run_metadata_csv,
)


def _write_pytest_json(path: Path, test_class: str, test_methods: list[tuple[str, str]]) -> None:
    """Write a minimal pytest-json-report payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tests = []
    for method, outcome in test_methods:
        tests.append(
            {
                "nodeid": f"test_example.py::{test_class}::{method}",
                "outcome": outcome,
                "setup": {"duration": 0.0},
                "call": {"duration": 0.1},
                "teardown": {"duration": 0.0},
            }
        )
    path.write_text(json.dumps({"tests": tests}))


def _node_json(base: Path, artifact: str, filename: str) -> Path:
    return base / artifact / filename


def test_single_artifact_counted_once(tmp_path: Path) -> None:
    """Flat per-node layout: exactly one report per artifact, no double-counting."""
    _write_pytest_json(
        _node_json(tmp_path, "upstream-pytorch-test_autograd", "test_results_test_autograd.json"),
        test_class="TestAutograd",
        test_methods=[("test_add", "passed"), ("test_mul", "passed")],
    )

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    assert result["total_stats"]["passed"] == 2
    assert len(result["test_rows"]) == 2


def test_two_distinct_artifacts_count_independently(tmp_path: Path) -> None:
    """Each per-node artifact contributes its own tests."""
    for node in ("upstream-pytorch-test_autograd", "upstream-pytorch-test_nn"):
        _write_pytest_json(
            _node_json(tmp_path, node, "test_results_stub.json"),
            test_class=f"Test_{node}",
            test_methods=[("test_one", "passed")],
        )

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    assert result["total_stats"]["passed"] == 2


def test_sharded_artifacts_count_all_shards(tmp_path: Path) -> None:
    """Sharded filenames are all counted."""
    node = "upstream-pytorch-test_ops-shard0"
    for shard in range(3):
        _write_pytest_json(
            _node_json(tmp_path, node, f"test_results_test_ops_shard{shard}.json"),
            test_class="TestOps",
            test_methods=[(f"test_shard{shard}", "passed")],
        )

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    assert result["total_stats"]["passed"] == 3


def test_nested_extract_layout_counted(tmp_path: Path) -> None:
    """Internal compute extracts each per-node tar into ``test-results/``,
    so the per-node JSON path is
    ``upstream-pytorch-X/outputs/upstream_pytorch_artifacts/test_results_X.json``
    rather than the flat ``upstream-pytorch-X/test_results_X.json``.
    """
    nested = (
        tmp_path
        / "upstream-pytorch-test_ops"
        / "outputs"
        / "upstream_pytorch_artifacts"
        / "test_results_test_ops.json"
    )
    _write_pytest_json(
        nested,
        test_class="TestOps",
        test_methods=[("test_a", "passed"), ("test_b", "failed")],
    )

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    assert result["total_stats"]["passed"] == 1
    assert result["total_stats"]["failed"] == 1


def test_dedupe_outputs_and_test_results_mirror(tmp_path: Path) -> None:
    """Some runs leave the same JSON file in both
    ``outputs/upstream_pytorch_artifacts/`` AND a duplicate ``test-results/``
    mirror inside the per-node artifact. The aggregator must dedupe so the
    same node's tests aren't counted twice.
    """
    node = tmp_path / "upstream-pytorch-test_ops"
    primary = node / "outputs" / "upstream_pytorch_artifacts" / "test_results_test_ops.json"
    mirror = (
        node
        / "test-results"
        / "outputs"
        / "upstream_pytorch_artifacts"
        / "test_results_test_ops.json"
    )
    for path in (primary, mirror):
        _write_pytest_json(path, test_class="TestOps", test_methods=[("test_a", "passed")])

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    # If dedupe failed we'd see passed=2 (counted twice).
    assert result["total_stats"]["passed"] == 1


def test_pass_rate_correct(tmp_path: Path) -> None:
    """Summary CSV pass rate is computed correctly."""
    artifact = "upstream-pytorch-test_example"
    _write_pytest_json(
        _node_json(tmp_path, artifact, "test_results_test_example.json"),
        test_class="TestExample",
        test_methods=[
            ("test_pass", "passed"),
            ("test_fail", "failed"),
            ("test_skip", "skipped"),
        ],
    )

    summary_path = tmp_path / "summary.csv"
    aggregate_summary(tmp_path, summary_path)

    rows = list(csv.DictReader(summary_path.open()))
    total = next(r for r in rows if r["test_class"] == "TOTAL")
    assert total["pass_rate(%)"] == "50.0"
    assert int(total["passed"]) == 1
    assert int(total["failed"]) == 1
    assert int(total["skipped"]) == 1


# ---------------------------------------------------------------------------
# File-level grouping for the GH panel — option 2: file in panel, class in CSV
# ---------------------------------------------------------------------------
def _write_pytest_json_full(path: Path, tests: list[dict]) -> None:
    """Write a pytest payload using full nodeids (path::Class::method)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tests": tests}))


def _entry(nodeid: str, outcome: str, duration: float = 0.1) -> dict:
    return {
        "nodeid": nodeid,
        "outcome": outcome,
        "setup": {"duration": 0.0},
        "call": {"duration": duration},
        "teardown": {"duration": 0.0},
    }


def test_aggregate_summary_returns_file_stats(tmp_path: Path) -> None:
    """aggregate_summary returns parallel file_stats / file_durations dicts
    keyed by test file (extracted from test_id) for the GH panel render."""
    _write_pytest_json_full(
        tmp_path / "upstream-pytorch-test_ops" / "test_results_test_ops.json",
        [
            _entry("tests/x/test/test_ops.py::TestA::test_1", "passed"),
            _entry("tests/x/test/test_ops.py::TestA::test_2", "failed"),
            _entry("tests/x/test/test_ops.py::TestB::test_3", "passed"),
        ],
    )
    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")

    # Class-level (canonical, used by CSV) preserves both classes.
    assert "TestA" in result["stats"]
    assert "TestB" in result["stats"]
    # File-level (panel) collapses to one row.
    assert list(result["file_stats"].keys()) == ["test_ops.py"]
    assert result["file_stats"]["test_ops.py"]["passed"] == 2
    assert result["file_stats"]["test_ops.py"]["failed"] == 1


def test_aggregate_summary_collapses_sharded_files(tmp_path: Path) -> None:
    """Sharded files (test_ops.py-shard0, test_ops.py-shard1, ...) all
    use the same nodeid — they collapse to a single file row."""
    for shard in range(3):
        _write_pytest_json_full(
            tmp_path
            / f"upstream-pytorch-test_ops-shard{shard}"
            / f"test_results_test_ops_shard{shard}.json",
            [
                _entry(f"tests/x/test/test_ops.py::TestX::shard{shard}_a", "passed"),
                _entry(f"tests/x/test/test_ops.py::TestX::shard{shard}_b", "passed"),
            ],
        )
    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")

    # 6 tests total, all under one file row.
    assert list(result["file_stats"].keys()) == ["test_ops.py"]
    assert result["file_stats"]["test_ops.py"]["passed"] == 6


def test_aggregate_summary_splits_multi_file_node(tmp_path: Path) -> None:
    """A bundled matrix node (e.g. test_autograd + 19 more) yields one
    file-level row per file — no collapsing across files."""
    _write_pytest_json_full(
        tmp_path
        / "upstream-pytorch-test_autograd_and_2_more"
        / "test_results_test_autograd_and_2_more.json",
        [
            _entry("tests/x/test/test_autograd.py::TestX::test_1", "passed"),
            _entry("tests/x/test/test_indexing.py::TestY::test_2", "passed"),
            _entry("tests/x/test/test_view_ops.py::TestZ::test_3", "failed"),
        ],
    )
    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")

    assert set(result["file_stats"].keys()) == {
        "test_autograd.py",
        "test_indexing.py",
        "test_view_ops.py",
    }
    assert result["file_stats"]["test_view_ops.py"]["failed"] == 1


def test_aggregate_summary_preserves_subdir_in_file_key(tmp_path: Path) -> None:
    """For tests under a subdirectory (e.g. dynamo/test_x.py),
    the file key keeps the subdir prefix so two test_x.py files in
    different subdirs don't collide.
    """
    _write_pytest_json_full(
        tmp_path / "upstream-pytorch-dynamo" / "test_results_dynamo.json",
        [
            _entry("tests/x/test/dynamo/test_compile.py::TestC::test_a", "passed"),
            _entry("tests/x/test/dynamo/test_misc.py::TestM::test_b", "passed"),
        ],
    )
    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")

    assert "dynamo/test_compile.py" in result["file_stats"]
    assert "dynamo/test_misc.py" in result["file_stats"]


def test_summary_csv_remains_class_keyed(tmp_path: Path) -> None:
    """Regression: the CSV stays class-keyed (option 2 — file in panel,
    class in CSV) so dashboards keep their per-class grain."""
    _write_pytest_json_full(
        tmp_path / "upstream-pytorch-x" / "test_results_x.json",
        [
            _entry("tests/x/test/test_ops.py::TestA::a", "passed"),
            _entry("tests/x/test/test_ops.py::TestB::b", "passed"),
        ],
    )
    summary_csv = tmp_path / "summary.csv"
    aggregate_summary(tmp_path, summary_csv)
    rows = list(csv.DictReader(summary_csv.open()))
    classes = {r["test_class"] for r in rows if r["test_class"] != "TOTAL"}
    assert classes == {"TestA", "TestB"}


def test_write_github_summary_renders_file_level_when_file_stats_present(tmp_path, monkeypatch):
    """Smoke: when file_stats is populated, the panel header reads
    ``test_file`` and the rows are file paths, not class names."""
    target = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))

    result = _summary_result(
        # Class-level data (would render in CSV).
        stats={"TestA": {"passed": 5}, "TestB": {"passed": 3}},
        # File-level data (used by the panel).
        file_stats={"test_x.py": {"passed": 8}},
    )
    write_github_summary(result)

    content = target.read_text()
    assert "| test_file |" in content  # column header
    assert "test_x.py" in content
    # Class names should NOT appear — option 2 uses file-level rendering.
    assert "TestA" not in content
    assert "TestB" not in content


# ---------------------------------------------------------------------------
# flatten_reports — gathers per-node artifacts into central <type>/ dirs and
# removes the now-duplicate per-node dirs to keep the bundle artifact small.
# ---------------------------------------------------------------------------
def _make_node(base: Path, label: str) -> Path:
    """Create a per-node artifact dir with a JSON, HTML, CSV, and log."""
    node = base / f"upstream-pytorch-{label}"
    node.mkdir(parents=True, exist_ok=True)
    (node / f"test_results_{label}.json").write_text("{}")
    (node / f"{label}.html").write_text("<html></html>")
    (node / f"test_results_{label}.csv").write_text("col\nval\n")
    (node / f"{label}.log").write_text("log line")
    return node


def test_flatten_reports_groups_files_into_type_dirs(tmp_path: Path) -> None:
    _make_node(tmp_path, "test_ops")
    _make_node(tmp_path, "test_nn")

    counts = flatten_reports(tmp_path)

    # Every type directory exists with both nodes' files.
    assert (tmp_path / "json_reports" / "test_results_test_ops.json").is_file()
    assert (tmp_path / "json_reports" / "test_results_test_nn.json").is_file()
    assert (tmp_path / "html_reports" / "test_ops.html").is_file()
    assert (tmp_path / "html_reports" / "test_nn.html").is_file()
    assert (tmp_path / "csv_reports" / "test_results_test_ops.csv").is_file()
    assert (tmp_path / "csv_reports" / "test_results_test_nn.csv").is_file()
    assert (tmp_path / "log_reports" / "test_ops.log").is_file()
    assert (tmp_path / "log_reports" / "test_nn.log").is_file()

    # Counts reflect actual flatten work.
    assert counts == {
        "json_reports": 2,
        "html_reports": 2,
        "csv_reports": 2,
        "log_reports": 2,
    }


def test_flatten_reports_removes_per_node_dirs(tmp_path: Path) -> None:
    # Per-node dirs are pure duplication after flatten — removing them
    # halves the upstream-pytorch-all bundle artifact size. This test
    # guards against the cleanup being reverted.
    _make_node(tmp_path, "test_ops")
    _make_node(tmp_path, "test_nn")
    assert (tmp_path / "upstream-pytorch-test_ops").is_dir()
    assert (tmp_path / "upstream-pytorch-test_nn").is_dir()

    flatten_reports(tmp_path)

    # Per-node dirs gone…
    assert not (tmp_path / "upstream-pytorch-test_ops").exists()
    assert not (tmp_path / "upstream-pytorch-test_nn").exists()
    # …but central <type>/ dirs remain.
    assert (tmp_path / "json_reports").is_dir()
    assert (tmp_path / "html_reports").is_dir()
    assert (tmp_path / "csv_reports").is_dir()


def test_flatten_reports_finds_nested_extract_layout(tmp_path: Path) -> None:
    """Regression: extracted per-node artifacts may have files nested
    under ``test-results/outputs/upstream_pytorch_artifacts/`` rather
    than flat at the artifact root. flatten_reports must use a recursive
    glob to find them; otherwise the central <type>/ dirs end up empty.
    """
    nested = (
        tmp_path
        / "upstream-pytorch-test_ops"
        / "test-results"
        / "outputs"
        / "upstream_pytorch_artifacts"
    )
    nested.mkdir(parents=True)
    (nested / "test_results_test_ops.json").write_text("{}")
    (nested / "test_ops.html").write_text("<html></html>")
    (nested / "test_results_test_ops.csv").write_text("col\nval\n")
    (nested / "test_ops.log").write_text("log line")

    counts = flatten_reports(tmp_path)
    assert counts == {
        "json_reports": 1,
        "html_reports": 1,
        "csv_reports": 1,
        "log_reports": 1,
    }
    assert (tmp_path / "json_reports" / "test_results_test_ops.json").is_file()
    assert (tmp_path / "html_reports" / "test_ops.html").is_file()
    # Per-node dir cleanup ran (we did flatten files), so the original is gone.
    assert not (tmp_path / "upstream-pytorch-test_ops").exists()


def test_flatten_reports_preserves_per_node_dirs_when_empty(tmp_path: Path) -> None:
    """Regression: if flatten globs find nothing (e.g. layout changed
    upstream and we don't recognize it), don't blow away the per-node
    dirs — that's the only copy of whatever made it to the artifact, and
    losing it would silently destroy debug evidence.
    """
    # Per-node dir exists but contains no recognized file types — only
    # something the flatten globs don't match.
    odd = tmp_path / "upstream-pytorch-mystery"
    odd.mkdir()
    (odd / "weird_format.xml").write_text("<xml/>")

    counts = flatten_reports(tmp_path)
    # Nothing flattened…
    assert all(c == 0 for c in counts.values())
    # …and the per-node dir was preserved.
    assert (tmp_path / "upstream-pytorch-mystery").is_dir()
    assert (tmp_path / "upstream-pytorch-mystery" / "weird_format.xml").is_file()


def test_flatten_reports_preserves_log_files(tmp_path: Path) -> None:
    # Logs are post-mortem evidence when pytest crashes before producing
    # JSON. They must survive the per-node-dir cleanup, so flatten_reports
    # gathers them into ``log_reports/`` alongside the other report types.
    _make_node(tmp_path, "test_ops")
    flatten_reports(tmp_path)
    assert (tmp_path / "log_reports" / "test_ops.log").is_file()
    # And the per-node copy is gone.
    assert not (tmp_path / "upstream-pytorch-test_ops").exists()


# ---------------------------------------------------------------------------
# write_github_summary — markdown panel rendering, all-skipped filtering
# ---------------------------------------------------------------------------
def _summary_result(
    stats: dict,
    durations: dict | None = None,
    file_stats: dict | None = None,
    file_durations: dict | None = None,
) -> dict:
    """Build the result dict aggregate_summary returns.

    Wraps each unit's stat dict in a defaultdict so missing-status lookups
    return 0 — matches what aggregate_summary produces in production.

    By default ``file_stats`` mirrors ``stats`` (treating the class names as
    file names) so simple tests can keep passing a single dict. Pass an
    explicit ``file_stats`` to render the panel from a different grouping
    than the canonical CSV.
    """
    from collections import defaultdict

    def _wrap(d: dict) -> dict:
        return {k: defaultdict(int, v) for k, v in d.items()}

    file_stats = file_stats if file_stats is not None else stats
    file_durations = (
        file_durations if file_durations is not None else dict.fromkeys(file_stats, 1.0)
    )
    if durations is None:
        durations = dict.fromkeys(stats, 1.0)
    sorted_statuses = sorted({s for class_stats in stats.values() for s in class_stats})
    total_stats: dict[str, int] = {}
    for class_stats in stats.values():
        for s, c in class_stats.items():
            total_stats[s] = total_stats.get(s, 0) + c
    return {
        "stats": _wrap(stats),
        "durations": durations,
        "file_stats": _wrap(file_stats),
        "file_durations": file_durations,
        "sorted_statuses": sorted_statuses,
        "total_stats": total_stats,
    }


def test_write_github_summary_hides_all_skipped_units(tmp_path, monkeypatch):
    """Files where every test was skipped have pass_rate N/A and add no
    signal — they're hidden from the panel. Visible files still render
    with their per-file stats."""
    target = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))

    stats = {
        "test_ran.py": {"passed": 8, "failed": 2},
        "test_all_skipped.py": {"skipped": 50},
        "test_also_all_skipped.py": {"skipped": 30, "skipped_on_retry": 1},
    }
    write_github_summary(_summary_result(stats))

    content = target.read_text()
    assert "test_ran.py" in content
    assert "test_all_skipped.py" not in content
    assert "test_also_all_skipped.py" not in content
    # Hidden count is surfaced.
    assert "Hiding 2 all-skipped file(s)" in content


def test_write_github_summary_shows_all_when_nothing_hidden(tmp_path, monkeypatch):
    target = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    stats = {
        "test_a.py": {"passed": 1, "failed": 1},
        "test_b.py": {"passed": 5},
    }
    write_github_summary(_summary_result(stats))
    content = target.read_text()
    assert "test_a.py" in content
    assert "test_b.py" in content
    # No hidden footer when nothing was filtered.
    assert "Hiding " not in content


def test_write_github_summary_full_total_includes_hidden(tmp_path, monkeypatch):
    """The footer's full-suite TOTAL must include the hidden files' rows
    so the panel doesn't lie about how big the suite is."""
    target = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))

    stats = {
        "test_ran.py": {"passed": 10},
        "test_all_skipped.py": {"skipped": 1000},
    }
    write_github_summary(_summary_result(stats))
    content = target.read_text()
    # Visible TOTAL row covers only test_ran.py = 10.
    assert "**TOTAL (visible)** | 10" in content
    # Footer mentions the full suite size including the hidden 1000.
    assert "Full suite TOTAL: 1010 tests" in content


def test_write_github_summary_no_op_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    # Doesn't raise; nothing to assert besides absence of side effects.
    write_github_summary(_summary_result({"TestX": {"passed": 1}}))


# ---------------------------------------------------------------------------
# Run metadata (commit SHA, branch, run_id) — for dashboard regression-attribution
# ---------------------------------------------------------------------------
def test_get_run_metadata_reads_env_vars(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123def456")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/x")
    monkeypatch.setenv("GITHUB_RUN_ID", "987654")
    md = _get_run_metadata()
    assert md == {
        "commit_sha": "abc123def456",
        "branch": "feature/x",
        "run_id": "987654",
    }


def test_get_run_metadata_defaults_to_unknown_outside_ci(monkeypatch):
    # Outside CI all three env vars are unset — the helper returns
    # "unknown" for each so dashboards can distinguish "no metadata"
    # from "missing column."
    for var in ("GITHUB_SHA", "GITHUB_REF_NAME", "GITHUB_REF", "GITHUB_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    assert _get_run_metadata() == {
        "commit_sha": "unknown",
        "branch": "unknown",
        "run_id": "unknown",
    }


def test_get_run_metadata_falls_back_to_github_ref_when_ref_name_unset(monkeypatch):
    # GITHUB_REF_NAME is the modern var; GITHUB_REF is the older form.
    # Some self-hosted runners only set the old one.
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    assert _get_run_metadata()["branch"] == "refs/heads/main"


# ---------------------------------------------------------------------------
# write_run_metadata_csv — sidecar CSV for dashboard regression-attribution
# ---------------------------------------------------------------------------
def test_run_metadata_csv_has_one_row_with_commit_and_stats(tmp_path, monkeypatch):
    """One sidecar row per run: commit/branch/run_id + total/pass_rate/duration.

    Dashboards join time-series plots on commit_sha/run_id from this CSV
    rather than denormalizing those columns onto every per-test row.
    """
    monkeypatch.setenv("GITHUB_SHA", "abc123def")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/x")
    monkeypatch.setenv("GITHUB_RUN_ID", "9876")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260512_104500")

    result = _summary_result({"TestRan": {"passed": 8, "failed": 2}})
    out = tmp_path / "run_metadata.csv"
    write_run_metadata_csv(result, out)

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["commit_sha"] == "abc123def"
    assert row["branch"] == "feature/x"
    assert row["run_id"] == "9876"
    assert row["run_timestamp"] == "20260512_104500"
    assert row["total"] == "10"
    assert row["passed"] == "8"
    assert row["failed"] == "2"
    assert row["pass_rate(%)"] == "80.0"


def test_run_metadata_csv_unknown_when_outside_ci(tmp_path, monkeypatch):
    """All env vars unset → "unknown" placeholders so dashboards can
    distinguish "no metadata" from "missing column."
    """
    for var in ("GITHUB_SHA", "GITHUB_REF_NAME", "GITHUB_REF", "GITHUB_RUN_ID"):
        monkeypatch.delenv(var, raising=False)

    result = _summary_result({"TestX": {"passed": 1}})
    out = tmp_path / "run_metadata.csv"
    write_run_metadata_csv(result, out)

    row = next(iter(csv.DictReader(out.open())))
    assert row["commit_sha"] == "unknown"
    assert row["branch"] == "unknown"
    assert row["run_id"] == "unknown"


def test_run_metadata_csv_includes_all_status_counts(tmp_path, monkeypatch):
    """Every status (passed/failed/skipped/xfailed/xpassed/error and the
    *_on_retry variants) gets a column so dashboards don't have to
    pre-pivot."""
    for var in ("GITHUB_SHA", "GITHUB_REF_NAME", "GITHUB_RUN_ID"):
        monkeypatch.delenv(var, raising=False)

    result = _summary_result(
        {
            "TestX": {
                "passed": 5,
                "failed": 1,
                "skipped": 2,
                "xfailed": 1,
                "xpassed": 1,
                "error": 1,
                "passed_on_retry": 1,
            }
        }
    )
    out = tmp_path / "run_metadata.csv"
    write_run_metadata_csv(result, out)

    row = next(iter(csv.DictReader(out.open())))
    assert row["passed"] == "5"
    assert row["failed"] == "1"
    assert row["skipped"] == "2"
    assert row["xfailed"] == "1"
    assert row["xpassed"] == "1"
    assert row["error"] == "1"
    assert row["passed_on_retry"] == "1"
    # Statuses not in this run still get a 0-valued column for stable
    # downstream consumption.
    assert row["xfailed_on_retry"] == "0"


def test_summary_csv_does_not_include_metadata_columns(tmp_path, monkeypatch):
    """Regression: per-test / per-class CSVs stay lean. Metadata lives in
    the sidecar CSV; dashboards join on it.
    """
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")

    _write_pytest_json(
        _node_json(tmp_path, "upstream-pytorch-test_x", "test_results_test_x.json"),
        test_class="TestX",
        test_methods=[("test_a", "passed"), ("test_b", "failed")],
    )
    summary_path = tmp_path / "summary.csv"
    aggregate_summary(tmp_path, summary_path)

    cols = csv.DictReader(summary_path.open()).fieldnames
    assert "commit_sha" not in cols
    assert "branch" not in cols
    assert "run_id" not in cols
