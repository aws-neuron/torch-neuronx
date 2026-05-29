"""Regression tests for ``write_parquet``.

These tests skip when ``pyarrow`` is unavailable so contributors without
the dev extras can still run the rest of the test suite. CI installs
``pyarrow`` via the ``dev`` extra so the tests run there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

# Skip the entire module if pyarrow isn't available; the writer is also
# imported inside the conditional so the import error is contained.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from write_parquet import (  # noqa: E402  (import-after-skip is intentional)
    _ATEN_OPS_FILES,
    _component_for,
    _normalize_status,
    write_run_metrics_parquet,
    write_test_failures_parquet,
)


def _row(test_id: str, status: str, *, duration: float = 0.1, reason: str = "") -> dict:
    """Build a per-test row in the shape ``aggregate_summary`` produces."""
    test_file, _, rest = test_id.partition("::")
    test_class, _, test_method = rest.partition("::")
    return {
        "test_id": test_id,
        "nodeid": test_id,
        "test_class": test_class,
        "test_method": test_method,
        "status": status,
        "duration(sec)": duration,
        "failure_reason": reason,
    }


# ---------------------------------------------------------------------------
# _component_for — directory-prefix routing + ATen split
# ---------------------------------------------------------------------------
def test_component_for_dynamo_subdir():
    assert _component_for("dynamo/test_compile.py") == "dynamo"
    assert _component_for("dynamo/test_misc.py") == "dynamo"


def test_component_for_distributed_subdir():
    assert _component_for("distributed/test_ddp.py") == "distributed"
    assert _component_for("distributed/fsdp/test_fsdp.py") == "distributed"


def test_component_for_profiler_subdir():
    assert _component_for("profiler/test_profiler.py") == "profiler"


def test_component_for_aten_ops_files():
    """Every file in the ATen ops bucket maps to 'aten_ops'."""
    for f in _ATEN_OPS_FILES:
        assert _component_for(f) == "aten_ops", f


def test_component_for_aten_core_fallback():
    """Root-level files not in the ops bucket land in 'aten_core'."""
    assert _component_for("test_autograd.py") == "aten_core"
    assert _component_for("test_nn.py") == "aten_core"
    assert _component_for("test_indexing.py") == "aten_core"
    assert _component_for("test_view_ops.py") == "aten_core"
    # Subdirectory under aten — nn/ — is also core.
    assert _component_for("nn/test_module_x.py") == "aten_core"


def test_normalize_status_replaces_spaces():
    assert _normalize_status("subtests passed") == "subtests_passed"
    assert _normalize_status("passed") == "passed"


# ---------------------------------------------------------------------------
# write_run_metrics_parquet — schema + per-component rows
# ---------------------------------------------------------------------------
def _read_parquet(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def test_run_metrics_emits_one_row_per_component_plus_overall(tmp_path, monkeypatch):
    """Every defined component appears as a row, including 'overall'."""
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("dynamo/test_compile.py::TestC::test_a", "passed"),
        _row("test_ops.py::TestOps::test_b", "failed", reason="bad output"),
        _row("test_autograd.py::TestAG::test_c", "passed"),
    ]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    data = _read_parquet(out)
    components = {r["component"] for r in data}
    assert components == {"overall", "dynamo", "distributed", "profiler", "aten_ops", "aten_core"}


def test_run_metrics_overall_row_sums_components(tmp_path, monkeypatch):
    """The 'overall' row equals the sum of per-component rows."""
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("dynamo/test_compile.py::TestC::test_a", "passed"),
        _row("dynamo/test_compile.py::TestC::test_b", "failed"),
        _row("test_ops.py::TestOps::test_c", "passed"),
        _row("test_autograd.py::TestAG::test_d", "skipped"),
    ]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    by_component = {r["component"]: r for r in _read_parquet(out)}
    overall = by_component["overall"]

    assert overall["passed"] == 2
    assert overall["failed"] == 1
    assert overall["skipped"] == 1
    assert overall["total"] == 4
    # Sum-of-parts == whole.
    parts_passed = sum(by_component[c]["passed"] for c in ("dynamo", "aten_ops", "aten_core"))
    assert parts_passed == overall["passed"]


def test_run_metrics_pass_rate_is_numeric(tmp_path, monkeypatch):
    """pass_rate_pct is a float (or NULL), not a CSV-style string."""
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("test_ops.py::T::a", "passed"),
        _row("test_ops.py::T::b", "failed"),
    ]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    aten_ops = next(r for r in _read_parquet(out) if r["component"] == "aten_ops")
    assert aten_ops["pass_rate_pct"] == 50.0


def test_run_metrics_pass_rate_null_for_all_skipped(tmp_path, monkeypatch):
    """A component with only skipped tests has NULL pass_rate (N/A semantics)."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("dynamo/test_x.py::T::a", "skipped")]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    dynamo = next(r for r in _read_parquet(out) if r["component"] == "dynamo")
    assert dynamo["pass_rate_pct"] is None


def test_run_metrics_empty_components_emit_zero_rows(tmp_path, monkeypatch):
    """Components with no tests still get a row (zeroed) so the schema is stable."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    # Only dynamo has data. Other components should still be present.
    rows = [_row("dynamo/test_x.py::T::a", "passed")]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    by_component = {r["component"]: r for r in _read_parquet(out)}
    assert by_component["distributed"]["total"] == 0
    assert by_component["profiler"]["total"] == 0
    assert by_component["aten_ops"]["total"] == 0
    assert by_component["aten_core"]["total"] == 0


def test_run_metrics_metadata_columns_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF_NAME", "release/2.11")
    monkeypatch.setenv("GITHUB_RUN_ID", "9876")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("test_ops.py::T::a", "passed")]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    for row in _read_parquet(out):
        assert row["commit_sha"] == "deadbeef"
        assert row["branch"] == "release/2.11"
        assert row["run_id"] == "9876"


def test_run_metrics_subtests_passed_normalized(tmp_path, monkeypatch):
    """``subtests passed`` (with space) lands in the snake_case column."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("dynamo/test_x.py::T::a", "subtests passed")]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    dynamo = next(r for r in _read_parquet(out) if r["component"] == "dynamo")
    assert dynamo["subtests_passed"] == 1


def test_run_metrics_flaky_and_failures_denormalized(tmp_path, monkeypatch):
    """`flaky` and `failures` columns aggregate retry / fail buckets."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("test_ops.py::T::a", "passed_on_retry"),
        _row("test_ops.py::T::b", "xpassed_on_retry"),
        _row("test_ops.py::T::c", "failed"),
        _row("test_ops.py::T::d", "error"),
    ]
    out = tmp_path / "rm.parquet"
    write_run_metrics_parquet(rows, out)

    aten_ops = next(r for r in _read_parquet(out) if r["component"] == "aten_ops")
    assert aten_ops["flaky"] == 2  # passed_on_retry + xpassed_on_retry
    assert aten_ops["failures"] == 2  # failed + error


# ---------------------------------------------------------------------------
# write_test_failures_parquet — failures-only filtering
# ---------------------------------------------------------------------------
def test_test_failures_filters_to_failed_and_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("test_ops.py::T::a", "passed"),
        _row("test_ops.py::T::b", "failed", reason="boom"),
        _row("test_ops.py::T::c", "error", reason="setup error"),
        _row("test_ops.py::T::d", "xfailed"),  # NOT a failure
        _row("test_ops.py::T::e", "passed_on_retry"),  # NOT a failure
    ]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    data = _read_parquet(out)
    statuses = {r["status"] for r in data}
    assert statuses == {"failed", "error"}
    assert len(data) == 2


def test_test_failures_truncates_reason_to_first_line(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("test_ops.py::T::a", "failed", reason="line one\nline two\nline three"),
    ]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    data = _read_parquet(out)
    assert data[0]["failure_reason"] == "line one"


def test_test_failures_empty_table_when_no_failures(tmp_path, monkeypatch):
    """Even with no failures, the parquet file is written with an empty table
    so downstream consumers can rely on the path existing."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("test_ops.py::T::a", "passed")]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    table = pq.read_table(out)
    assert table.num_rows == 0
    # Schema is still fully populated so Athena DDL works against an empty file.
    assert "test_id" in table.schema.names
    assert "component" in table.schema.names
    # File/class/method are intentionally NOT stored — they're derivable
    # from test_id via string ops at query time.
    assert "test_file" not in table.schema.names
    assert "test_class" not in table.schema.names
    assert "test_method" not in table.schema.names


def test_test_failures_attaches_component(tmp_path, monkeypatch):
    """Each failure row carries its component bucket."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [
        _row("dynamo/test_compile.py::T::a", "failed"),
        _row("test_autograd.py::T::b", "failed"),
        _row("test_ops.py::T::c", "error"),
    ]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    data = _read_parquet(out)
    by_test = {r["test_id"]: r for r in data}
    assert by_test["dynamo/test_compile.py::T::a"]["component"] == "dynamo"
    assert by_test["test_autograd.py::T::b"]["component"] == "aten_core"
    assert by_test["test_ops.py::T::c"]["component"] == "aten_ops"


def test_test_failures_metadata_columns_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("GITHUB_REF_NAME", "release/2.11")
    monkeypatch.setenv("GITHUB_RUN_ID", "9876")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("test_ops.py::T::a", "failed", reason="boom")]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    row = _read_parquet(out)[0]
    assert row["commit_sha"] == "deadbeef"
    assert row["branch"] == "release/2.11"
    assert row["run_id"] == "9876"


def test_test_failures_rows_sorted_by_component_then_test_id(tmp_path, monkeypatch):
    """Rows are sorted by (component, test_id) before write so parquet's
    row-group min/max stats are tight and Athena predicate pushdown can
    skip row groups outside a component-filtered query."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    # Deliberately pass rows out of order — writer must sort.
    rows = [
        _row("test_ops.py::T::z", "failed"),  # aten_ops
        _row("dynamo/test_compile.py::T::a", "failed"),  # dynamo
        _row("test_autograd.py::T::b", "failed"),  # aten_core
        _row("dynamo/test_compile.py::T::aaa", "error"),  # dynamo
        _row("distributed/test_ddp.py::T::x", "failed"),  # distributed
    ]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    data = _read_parquet(out)
    keys = [(r["component"], r["test_id"]) for r in data]
    assert keys == sorted(keys)
    # Sanity: dynamo rows come together (they wouldn't if sort were broken).
    components = [r["component"] for r in data]
    assert components.count("dynamo") == 2
    first_dynamo = components.index("dynamo")
    assert components[first_dynamo : first_dynamo + 2] == ["dynamo", "dynamo"]


def test_test_failures_preserves_full_test_id(tmp_path, monkeypatch):
    """The full pytest nodeid is preserved verbatim — readers extract
    file/class/method from it via string ops if needed."""
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")

    rows = [_row("test_ops.py::TestOps::test_index_put_neuron_float32", "failed")]
    out = tmp_path / "tf.parquet"
    write_test_failures_parquet(rows, out)

    row = _read_parquet(out)[0]
    assert row["test_id"] == "test_ops.py::TestOps::test_index_put_neuron_float32"
