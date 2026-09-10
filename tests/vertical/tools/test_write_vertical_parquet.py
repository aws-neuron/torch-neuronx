"""Unit tests for the vertical coverage parquet writers.

These run without a Neuron host: they exercise only the CSV-to-parquet
conversion (pyarrow + stdlib), the same way the upstream parquet writers are
tested on a plain runner.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import write_vertical_parquet as wv

_METADATA = {
    "commit_sha": "abc123",
    "branch": "main",
    "run_id": "42",
    "workflow_type": "unit",
}
_TS = datetime(2026, 9, 8, 12, 0, 0)

_ROWS = [
    {
        "op_name": "aten::add.Tensor",
        "op_group": "elementwise",
        "status": "runs_on_neuron",
        "input_count": "3",
        "input_element_count": "4096",
        "roofline_ratio": "0.75",
        "exec_avg_ms": "",  # empty cell -> NULL
        "neuron_primitive_ops": "aten::add; aten::mul",
        "last_updated": "2026-09-08 12:00:00",
    },
    {
        "op_name": "aten::linear",
        "op_group": "elementwise",
        "status": "cpu_fallback",
        "input_count": "2",
    },
    {
        "op_name": "aten::sort",
        "op_group": "reduction",
        "status": "failed_lowering",
    },
]


def test_op_coverage_table_types_and_nulls():
    table = wv._make_op_coverage_table(_ROWS, _METADATA, _TS)
    assert table.num_rows == 3
    d = table.to_pydict()
    # Empty numeric cells become NULL, populated ones parse.
    add_idx = d["op_name"].index("aten::add.Tensor")
    assert d["input_count"][add_idx] == 3
    assert d["roofline_ratio"][add_idx] == pytest.approx(0.75)
    assert d["exec_avg_ms"][add_idx] is None
    # A row missing the column entirely also reads NULL, not a crash.
    sort_idx = d["op_name"].index("aten::sort")
    assert d["input_count"][sort_idx] is None
    assert "suite" not in d
    # Run identity stamped on every row.
    assert set(d["commit_sha"]) == {"abc123"}


def test_run_metrics_rollup_and_supported_rate():
    table = wv._make_run_metrics_table(_ROWS, _METADATA, _TS)
    d = table.to_pydict()
    by_component = {c: i for i, c in enumerate(d["component"])}
    assert "overall" in by_component
    # overall: 3 ops, 1 runs_on_neuron -> 33.3%.
    ov = by_component["overall"]
    assert d["n_ops"][ov] == 3
    assert d["runs_on_neuron"][ov] == 1
    assert d["supported_rate_pct"][ov] == pytest.approx(33.3, abs=0.1)
    # Per-group split: elementwise has 2 ops (1 supported -> 50%).
    ew = by_component["elementwise"]
    assert d["n_ops"][ew] == 2
    assert d["supported_rate_pct"][ew] == pytest.approx(50.0)


def test_round_trip_write_read(tmp_path):
    out = tmp_path / "op_coverage.parquet"
    wv.write_op_coverage_parquet(_ROWS, out)
    assert out.exists()
    assert pq.read_table(out).num_rows == 3


def test_read_coverage_csv(tmp_path):
    csv_path = tmp_path / "cov.csv"
    csv_path.write_text("op_name,status\naten::add,runs_on_neuron\n")
    rows = wv.read_coverage_csv(csv_path)
    assert rows == [{"op_name": "aten::add", "status": "runs_on_neuron"}]


def test_resolve_run_timestamp_malformed(monkeypatch):
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "not-a-timestamp")
    # Falls back to now() rather than raising.
    assert isinstance(wv._resolve_run_timestamp(), datetime)
