"""Unit tests for the vertical coverage aggregator CLI.

These run without a Neuron host: they exercise only the CSV-to-parquet path
selection (suite -> table variant) and the partition layout.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import aggregate_vertical_results as agg

_CSV = "op_name,op_group,status\naten::add,elementwise,runs_on_neuron\n"


def _write_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "target_ops_coverage_20260908_120000.csv"
    csv_path.write_text(_CSV)
    return csv_path


def _run(csv_path: Path, suite: str, out: Path, run_kind: str = "coverage") -> int:
    return agg.main(
        [
            "--csv",
            str(csv_path),
            "--suite",
            suite,
            "--output-dir",
            str(out),
            "--run-kind",
            run_kind,
        ]
    )


def test_eager_suite_writes_eager_tables(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "results"

    assert _run(csv_path, "target_ops_coverage", out) == 0

    dt = datetime.now().strftime("%Y-%m-%d")
    root = out / "custom-test-artifacts" / "vertical_ops"
    for table in ("op_coverage_eager", "run_metrics_eager"):
        assert (root / table / "branch=unknown" / f"dt={dt}" / "run_unknown.parquet").exists()
    # No suite= partition and no torch_compile tables leaked in.
    assert not list(root.glob("**/suite=*"))
    assert not (root / "op_coverage_torch_compile").exists()


def test_compile_suite_writes_torch_compile_tables(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    csv_path = tmp_path / "target_ops_compile_coverage_20260908_120000.csv"
    csv_path.write_text(_CSV)
    out = tmp_path / "results"

    assert _run(csv_path, "target_ops_compile_coverage", out) == 0

    root = out / "custom-test-artifacts" / "vertical_ops"
    assert (root / "op_coverage_torch_compile").exists()
    assert (root / "run_metrics_torch_compile").exists()


def test_perf_run_kind_writes_op_perf_only(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "results"

    assert _run(csv_path, "target_ops_coverage", out, run_kind="perf") == 0

    dt = datetime.now().strftime("%Y-%m-%d")
    root = out / "custom-test-artifacts" / "vertical_ops"
    assert (root / "op_perf_eager" / "branch=unknown" / f"dt={dt}" / "run_unknown.parquet").exists()
    # A perf run writes neither the coverage table nor the functionality roll-up.
    assert not (root / "op_coverage_eager").exists()
    assert not (root / "run_metrics_eager").exists()


def test_unknown_suite_is_rejected(tmp_path):
    csv_path = _write_csv(tmp_path)
    out = tmp_path / "results"
    assert _run(csv_path, "not_a_suite", out) == 1
    assert not (out / "custom-test-artifacts").exists()
