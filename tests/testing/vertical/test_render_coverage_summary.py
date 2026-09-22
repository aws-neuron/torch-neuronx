"""Unit tests for the vertical coverage step-summary renderer.

These run without a Neuron host: they exercise only the CSV-to-markdown roll-up.
"""

from __future__ import annotations

import csv
from pathlib import Path

import render_coverage_summary as rcs

_EAGER = (
    "op_name,op_group,status,supported_by,error,time_to_first_ms,exec_avg_ms\n"
    "aten::add,elementwise,runs_on_neuron,registered,,10.0,1.0\n"
    "aten::gather,indexing,cpu_fallback,composite,could not run gather,20.0,2.0\n"
    "aten::foo,misc,failed_lowering,unknown,boom,30.0,3.0\n"
)
_COMPILE = (
    "op_name,op_group,status,supported_by,error,time_to_first_ms,exec_avg_ms\n"
    "aten::add,elementwise,supported,registered,,100.0,5.0\n"
    "aten::gather,indexing,compile_timeout,registered,compile exceeded 1s,900.0,\n"
)


def _write_pair(tmp_path: Path) -> Path:
    (tmp_path / "target_ops_coverage_20260908_120000.csv").write_text(_EAGER)
    (tmp_path / "target_ops_compile_coverage_20260908_120000.csv").write_text(_COMPILE)
    return tmp_path


def test_summarize_suite_counts_and_rate():
    rows = list(csv.DictReader(_EAGER.splitlines()))
    s = rcs.summarize_suite(rows)
    assert s["total"] == 3
    assert s["on_neuron"] == 1
    assert s["rate"] == 33.3
    assert len(s["gaps"]) == 2


def test_render_coverage_writes_panel(tmp_path):
    _write_pair(tmp_path)
    md = rcs.render(tmp_path, tmp_path, kind="coverage")

    # Persisted for the append-step-summary action.
    assert (tmp_path / "step_summary.md").read_text() == md

    # Both suites appear with their headline label.
    assert "## eager" in md
    assert "## torch.compile" in md
    assert "On Neuron:** 1 (33.3%)" in md

    # The gap table lists the non-supported ops from both suites, worst first.
    assert "Ops not fully on Neuron (3)" in md
    assert md.index("failed_lowering") < md.index("cpu_fallback")
    assert "compile exceeded 1s" in md

    # Coverage kind does not emit the perf slowest tables.
    assert "Slowest compiles" not in md


def test_render_perf_adds_slowest_tables(tmp_path):
    _write_pair(tmp_path)
    md = rcs.render(tmp_path, tmp_path, kind="perf")
    assert "Vertical Op Perf Results" in md
    assert "Slowest compiles" in md
    assert "Slowest exec" in md
    # The slowest compile is the compile-suite gather at 900ms.
    slow = md[md.index("Slowest compiles") :]
    assert "aten::gather" in slow


def test_missing_csv_renders_empty_panel(tmp_path):
    md = rcs.render(tmp_path, tmp_path, kind="coverage")
    assert "No coverage CSV was found" in md
    assert (tmp_path / "step_summary.md").exists()


def test_github_step_summary_append(tmp_path, monkeypatch):
    _write_pair(tmp_path)
    gh = tmp_path / "gh_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh))
    md = rcs.render(tmp_path, tmp_path, kind="coverage")
    assert gh.read_text() == md
