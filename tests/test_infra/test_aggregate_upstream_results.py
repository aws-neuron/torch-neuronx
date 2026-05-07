"""Regression tests for ``aggregate_upstream_results``.

The aggregator globs ``outputs/`` specifically to avoid double-counting from
the duplicate ``test-results/`` copy in each per-node artifact.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from aggregate_upstream_results import aggregate_summary


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


def _outputs_json(base: Path, artifact: str, filename: str) -> Path:
    return base / artifact / "outputs" / "upstream_pytorch_artifacts" / "json_reports" / filename


def test_duplicate_artifact_paths_not_double_counted(tmp_path: Path) -> None:
    """Only the ``outputs/`` copy is counted, not the ``test-results/`` copy."""
    artifact = "upstream-pytorch-test_autograd"
    for subpath in ("outputs", "test-results"):
        _write_pytest_json(
            tmp_path
            / artifact
            / subpath
            / "upstream_pytorch_artifacts"
            / "json_reports"
            / "test_results_test_autograd.json",
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
            _outputs_json(tmp_path, node, "test_results_stub.json"),
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
            _outputs_json(tmp_path, node, f"test_results_test_ops_shard{shard}.json"),
            test_class="TestOps",
            test_methods=[(f"test_shard{shard}", "passed")],
        )

    result = aggregate_summary(tmp_path, tmp_path / "summary.csv")
    assert result["total_stats"]["passed"] == 3


def test_pass_rate_correct(tmp_path: Path) -> None:
    """Summary CSV pass rate is computed correctly."""
    artifact = "upstream-pytorch-test_example"
    _write_pytest_json(
        _outputs_json(tmp_path, artifact, "test_results_test_example.json"),
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
