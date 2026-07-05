"""Regression tests for ``job_metadata`` (version-metadata capture + writer).

These tests skip when ``pyarrow`` is unavailable so contributors without
the dev extras can still run the rest of the test suite. CI installs
``pyarrow`` via the ``dev`` extra so the tests run there.

Two sides are covered:
  * node side — ``collect_versions`` / ``write_versions_sidecar`` (probing is
    defensive; missing packages become None, never an exception).
  * aggregator side — ``write_job_metadata_parquet`` (collapse per-node
    sidecars to one row, consistency flag, graceful skip on no data).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

# Skip the entire module if pyarrow isn't available; the writer is imported
# after the skip so the import error is contained.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from job_metadata import (  # noqa: E402  (import-after-skip is intentional)
    VERSION_FIELDS,
    VERSIONS_SIDECAR_NAME,
    _collapse_versions,
    collect_versions,
    write_job_metadata_parquet,
    write_versions_sidecar,
)

# A realistic version set (values are illustrative, not probed from a live host).
_BASE_VERSIONS = {
    "pytorch_version": "2.12.0",
    "torch_neuronx_version": "2.12.0.1.0",
    "neuronx_cc_version": "2.21.18209.0+043b1bf7",
    "aws_neuronx_runtime_lib_version": "2.28.23.0",
    "aws_neuronx_collectives_version": "2.28.27.0",
    "aws_neuronx_dkms_version": "2.24.7.0",
    "aws_neuronx_tools_version": "2.28.55.0",
}

_METADATA = {
    "run_id": "28519648174",
    "commit_sha": "b615e6a877e6970a2139e56bc4a40ba258a5da8a",
    "branch": "main",
    "workflow_type": "post_merge",
}


def _make_node_sidecar(results_dir: Path, label: str, versions: dict) -> None:
    """Create a per-node artifact dir + env_versions.json in the nested layout
    the aggregator globs (``upstream-pytorch-<label>/test-results/.../``)."""
    node_dir = (
        results_dir
        / f"upstream-pytorch-{label}"
        / "test-results"
        / "outputs"
        / "upstream_pytorch_artifacts"
    )
    node_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_label": label,
        "run_id": _METADATA["run_id"],
        "commit_sha": _METADATA["commit_sha"],
        "run_timestamp": "20260701_144754",
        "versions": versions,
    }
    with open(node_dir / VERSIONS_SIDECAR_NAME, "w") as f:
        json.dump(payload, f)


def _write(results_dir: Path, out: Path, versions_per_node: list[dict]) -> list[dict] | None:
    """Seed sidecars, run the writer, return the parquet rows (or None if skipped)."""
    for i, v in enumerate(versions_per_node):
        _make_node_sidecar(results_dir, f"node{i}", v)
    ts = datetime(2026, 7, 1, 14, 47, 54)
    result = write_job_metadata_parquet(results_dir, _METADATA, ts, out)
    if result is None:
        return None
    # Read the single file directly so Hive partition dirs in the path aren't
    # inferred as columns clashing with the real branch/dt columns.
    return pq.ParquetFile(out).read().to_pylist()


# ---------------------------------------------------------------------------
# Node side — probing
# ---------------------------------------------------------------------------


def test_collect_versions_returns_all_fields():
    """Every declared field is present; values are str-or-None, never raises."""
    versions = collect_versions()
    assert set(versions.keys()) == set(VERSION_FIELDS)
    for value in versions.values():
        assert value is None or isinstance(value, str)


def test_collect_versions_missing_packages_are_none():
    """On a host without the neuron packages the neuron fields are None (not an
    error). torch may or may not be importable, so we only assert the dpkg ones."""
    versions = collect_versions()
    for field in (
        "aws_neuronx_runtime_lib_version",
        "aws_neuronx_collectives_version",
        "aws_neuronx_dkms_version",
        "aws_neuronx_tools_version",
    ):
        assert versions[field] is None or isinstance(versions[field], str)


def test_write_versions_sidecar_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    monkeypatch.setenv("GITHUB_SHA", "cafebabe")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260701_010203")
    sidecar = write_versions_sidecar(tmp_path, node_label="test_ops_shard0")

    assert sidecar.name == VERSIONS_SIDECAR_NAME
    payload = json.loads(sidecar.read_text())
    assert payload["node_label"] == "test_ops_shard0"
    assert payload["run_id"] == "999"
    assert payload["commit_sha"] == "cafebabe"
    assert set(payload["versions"].keys()) == set(VERSION_FIELDS)


def test_write_versions_sidecar_node_label_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NODE_LABEL", "from_env")
    payload = json.loads(write_versions_sidecar(tmp_path).read_text())
    assert payload["node_label"] == "from_env"


# ---------------------------------------------------------------------------
# Aggregator side — collapse + write
# ---------------------------------------------------------------------------


def test_collapse_versions_homogeneous():
    sidecars = [{"versions": dict(_BASE_VERSIONS)} for _ in range(4)]
    collapsed, consistent = _collapse_versions(sidecars)
    assert consistent is True
    assert collapsed == _BASE_VERSIONS


def test_collapse_versions_heterogeneous_majority_wins():
    odd = dict(_BASE_VERSIONS)
    odd["neuronx_cc_version"] = "2.99.99999.0+deadbeef"
    sidecars = [{"versions": dict(_BASE_VERSIONS)} for _ in range(3)] + [{"versions": odd}]
    collapsed, consistent = _collapse_versions(sidecars)
    assert consistent is False
    # 3-of-4 majority wins
    assert collapsed["neuronx_cc_version"] == _BASE_VERSIONS["neuronx_cc_version"]


def test_collapse_versions_all_none():
    sidecars = [{"versions": {k: None for k in VERSION_FIELDS}} for _ in range(2)]
    collapsed, consistent = _collapse_versions(sidecars)
    assert consistent is True  # no field had conflicting non-null values
    assert all(v is None for v in collapsed.values())


def test_write_parquet_homogeneous_run(tmp_path):
    out = tmp_path / "job_metadata.parquet"
    rows = _write(tmp_path, out, [dict(_BASE_VERSIONS) for _ in range(4)])
    assert rows is not None and len(rows) == 1
    row = rows[0]
    assert row["pytorch_version"] == "2.12.0"
    assert row["aws_neuronx_dkms_version"] == "2.24.7.0"
    assert row["node_count"] == 4
    assert row["versions_consistent"] is True
    # run-level metadata is stamped from the aggregator's metadata dict
    assert row["run_id"] == _METADATA["run_id"]
    assert row["workflow_type"] == "post_merge"


def test_write_parquet_heterogeneous_flags_inconsistent(tmp_path):
    odd = dict(_BASE_VERSIONS)
    odd["neuronx_cc_version"] = "2.99.99999.0+deadbeef"
    out = tmp_path / "job_metadata.parquet"
    rows = _write(tmp_path, out, [dict(_BASE_VERSIONS), dict(_BASE_VERSIONS), odd])
    assert rows[0]["versions_consistent"] is False
    assert rows[0]["node_count"] == 3


def test_write_parquet_missing_packages_become_null(tmp_path):
    empty = {k: None for k in VERSION_FIELDS}
    out = tmp_path / "job_metadata.parquet"
    rows = _write(tmp_path, out, [empty, empty])
    assert rows[0]["pytorch_version"] is None
    assert rows[0]["node_count"] == 2


def test_write_parquet_skips_when_no_sidecars(tmp_path):
    """Old image without the collector → no sidecars → writer returns None and
    writes nothing (graceful degradation, mirrors the workflow_type rollout)."""
    out = tmp_path / "job_metadata.parquet"
    result = write_job_metadata_parquet(tmp_path, _METADATA, datetime(2026, 7, 1), out)
    assert result is None
    assert not out.exists()


def test_write_parquet_schema_matches_expected_columns(tmp_path):
    """Schema-drift guard for the job_metadata Athena-facing column set."""
    out = tmp_path / "job_metadata.parquet"
    _write(tmp_path, out, [dict(_BASE_VERSIONS)])
    schema = pq.ParquetFile(out).schema_arrow
    expected = {
        "run_id",
        "commit_sha",
        "branch",
        "workflow_type",
        "run_timestamp",
        *VERSION_FIELDS,
        "node_count",
        "versions_consistent",
    }
    assert set(schema.names) == expected
