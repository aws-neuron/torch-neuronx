"""Parquet writers for the vertical op-coverage dashboard pipeline.

This mirrors the upstream test pipeline
(``tests/pytorch_tests_native/runner/write_parquet.py``): it turns an
aggregated coverage CSV into typed parquet that a DevOps job mirrors to S3 and
Athena reads. It is credential-free. The workflow uploads the parquet tree under
the ``custom-test-artifacts`` artifact name, and the mirror copies the tree to
S3 verbatim.

Two tables, one file each per run:

  1. ``run_metrics`` -- one row per ``(run, component)``, where
     ``component='overall'`` is the suite-wide roll-up and each other component
     is one ``op_group``. Drives the headline tile and the per-group breakdown.
  2. ``op_coverage`` -- one row per op. Mirrors the coverage CSV but typed and
     partition-pruned for Athena. Drives per-op analysis: which ops fall back,
     roofline category, timings.

The coverage CSV columns come from ``tests/vertical/conftest.py``
(``COVERAGE_FIELDS``). Keep this module and that list in sync.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Status the coverage test writes when the op runs fully on Neuron. Every other
# status is a gap of some kind (fallback, lowering failure, no sample, ...).
SUPPORTED_STATUS = "runs_on_neuron"

# Per-status counts materialized on every run_metrics row. A status the run did
# not produce reads 0, so the schema is stable run-to-run.
_STATUS_FIELDS = (
    "runs_on_neuron",
    "cpu_fallback",
    "failed_lowering",
    "no_input_sample",
    "unresolved",
)

# op_coverage columns typed as int64 (nullable). Empty CSV cells become NULL.
_INT_FIELDS = (
    "input_count",
    "input_element_count",
)

# op_coverage columns typed as float64 (nullable).
_FLOAT_FIELDS = (
    "time_to_first_ms",
    "ttf_slowest_ms",
    "exec_avg_ms",
    "exec_min_ms",
    "exec_max_ms",
    "t_roofline_ms",
    "t_compute_ms",
    "t_memory_ms",
    "roofline_ratio",
)

# op_coverage columns kept as strings.
_STR_FIELDS = (
    "op_name",
    "op_group",
    "status",
    "supported_by",
    "input_source",
    "neuron_primitive_ops",
    "fallback_primitive_ops",
    "bound",
    "roofline_category",
    "correctness_test",
    "last_updated",
    "error",
)

# Run-identity columns every row repeats. run_timestamp is a datetime; the rest
# come from the run-metadata dict.
_RUN_METADATA_COLUMNS = ("commit_sha", "branch", "run_id", "workflow_type")


def _get_run_metadata() -> dict:
    """Pull commit/branch/run-id/workflow from GitHub Actions env vars.

    Defaults to ``"unknown"`` outside CI so a dashboard can tell "no metadata"
    from "missing column". Mirrors the upstream aggregator so the two pipelines
    stay consistent.
    """
    return {
        "commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "branch": (os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF", "unknown")),
        "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "workflow_type": os.environ.get("GITHUB_WORKFLOW", "unknown"),
    }


def _resolve_run_timestamp() -> datetime:
    """Parse ``TEST_RUN_TIMESTAMP`` (``YYYYMMDD_HHMMSS``) into a datetime.

    Falls back to ``now`` on a missing or malformed value so the writer never
    crashes a run on a bad env var.
    """
    raw = os.environ.get("TEST_RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.now()


def read_coverage_csv(path: str | Path) -> list[dict]:
    """Read an aggregated coverage CSV into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_int(value: str | None) -> int | None:
    """CSV cell to int, or None for an empty or malformed cell."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: str | None) -> float | None:
    """CSV cell to float, or None for an empty or malformed cell."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_run_metadata(
    columnar: dict[str, list], metadata: dict, run_timestamp: datetime
) -> None:
    """Append the shared run-identity columns for one row."""
    for col in _RUN_METADATA_COLUMNS:
        columnar[col].append(metadata.get(col, "unknown"))
    columnar["run_timestamp"].append(run_timestamp)


def _make_op_coverage_table(rows: list[dict], metadata: dict, run_timestamp: datetime) -> pa.Table:
    """Build the op_coverage table -- one typed row per op."""
    # Sort by (op_group, op_name) so parquet row-group min/max stats are tight
    # for Athena predicate pushdown on those columns.
    rows = sorted(rows, key=lambda r: (r.get("op_group") or "", r.get("op_name") or ""))

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            *((f, pa.string()) for f in _STR_FIELDS),
            *((f, pa.int64()) for f in _INT_FIELDS),
            *((f, pa.float64()) for f in _FLOAT_FIELDS),
        ]
    )

    columnar: dict[str, list] = {name: [] for name in schema.names}
    for row in rows:
        _append_run_metadata(columnar, metadata, run_timestamp)
        for f in _STR_FIELDS:
            columnar[f].append(row.get(f) or "")
        for f in _INT_FIELDS:
            columnar[f].append(_as_int(row.get(f)))
        for f in _FLOAT_FIELDS:
            columnar[f].append(_as_float(row.get(f)))

    return pa.table(columnar, schema=schema)


def _supported_rate(status_counts: dict) -> float | None:
    """Percent of ops that run on Neuron, or None when no op counted."""
    total = sum(status_counts.values())
    if total == 0:
        return None
    return round(status_counts.get(SUPPORTED_STATUS, 0) / total * 100, 1)


def _make_run_metrics_table(rows: list[dict], metadata: dict, run_timestamp: datetime) -> pa.Table:
    """Build the run_metrics table -- one row per op_group plus an overall row."""
    per_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row.get("op_group") or "unknown"
        status = row.get("status") or "unknown"
        per_group[group][status] += 1
        overall[status] += 1

    def _row(component: str, counts: dict[str, int]) -> dict:
        out = {
            **{c: metadata.get(c, "unknown") for c in _RUN_METADATA_COLUMNS},
            "run_timestamp": run_timestamp,
            "component": component,
            "n_ops": sum(counts.values()),
            "supported_rate_pct": _supported_rate(counts),
        }
        for f in _STATUS_FIELDS:
            out[f] = int(counts.get(f, 0))
        return out

    ordered = [_row("overall", overall)]
    ordered += [_row(g, per_group[g]) for g in sorted(per_group)]

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("component", pa.string()),
            ("n_ops", pa.int64()),
            ("supported_rate_pct", pa.float64()),
            *((f, pa.int64()) for f in _STATUS_FIELDS),
        ]
    )
    columnar = {name: [] for name in schema.names}
    for row in ordered:
        for name in schema.names:
            columnar[name].append(row[name])
    return pa.table(columnar, schema=schema)


def write_op_coverage_parquet(rows: list[dict], output_path: Path) -> None:
    """Write the op_coverage parquet file (one row per op)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_op_coverage_table(rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Op coverage parquet: %s (%d rows)", output_path, table.num_rows)


def write_run_metrics_parquet(rows: list[dict], output_path: Path) -> None:
    """Write the run_metrics parquet file (one row per op_group plus overall)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_run_metrics_table(rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Run metrics parquet: %s (%d rows)", output_path, table.num_rows)
