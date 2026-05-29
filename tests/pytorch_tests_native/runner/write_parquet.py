"""Parquet writers for the upstream test dashboard pipeline.

Two outputs, one file each per run:

  1. ``run_metrics`` — one row per ``(run, component)``, where
     ``component='overall'`` is the suite-wide roll-up. Drives the
     headline tile, pass-rate trend, and per-component breakdown panels.
  2. ``test_failures`` — one row per failing test (status ``failed`` or
     ``error`` only — ``xfailed`` excluded since those are known issues
     the team is working through, not regressions). Drives the failures
     table panel and acts as the source for newly-failing test detection
     across runs.

The schemas, partitioning convention, and component split are documented
in ``docs/pt_upstream/grafana_athena_setup.md``. Keep that doc and this
module in sync — the Athena DDL there relies on the column names and
types defined here.

If you change the on-disk schema below, update the matching DDL in
``athena/run_metrics.sql`` and ``athena/test_failures.sql`` in the same
PR, then manually re-run the DDL on the prod Athena workgroup.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from aggregate_upstream_results import (
    FAILURE_STATUSES,
    _calc_pass_rate,
    _file_from_row,
    _get_run_metadata,
)

logger = logging.getLogger(__name__)

# Component split. The 'aten_ops' bucket holds the OpInfo / parametrized
# operator-correctness suites; 'aten_core' is the rest of the root-level
# tests (autograd, nn module behavior, tensor mechanics, serialization,
# etc.). See docs/pt_upstream/grafana_athena_setup.md for rationale.
_ATEN_OPS_FILES = frozenset(
    {
        "test_ops.py",
        "test_ops_gradients.py",
        "test_ops_fwd_gradients.py",
        "test_meta.py",
        "test_decomp.py",
        "test_modules.py",
        "test_foreach.py",
        "test_unary_ufuncs.py",
        "test_binary_ufuncs.py",
        "test_linalg.py",
        "test_reductions.py",
    }
)

# Components emitted in run_metrics. ``overall`` is the suite-wide roll-up.
_COMPONENTS = ("overall", "dynamo", "distributed", "profiler", "aten_ops", "aten_core")

# Status columns that exist on a per-component aggregate. Mirrors the
# columns the run-metadata CSV writes so dashboards that join CSV + parquet
# don't see schema drift.
_STATUS_FIELDS = (
    "passed",
    "passed_on_retry",
    "failed",
    "error",
    "skipped",
    "skipped_on_retry",
    "xfailed",
    "xfailed_on_retry",
    "xpassed",
    "xpassed_on_retry",
    "subtests_passed",
)


def _component_for(test_file: str) -> str:
    """Map a test file path to its dashboard component bucket.

    Subdirectory tests (``dynamo/test_x.py``, ``distributed/test_y.py``,
    ``profiler/test_z.py``) collapse to their top-level directory. Root
    files split between 'aten_ops' (OpInfo-iterating files) and
    'aten_core' (everything else).
    """
    if test_file.startswith("dynamo/"):
        return "dynamo"
    if test_file.startswith("distributed/"):
        return "distributed"
    if test_file.startswith("profiler/"):
        return "profiler"
    if test_file in _ATEN_OPS_FILES:
        return "aten_ops"
    return "aten_core"


def _normalize_status(status: str) -> str:
    """pytest-json-report uses ``subtests passed`` with a space; parquet
    column names should be snake_case. Normalize at write time."""
    return status.replace(" ", "_")


def _component_aggregates(test_rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-test rows into per-component status counts + duration.

    Returns ``{component: {status: count, ..., '_duration': seconds}}``
    with one entry per real component (no 'overall'). The caller adds
    the overall roll-up row separately.
    """
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for row in test_rows:
        test_file = _file_from_row(row)
        component = _component_for(test_file)
        status = _normalize_status(row.get("status") or "unknown")
        agg[component][status] += 1
        with contextlib.suppress(TypeError, ValueError):
            agg[component]["_duration"] += float(row.get("duration(sec)", 0) or 0)
    return agg


def _stats_to_dict(stats: dict, component: str, metadata: dict, run_timestamp: str) -> dict:
    """Build a single per-component row in the schema run_metrics expects."""
    row = {
        "commit_sha": metadata["commit_sha"],
        "branch": metadata["branch"],
        "run_id": metadata["run_id"],
        "run_timestamp": run_timestamp,
        "component": component,
    }
    # Per-status counts, defaulting to 0 for absent statuses so every
    # column is present in every row.
    for field in _STATUS_FIELDS:
        row[field] = int(stats.get(field, 0))

    # Roll-ups.
    total = sum(int(stats.get(f, 0)) for f in _STATUS_FIELDS)
    row["total"] = total
    row["pass_rate_pct"] = _pass_rate_value(stats)
    row["duration_sec"] = round(float(stats.get("_duration", 0.0)), 3)
    row["flaky"] = int(stats.get("passed_on_retry", 0)) + int(stats.get("xpassed_on_retry", 0))
    row["failures"] = int(stats.get("failed", 0)) + int(stats.get("error", 0))
    return row


def _resolve_run_timestamp() -> datetime:
    """Parse ``TEST_RUN_TIMESTAMP`` (``YYYYMMDD_HHMMSS``) into a datetime.

    Falls back to UTC ``now`` on missing or malformed values so the writer
    never crashes a run on a bad env var. Both parquet writers share this
    so timestamps stay consistent within one run.
    """
    raw = os.environ.get("TEST_RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.utcnow()


def _pass_rate_value(stats: dict) -> float | None:
    """Compute pass rate as a float (or None for N/A) for parquet.

    ``_calc_pass_rate`` from the aggregator returns a string ('92.6' or
    'N/A') because its output goes into a CSV column. For parquet we
    want a numeric type so Grafana can render it without a cast, with
    NULL as the missing-value sentinel.
    """
    rate = _calc_pass_rate(stats)
    if rate == "N/A":
        return None
    return float(rate)


def _make_run_metrics_table(
    test_rows: list[dict], metadata: dict, run_timestamp: datetime
) -> pa.Table:
    """Build the run_metrics pyarrow Table with one row per component."""
    per_component = _component_aggregates(test_rows)

    # Synthesize the 'overall' aggregate from the union of components so
    # the roll-up is exactly consistent with the per-component rows
    # (sums-of-parts == whole). Computing it from test_rows directly
    # would be equivalent but harder to keep in sync if either path
    # gains a filter later.
    overall: dict = defaultdict(int)
    for stats in per_component.values():
        for k, v in stats.items():
            overall[k] += v

    rows: list[dict] = []
    rows.append(_stats_to_dict(overall, "overall", metadata, run_timestamp.isoformat()))
    # Emit a row for every defined real component, even when the run
    # hit zero tests in that component. Stable schema across runs lets
    # Grafana panels assume the row exists.
    for component in _COMPONENTS:
        if component == "overall":
            continue
        rows.append(
            _stats_to_dict(
                per_component.get(component, {}),
                component,
                metadata,
                run_timestamp.isoformat(),
            )
        )

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("component", pa.string()),
            *((field, pa.int64()) for field in _STATUS_FIELDS),
            ("total", pa.int64()),
            ("pass_rate_pct", pa.float64()),
            ("duration_sec", pa.float64()),
            ("flaky", pa.int64()),
            ("failures", pa.int64()),
        ]
    )
    columnar = {name: [] for name in schema.names}
    for row in rows:
        ts = datetime.fromisoformat(row["run_timestamp"])
        for name in schema.names:
            columnar[name].append(ts if name == "run_timestamp" else row[name])

    return pa.table(columnar, schema=schema)


def _make_test_failures_table(
    test_rows: list[dict], metadata: dict, run_timestamp: datetime
) -> pa.Table:
    """Build the test_failures table — failed/error rows only."""
    # Sort by (component, test_id) before write so parquet's per-row-group
    # min/max statistics on those columns are tight. Athena uses those
    # stats for predicate pushdown — a query filtering by component skips
    # row groups whose component-range doesn't overlap the filter.
    failures = sorted(
        (r for r in test_rows if r.get("status") in FAILURE_STATUSES),
        key=lambda r: (
            _component_for(_file_from_row(r)),
            r.get("test_id") or r.get("nodeid") or "",
        ),
    )

    # ``test_id`` (full pytest nodeid: file::class::method) is the only
    # test-identity column we store. File / class / method are derivable
    # from it via string ops in the dashboard query; storing them
    # separately just duplicates information and risks drift.
    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("component", pa.string()),
            ("test_id", pa.string()),
            ("status", pa.string()),
            ("failure_reason", pa.string()),
            ("duration_sec", pa.float64()),
        ]
    )

    columnar: dict[str, list] = {name: [] for name in schema.names}
    for row in failures:
        try:
            duration = float(row.get("duration(sec)", 0) or 0)
        except (TypeError, ValueError):
            duration = 0.0
        # Truncate the failure reason to its first line — full
        # multi-line reason is preserved in the CSV artifact.
        reason = (row.get("failure_reason") or "").strip()
        first_line = reason.splitlines()[0] if reason else ""

        columnar["commit_sha"].append(metadata["commit_sha"])
        columnar["branch"].append(metadata["branch"])
        columnar["run_id"].append(metadata["run_id"])
        columnar["run_timestamp"].append(run_timestamp)
        # ``component`` IS NOT derivable from test_id without re-running
        # the file→component mapping, so we keep it as its own column.
        columnar["component"].append(_component_for(_file_from_row(row)))
        columnar["test_id"].append(row.get("test_id") or row.get("nodeid") or "")
        columnar["status"].append(row.get("status") or "")
        columnar["failure_reason"].append(first_line)
        columnar["duration_sec"].append(duration)

    return pa.table(columnar, schema=schema)


def write_run_metrics_parquet(test_rows: list[dict], output_path: Path) -> None:
    """Write the run_metrics parquet file.

    One row per component, plus one ``component='overall'`` roll-up row.
    Six rows per run with the current 5-bucket split.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_run_metrics_table(test_rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Run metrics parquet: {output_path} ({table.num_rows} rows)")


def write_test_failures_parquet(test_rows: list[dict], output_path: Path) -> None:
    """Write the test_failures parquet file (failed/error rows only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_test_failures_table(test_rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Test failures parquet: {output_path} ({table.num_rows} rows)")
