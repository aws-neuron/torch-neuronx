"""Parquet writers for the upstream test dashboard pipeline.

Three outputs, one file each per run:

  1. ``run_metrics`` — one row per ``(run, component)``, where
     ``component='overall'`` is the suite-wide roll-up. Drives the
     headline tile, pass-rate trend, and per-component breakdown panels.
  2. ``test_failures`` — one row per failing test (status ``failed`` or
     ``error`` only — ``xfailed`` excluded since those are known issues
     the team is working through, not regressions). Drives the failures
     table panel and acts as the source for newly-failing test detection
     across runs.
  3. ``test_aggregated`` — one row per non-skipped test (passed,
     passed_on_retry, failed, error, xfailed, xpassed, and their
     ``_on_retry`` variants). Mirrors the all-tests CSV but typed and
     partition-pruned for Athena. Drives ad-hoc per-test analysis:
     longest-duration tests, xfail categorization, status churn.

The schemas, partitioning convention, and component split are documented
in ``docs/pt_upstream/grafana_athena_setup.md``. Keep that doc and this
module in sync — the Athena DDL there relies on the column names and
types defined here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from aggregate_upstream_results import (
    FAILURE_STATUSES,
    SKIPPED_STATUSES,
    _calc_pass_rate,
    _file_from_row,
    _get_run_metadata,
)
from persistent_cache import CACHE_METRICS_FILE_NAME

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


def _stats_to_dict(
    stats: dict,
    component: str,
    metadata: dict,
    run_timestamp: str,
    persistent_cache_present: bool | None = None,
) -> dict:
    """Build a single per-component row in the schema run_metrics expects."""
    row = {
        "commit_sha": metadata["commit_sha"],
        "branch": metadata["branch"],
        "run_id": metadata["run_id"],
        "workflow_type": metadata["workflow_type"],
        "run_timestamp": run_timestamp,
        "component": component,
        # Run-level flag (same on every component row): did this run hydrate a
        # non-empty persistent NEFF cache? NULL for runs/images without the
        # cache feature, so history degrades gracefully (like workflow_type did).
        "persistent_cache_present": persistent_cache_present,
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


# The five run-identity columns every per-row table repeats. run_timestamp is a
# datetime; the rest come from the run metadata dict.
_RUN_METADATA_COLUMNS = ("commit_sha", "branch", "run_id", "workflow_type")


def _append_run_metadata(
    columnar: dict[str, list], metadata: dict, run_timestamp: datetime
) -> None:
    """Append the shared run-identity columns for one row."""
    for col in _RUN_METADATA_COLUMNS:
        columnar[col].append(metadata.get(col, "unknown"))
    columnar["run_timestamp"].append(run_timestamp)


def _make_run_metrics_table(
    test_rows: list[dict],
    metadata: dict,
    run_timestamp: datetime,
    persistent_cache_present: bool | None = None,
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
    rows.append(
        _stats_to_dict(
            overall, "overall", metadata, run_timestamp.isoformat(), persistent_cache_present
        )
    )
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
                persistent_cache_present,
            )
        )

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("component", pa.string()),
            ("persistent_cache_present", pa.bool_()),
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
            ("workflow_type", pa.string()),
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

        _append_run_metadata(columnar, metadata, run_timestamp)
        # ``component`` IS NOT derivable from test_id without re-running
        # the file→component mapping, so we keep it as its own column.
        columnar["component"].append(_component_for(_file_from_row(row)))
        columnar["test_id"].append(row.get("test_id") or row.get("nodeid") or "")
        columnar["status"].append(row.get("status") or "")
        columnar["failure_reason"].append(first_line)
        columnar["duration_sec"].append(duration)

    return pa.table(columnar, schema=schema)


# ---------------------------------------------------------------------------
# test_aggregated — one row per non-skipped test
# ---------------------------------------------------------------------------

# Rules to collapse the first line of an error into an ``error_category``
# bucket dashboards can ``GROUP BY``. Each rule replaces a variable token
# (op name, worker ID, ...) with a stable placeholder so semantically
# identical errors hit one bucket. Add a new rule when a new error family
# fragments existing buckets in real data — the raw first line is always
# preserved in ``error_signature``.
_ERROR_CATEGORY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpInfo dtype-mismatch: "for <op> on device type <dev>"
    (re.compile(r" for [\w.]+ on device type \w+ "), " for <op> on device type <dev> "),
    # NRT errors: "operation=aten::scatter_add_" → "operation=aten::<op>"
    (re.compile(r"operation=aten::[\w.]+"), "operation=aten::<op>"),
    # Worker-crash messages embed the test nodeid; collapse so all
    # crashes share one bucket (the nodeid is in the test_id column).
    (
        re.compile(r"worker '[^']+' crashed while running '[^']+'"),
        "worker '<gw>' crashed while running '<nodeid>'",
    ),
)


def _normalize_signature(first_line: str) -> str:
    """Collapse dynamic tokens in ``first_line`` per ``_ERROR_CATEGORY_RULES``."""
    if not first_line:
        return ""
    out = first_line
    for pattern, repl in _ERROR_CATEGORY_RULES:
        out = pattern.sub(repl, out)
    return out


def _make_test_aggregated_table(
    test_rows: list[dict], metadata: dict, run_timestamp: datetime
) -> pa.Table:
    """Build the test_aggregated table — one row per non-skipped test.

    Filter mirrors ``write_all_tests_csv`` (skipped excluded). Schema
    diverges from ``test_failures`` by materializing ``test_class`` /
    ``test_method`` (so dashboards group without regex), and by emitting
    the raw first line as ``error_signature`` plus the normalized form as
    ``error_category`` — see ``_ERROR_CATEGORY_RULES``.
    """
    # Sort by (component, test_id) so per-row-group min/max stats are
    # tight enough for Athena's component-filter predicate pushdown.
    rows = sorted(
        (r for r in test_rows if r.get("status") not in SKIPPED_STATUSES),
        key=lambda r: (
            _component_for(_file_from_row(r)),
            r.get("test_id") or r.get("nodeid") or "",
        ),
    )

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("component", pa.string()),
            ("test_id", pa.string()),
            ("test_class", pa.string()),
            ("test_method", pa.string()),
            ("status", pa.string()),
            ("duration_sec", pa.float64()),
            ("failure_reason", pa.string()),
            ("error_signature", pa.string()),
            ("error_category", pa.string()),
        ]
    )

    columnar: dict[str, list] = {name: [] for name in schema.names}
    for row in rows:
        try:
            duration = float(row.get("duration(sec)", 0) or 0)
        except (TypeError, ValueError):
            duration = 0.0

        # failure_reason is already capped by extract_test_data_from_json.
        reason = (row.get("failure_reason") or "").strip()
        signature = next((ln.strip() for ln in reason.split("\n") if ln.strip()), "")
        category = _normalize_signature(signature)

        _append_run_metadata(columnar, metadata, run_timestamp)
        columnar["component"].append(_component_for(_file_from_row(row)))
        columnar["test_id"].append(row.get("test_id") or row.get("nodeid") or "")
        columnar["test_class"].append(row.get("test_class") or "")
        columnar["test_method"].append(row.get("test_method") or "")
        columnar["status"].append(row.get("status") or "")
        columnar["duration_sec"].append(duration)
        columnar["failure_reason"].append(reason)
        columnar["error_signature"].append(signature)
        columnar["error_category"].append(category)

    return pa.table(columnar, schema=schema)


def write_run_metrics_parquet(
    test_rows: list[dict],
    output_path: Path,
    persistent_cache_present: bool | None = None,
) -> None:
    """Write the run_metrics parquet file.

    One row per component, plus one ``component='overall'`` roll-up row.
    Six rows per run with the current 5-bucket split.

    ``persistent_cache_present`` is a run-level flag (True/False if the run used
    the persistent NEFF cache, None if the feature wasn't active) stamped
    onto every row.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_run_metrics_table(
        test_rows, _get_run_metadata(), _resolve_run_timestamp(), persistent_cache_present
    )
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Run metrics parquet: {output_path} ({table.num_rows} rows)")


def write_test_failures_parquet(test_rows: list[dict], output_path: Path) -> None:
    """Write the test_failures parquet file (failed/error rows only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_test_failures_table(test_rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Test failures parquet: {output_path} ({table.num_rows} rows)")


def write_test_aggregated_parquet(test_rows: list[dict], output_path: Path) -> None:
    """Write the test_aggregated parquet — one row per non-skipped test.

    Mirrors the layout of ``upstream_pytorch_all_tests_aggregated_<ts>.csv``
    but typed and partition-pruned for Athena. Skipped tests (which are
    the majority of rows on a typical run) live only in the skipped CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = _make_test_aggregated_table(test_rows, _get_run_metadata(), _resolve_run_timestamp())
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Test aggregated parquet: {output_path} ({table.num_rows} rows)")


# Numeric fields carried from the sidecar. Absent keys default to NULL so a
# download-only run (no upload_* keys) still writes a well-formed row.
_CACHE_METRIC_INT_FIELDS = (
    "download_bytes",
    "upload_bytes",
    "neffs_before",
    "neffs_after",
)
_CACHE_METRIC_FLOAT_FIELDS = (
    "download_seconds",
    "upload_seconds",
)


def collect_cache_metrics(results_dir: Path) -> list[dict]:
    """Read the per-job ``cache_metrics.json`` sidecars, deduplicated by job_name.

    The glob matches each sidecar twice (per-node artifact + re-bundled dir), so
    we keep one entry per job to avoid double-counting under a dashboard SUM.
    Returns [] when none are present. Never raises.
    """
    by_job: dict[str, dict] = {}
    for path in results_dir.glob(f"upstream-pytorch-*/**/{CACHE_METRICS_FILE_NAME}"):
        try:
            with open(path) as f:
                sidecar = json.load(f)
        except Exception as exc:
            logger.warning("Could not read cache metrics sidecar %s: %s", path, exc)
            continue
        # Key on job_name (the shard's cache scope). Fall back to the path so a
        # malformed sidecar without job_name still counts once rather than
        # collapsing all such sidecars into a single entry.
        by_job[sidecar.get("job_name") or str(path)] = sidecar
    logger.info("Collected %d unique cache metrics sidecar(s)", len(by_job))
    return list(by_job.values())


def run_persistent_cache_present(sidecars: list[dict]) -> bool | None:
    """Run-level ``persistent_cache_present`` from the per-job sidecars.

    None when no sidecar was written (feature off) so run_metrics reads NULL
    rather than a misleading False; else True if any job hydrated a non-empty cache.
    """
    if not sidecars:
        return None
    return any(bool(s.get("persistent_cache_present")) for s in sidecars)


def write_cache_metrics_parquet(
    results_dir: Path,
    metadata: dict,
    run_timestamp: datetime,
    output_path: Path,
    sidecars: list[dict] | None = None,
) -> Path | None:
    """Write the cache_metrics parquet (one row per job) from the per-job sidecars.

    ``sidecars`` may be passed in to reuse an earlier collect; otherwise they are
    collected from ``results_dir``. Returns the output path, or None when none
    were found (the table is skipped for that run).
    """
    if sidecars is None:
        sidecars = collect_cache_metrics(results_dir)
    if not sidecars:
        logger.warning("No cache_metrics.json sidecars found; skipping cache_metrics parquet")
        return None

    schema = pa.schema(
        [
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("run_id", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            ("job_name", pa.string()),
            ("persistent_cache_present", pa.bool_()),
            *((field, pa.int64()) for field in _CACHE_METRIC_INT_FIELDS),
            *((field, pa.float64()) for field in _CACHE_METRIC_FLOAT_FIELDS),
        ]
    )

    columnar: dict[str, list] = {name: [] for name in schema.names}
    for s in sidecars:
        _append_run_metadata(columnar, metadata, run_timestamp)
        columnar["job_name"].append(s.get("job_name") or "unknown")
        columnar["persistent_cache_present"].append(bool(s.get("persistent_cache_present")))
        for field in _CACHE_METRIC_INT_FIELDS:
            v = s.get(field)
            columnar[field].append(int(v) if v is not None else None)
        for field in _CACHE_METRIC_FLOAT_FIELDS:
            v = s.get(field)
            columnar[field].append(float(v) if v is not None else None)

    table = pa.table(columnar, schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    logger.info(f"Cache metrics parquet: {output_path} ({table.num_rows} rows)")
    return output_path
