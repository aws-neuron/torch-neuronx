#!/usr/bin/env python3
"""Aggregate PyTorch upstream test results from all matrix nodes.

Produces a per-class CSV summary with dynamic status columns, ``pass_rate(%)``,
``duration(sec)``, and a TOTAL row. Writes a markdown table to the GitHub
Actions job summary and prints the CSV contents to stdout for log visibility.

Flow:
    1. Download per-node artifacts (done by the caller workflow)
    2. For each node, parse the pytest JSON reports
    3. Aggregate by test_class with dynamic status columns
    4. Compute pass_rate excluding skipped tests
    5. Write the summary CSV + a GitHub Actions job summary
"""

import contextlib
import csv
import glob
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Shared helper lives in convert_reports so per-node and aggregation flows stay
# in sync. Supports being imported as a sibling module regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_reports import (
    apply_retry_outcomes,
    extract_test_data_from_json,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Statuses considered "failing" for the failures-only CSV. ``error`` covers
# pytest collection / fixture errors. ``passed_on_retry`` is intentionally
# EXCLUDED — those tests ultimately passed and are tracked separately as a
# flakiness signal, not as failures that need investigation.
FAILURE_STATUSES = ("failed", "error")

# Statuses that count as "this test didn't actually run" — used to filter
# all-skipped classes out of the rendered GH summary panel and to split
# skipped rows into their own CSV in the aggregator.
SKIPPED_STATUSES = ("skipped", "skipped_on_retry")


def _get_run_metadata() -> dict:
    """Pull commit/branch/run-id from GH Actions env vars.

    Defaults to ``"unknown"`` outside CI so dashboards can still distinguish
    "no metadata available" from "missing column."
    """
    return {
        "commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "branch": (os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF", "unknown")),
        "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
    }


def write_run_metadata_csv(result: dict, output_csv: Path) -> None:
    """Write a single-row sidecar CSV with run metadata + roll-up stats.

    Lets dashboards attribute pass rate / failure counts to a specific
    commit without denormalizing the metadata onto every per-test row.
    The full per-test CSVs stay lean; this is the join key for time-series
    plots ("pass rate over time on main, with each point = one run").
    """
    metadata = _get_run_metadata()
    total_stats = result["total_stats"]
    total_count = sum(total_stats.values())
    pass_rate = _calc_pass_rate(total_stats)
    total_duration = round(sum(result["durations"].values()), 3)

    fieldnames = [
        "commit_sha",
        "branch",
        "run_id",
        "run_timestamp",
        "total",
        "passed",
        "passed_on_retry",
        "failed",
        "skipped",
        "skipped_on_retry",
        "xfailed",
        "xfailed_on_retry",
        "xpassed",
        "xpassed_on_retry",
        "error",
        "pass_rate(%)",
        "duration(sec)",
    ]
    row = {
        **metadata,
        "run_timestamp": os.environ.get(
            "TEST_RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        "total": total_count,
        "pass_rate(%)": pass_rate,
        "duration(sec)": total_duration,
        # Per-status counts; default 0 for absent statuses.
        **{
            k: total_stats.get(k, 0)
            for k in fieldnames
            if k
            not in {
                "commit_sha",
                "branch",
                "run_id",
                "run_timestamp",
                "total",
                "pass_rate(%)",
                "duration(sec)",
            }
        },
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Run metadata CSV: {output_csv}")


def _calc_pass_rate(status_counts: dict) -> str:
    """Calculate pass rate excluding skipped tests.

    Each ``*_on_retry`` status counts the same as its base status:
    - ``passed_on_retry`` / ``xpassed_on_retry`` → pass (numerator)
    - ``xfailed_on_retry`` → denom only (same as xfailed)
    - ``skipped_on_retry`` → excluded (same as skipped)
    - ``failed`` / ``error`` → denom only
    """
    passed = (
        status_counts.get("passed", 0)
        + status_counts.get("xpassed", 0)
        + status_counts.get("passed_on_retry", 0)
        + status_counts.get("xpassed_on_retry", 0)
    )
    denom = (
        passed
        + status_counts.get("failed", 0)
        + status_counts.get("xfailed", 0)
        + status_counts.get("xfailed_on_retry", 0)
        + status_counts.get("error", 0)
    )
    if denom == 0:
        return "N/A"
    return f"{passed / denom * 100:.1f}"


def _file_from_row(row: dict) -> str:
    """Extract the test file from a row's test_id (or nodeid as fallback).

    ``test_id`` is shaped like ``test_ops.py::TestClass::test_method`` or
    ``dynamo/test_x.py::TestY::test_z`` — the leading segment before the
    first ``::`` is the file. Sharded files collapse together because
    pytest-shard doesn't tag tests with the shard id.
    """
    test_id = row.get("test_id") or row.get("nodeid") or ""
    if "::" in test_id:
        return test_id.split("::", 1)[0]
    return test_id or "unknown"


def _group_from_file(test_file: str) -> str:
    """Collapse a file path into its top-level group for the panel breakdown.

    Files under a subdirectory (``dynamo/test_x.py``) collapse to that
    directory (``dynamo/``); root-level files (``test_ops.py``) keep their
    own row. Keeps the panel readable on the full suite (~hundreds of
    files → ~tens of groups).
    """
    if "/" in test_file:
        return test_file.split("/", 1)[0] + "/"
    return test_file


def aggregate_summary(results_dir: Path, output_csv: Path) -> dict:
    """Aggregate test stats from all node artifacts and write summary CSV.

    Args:
        results_dir: Directory containing per-node artifact subdirectories,
            each with ``test_results_*.json`` files directly inside (no
            nested ``json_reports/`` layer — the runner writes them flat).
        output_csv: Path where the summary CSV is written.

    Returns:
        Dict with ``test_rows`` (list of per-test dicts), ``stats`` (nested
        dict: class → status → count), ``durations`` (dict: class → total sec),
        ``total_stats`` (dict: status → count across all classes), plus
        parallel ``file_stats`` and ``file_durations`` keyed by test file
        for the GH panel render (file-level instead of class-level).
    """
    # Per-node artifacts may contain two copies of each report (one under
    # `outputs/upstream_pytorch_artifacts/` and a `test-results/` mirror).
    # Dedupe by (per-node-dir, filename); prefer the `outputs/` copy.
    candidates = list(results_dir.glob("upstream-pytorch-*/**/test_results_*.json"))

    by_node_and_name: dict[tuple[str, str], str] = {}
    for p in candidates:
        # Per-node dir is the immediate child of results_dir.
        try:
            node_dir = p.relative_to(results_dir).parts[0]
        except ValueError:
            continue
        key = (node_dir, p.name)
        existing = by_node_and_name.get(key)
        if existing is None or "/outputs/" in str(p):
            by_node_and_name[key] = str(p)
    json_files = list(by_node_and_name.values())

    if not json_files:
        logger.warning(f"No JSON files found under {results_dir}")
        return {
            "test_rows": [],
            "stats": {},
            "durations": {},
            "file_stats": {},
            "file_durations": {},
            "group_stats": {},
            "group_durations": {},
            "total_stats": {},
        }

    # Split main vs retry reports. Retry files end in ``_retry.json``
    # and are merged into their paired main report via
    # :func:`apply_retry_outcomes` (see convert_reports). Retry
    # reports are NOT tallied on their own to prevent double-counting:
    # the retry JSON only contains the previously-failing subset.
    main_json_files = [p for p in json_files if not p.endswith("_retry.json")]
    retry_json_files = {p for p in json_files if p.endswith("_retry.json")}
    logger.info(
        f"Found {len(main_json_files)} main + {len(retry_json_files)} retry "
        f"JSON report files to process"
    )

    # Aggregate counts per class (canonical / CSV), per file, and per
    # top-level group (panel — collapses dynamo/foo.py + dynamo/bar.py
    # into a single "dynamo/" row to keep the panel readable).
    stats = defaultdict(lambda: defaultdict(int))  # {class: {status: count}}
    durations = defaultdict(float)  # {class: total_duration}
    file_stats = defaultdict(lambda: defaultdict(int))  # {file: {status: count}}
    file_durations = defaultdict(float)  # {file: total_duration}
    group_stats = defaultdict(lambda: defaultdict(int))  # {group: {status: count}}
    group_durations = defaultdict(float)  # {group: total_duration}
    all_statuses = set()
    all_test_rows = []

    for json_file in main_json_files:
        main_rows = extract_test_data_from_json(json_file)

        # Merge companion retry report, if present. Sibling lookup by
        # filename so we stay within the same per-node artifact dir.
        retry_path = json_file.replace(".json", "_retry.json")
        if retry_path in retry_json_files:
            retry_rows = extract_test_data_from_json(retry_path)
            main_rows = apply_retry_outcomes(main_rows, retry_rows)

        for row in main_rows:
            cls = row.get("test_class") or "unknown"
            test_file = _file_from_row(row)
            group = _group_from_file(test_file)
            status = row.get("status") or "unknown"
            stats[cls][status] += 1
            file_stats[test_file][status] += 1
            group_stats[group][status] += 1
            all_statuses.add(status)
            with contextlib.suppress(ValueError):
                duration = float(row.get("duration(sec)", 0))
                durations[cls] += duration
                file_durations[test_file] += duration
                group_durations[group] += duration
            all_test_rows.append(row)

    # Create summary CSV with dynamic columns
    sorted_statuses = sorted(all_statuses)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_class", *sorted_statuses, "total", "pass_rate(%)", "duration(sec)"])
        totals = [0] * len(sorted_statuses)
        total_duration = 0.0
        for cls in sorted(stats.keys()):
            counts = [stats[cls][s] for s in sorted_statuses]
            pass_rate = _calc_pass_rate(stats[cls])
            writer.writerow([cls, *counts, sum(counts), pass_rate, round(durations[cls], 3)])
            totals = [t + c for t, c in zip(totals, counts, strict=False)]
            total_duration += durations[cls]
        # Totals row
        total_stats = defaultdict(int)
        for cls_stats in stats.values():
            for s, c in cls_stats.items():
                total_stats[s] += c
        total_pass_rate = _calc_pass_rate(total_stats)
        writer.writerow(["TOTAL", *totals, sum(totals), total_pass_rate, round(total_duration, 3)])

    logger.info(f"Summary saved to {output_csv}")

    return {
        "test_rows": all_test_rows,
        "stats": dict(stats),
        "durations": dict(durations),
        "file_stats": {f: dict(s) for f, s in file_stats.items()},
        "file_durations": dict(file_durations),
        "group_stats": {g: dict(s) for g, s in group_stats.items()},
        "group_durations": dict(group_durations),
        "total_stats": dict(total_stats),
        "sorted_statuses": sorted_statuses,
    }


def _is_all_skipped(unit_stats: dict) -> bool:
    """A grouping (class or file) with no executed tests — only skips.

    These rows are pure noise in the GH summary panel: pass_rate is N/A and
    they outnumber the rows that actually ran. We hide them in the
    rendered markdown but keep them in the canonical summary CSV so
    dashboards and historical comparisons aren't affected.
    """
    return all(status in SKIPPED_STATUSES for status in unit_stats)


# Back-compat alias for tests that imported the old class-specific name.
_is_all_skipped_class = _is_all_skipped


def write_all_tests_csv(test_rows: list, output_csv: Path) -> None:
    """Write test results excluding skipped tests.

    Skipped tests typically dominate (90%+) and cloud analysis. They
    are written to a separate ``*_skipped_*.csv`` instead.
    """
    if not test_rows:
        return
    fieldnames = [
        "test_id",
        "status",
        "duration(sec)",
        "failure_reason",
        "timestamp",
    ]
    rows = [r for r in test_rows if r.get("status") not in SKIPPED_STATUSES]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"All-tests CSV (excl. skipped): {output_csv} ({len(rows)} rows)")


def write_skipped_csv(test_rows: list, output_csv: Path) -> int:
    """Write skipped tests to a separate CSV.

    Returns:
        Number of rows written.
    """
    skipped = [r for r in test_rows if r.get("status") in SKIPPED_STATUSES]
    skipped.sort(key=lambda r: (r.get("test_class") or "", r.get("test_method") or ""))

    fieldnames = [
        "test_id",
        "status",
        "duration(sec)",
        "failure_reason",
        "timestamp",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(skipped)

    logger.info(f"Skipped CSV: {output_csv} ({len(skipped)} rows)")
    return len(skipped)


def write_failures_only_csv(test_rows: list, output_csv: Path) -> int:
    """Write a failures-only slice of the aggregated test results.

    Companion to :func:`write_all_tests_csv` that filters to rows whose
    status is in :data:`FAILURE_STATUSES` (currently ``failed`` and
    ``error``). ``passed_on_retry`` is **not** considered a failure
    since the test ultimately passed — those tests are tracked as a
    flakiness signal elsewhere, not as outstanding bugs.

    Sorted by (test_class, test_method) so diffs between runs are
    stable. Writes the same columns as the all-tests CSV for easy
    side-by-side consumption.

    Returns:
        Number of rows written (0 if no failures).
    """
    failures = [row for row in test_rows if row.get("status") in FAILURE_STATUSES]

    # Stable ordering so the file is diffable run-to-run.
    failures.sort(key=lambda r: (r.get("test_class") or "", r.get("test_method") or ""))

    fieldnames = [
        "test_id",
        "status",
        "duration(sec)",
        "failure_reason",
        "timestamp",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Always write the file (even when empty) so downstream consumers
    # can trust the path exists after a successful aggregation pass.
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)

    logger.info(f"Failures-only CSV: {output_csv} ({len(failures)} rows)")
    return len(failures)


def write_flaky_csv(test_rows: list, output_csv: Path) -> int:
    """Write a flaky-tests report: tests that failed in main but recovered on retry.

    Captures every row whose status ends with ``_on_retry`` — these are
    tests that the main run reported as failed/error but the retry pass
    revealed a different (non-failing) outcome. This is the worker-crash
    flakiness signal.

    Sorted by (test_class, test_method) for stable diffs.

    Returns:
        Number of rows written (0 if no flaky tests).
    """
    flaky = [row for row in test_rows if row.get("status", "").endswith("_on_retry")]
    flaky.sort(key=lambda r: (r.get("test_class") or "", r.get("test_method") or ""))

    fieldnames = [
        "test_id",
        "status",
        "duration(sec)",
        "failure_reason",
        "timestamp",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flaky)

    logger.info(f"Flaky CSV: {output_csv} ({len(flaky)} rows)")
    return len(flaky)


def merge_benchmark_jsonl(results_dir: Path, output_jsonl: Path) -> None:
    """Concatenate per-node benchmark_results.jsonl files into one.

    Recursive glob handles both flat per-node artifact layouts and the
    nested ``test-results/...`` layout from internal compute.
    """
    jsonl_files = [
        str(p) for p in results_dir.glob("upstream-pytorch-*/**/benchmark_results.jsonl")
    ]
    if not jsonl_files:
        logger.info("No benchmark_results.jsonl files found")
        return
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with open(output_jsonl, "w") as out:
        for jf in sorted(jsonl_files):
            with open(jf) as inp:
                for line in inp:
                    line = line.rstrip("\n")
                    if line:
                        out.write(line + "\n")
                        total_lines += 1
    logger.info(f"Merged {len(jsonl_files)} JSONL files → {output_jsonl} ({total_lines} records)")


# File type → (glob, destination subdirectory for flattening).
# Reports live flat inside each per-node artifact; flatten_reports
# copies them into central ``<type>/`` directories for easier review.
# Logs are included so post-mortem evidence survives the per-node-dir
# cleanup that runs at the end of flatten_reports.
_FLATTEN_GROUPS = (
    ("json_reports", "test_results_*.json"),
    ("html_reports", "*.html"),
    ("csv_reports", "test_results_*.csv"),
    ("log_reports", "*.log"),
)


def flatten_reports(results_dir: Path) -> dict[str, int]:
    """Collect per-node reports into central ``<type>_reports/`` dirs.

    Output layout:

        results_dir/
        ├── json_reports/   # every node's test_results_*.json
        ├── html_reports/   # every node's *.html
        ├── csv_reports/    # every node's test_results_*.csv
        └── log_reports/    # every node's *.log (post-mortem evidence)

    Recursive glob handles both flat and nested ``test-results/...``
    per-node layouts. Per-node dirs are removed only when files were
    successfully flattened, so a glob miss doesn't destroy the originals.
    Returns per-type counts.
    """
    counts: dict[str, int] = {}
    total_copied = 0
    for dst_name, glob_pattern in _FLATTEN_GROUPS:
        dst_dir = results_dir / dst_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_files = list(results_dir.glob(f"upstream-pytorch-*/**/{glob_pattern}"))
        copied = 0
        for src in src_files:
            dst = dst_dir / src.name
            if dst.exists():
                logger.warning(f"Skipping duplicate {dst_name}/{dst.name} from {src}")
                continue
            shutil.copy2(src, dst)
            copied += 1

        counts[dst_name] = copied
        total_copied += copied
        logger.info(f"Flattened {copied} files → {dst_dir}/")

    # Per-node dirs are duplicated under the central <type>/ subdirs only
    # if we actually flattened files out of them. If the flatten globs
    # didn't match anything (layout regression, empty artifacts, etc.),
    # leave the per-node dirs alone so we don't accidentally throw away
    # the only copy of the data.
    if total_copied > 0:
        removed = 0
        for node_dir in glob.glob(str(results_dir / "upstream-pytorch-*")):
            if os.path.isdir(node_dir):
                shutil.rmtree(node_dir)
                removed += 1
        if removed:
            logger.info(f"Removed {removed} per-node upstream-pytorch-*/ dirs (deduped)")
    else:
        logger.warning(
            "flatten_reports found no files to flatten; leaving per-node "
            "upstream-pytorch-*/ dirs in place. Investigate the artifact "
            "layout if the central <type>/ dirs are empty."
        )

    return counts


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/upstream-results")
    run_timestamp = os.environ.get("TEST_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = results_dir / f"upstream_pytorch_summary_{run_timestamp}.csv"
    all_tests_csv = results_dir / f"upstream_pytorch_all_tests_aggregated_{run_timestamp}.csv"
    failures_csv = results_dir / f"upstream_pytorch_failures_{run_timestamp}.csv"
    flaky_csv = results_dir / f"upstream_pytorch_flaky_{run_timestamp}.csv"
    skipped_csv = results_dir / f"upstream_pytorch_skipped_{run_timestamp}.csv"
    metadata_csv = results_dir / f"upstream_pytorch_run_metadata_{run_timestamp}.csv"
    benchmark_jsonl = results_dir / "benchmark_results.jsonl"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    result = aggregate_summary(results_dir, summary_csv)

    if not result["test_rows"]:
        logger.warning("No test data found — skipping summary")
        return 0

    write_all_tests_csv(result["test_rows"], all_tests_csv)
    write_failures_only_csv(result["test_rows"], failures_csv)
    write_flaky_csv(result["test_rows"], flaky_csv)
    write_skipped_csv(result["test_rows"], skipped_csv)
    write_run_metadata_csv(result, metadata_csv)
    merge_benchmark_jsonl(results_dir, benchmark_jsonl)

    # Late import: aggregate_panel imports back from this module for shared
    # helpers (FAILURE_STATUSES, _calc_pass_rate, etc.). Top-level import
    # would create a circular dependency at module load.
    from aggregate_panel import write_github_summary

    write_github_summary(result)

    flatten_reports(results_dir)

    # Print the summary CSV to stdout for log visibility
    print("=" * 60)
    print("  PT Upstream Test Results (Aggregated)")
    print("=" * 60)
    with open(summary_csv) as f:
        sys.stdout.write(f.read())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
