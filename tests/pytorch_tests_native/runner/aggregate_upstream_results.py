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


def aggregate_summary(results_dir: Path, output_csv: Path) -> dict:
    """Aggregate test stats from all node artifacts and write summary CSV.

    Args:
        results_dir: Directory containing per-node artifact subdirectories,
            each with ``json_reports/test_results_*.json`` files (the structure
            produced by ``run_upstream_tests.sh`` and preserved by
            ``upload-artifact``/``download-artifact``).
        output_csv: Path where the summary CSV is written.

    Returns:
        Dict with ``test_rows`` (list of per-test dicts), ``stats`` (nested
        dict: class → status → count), ``durations`` (dict: class → total sec),
        and ``total_stats`` (dict: status → count across all classes).
    """
    # Find all pytest JSON report files across all node artifacts.
    #
    # Each per-node artifact contains the same report at two subpaths
    # (outputs/… and test-results/…) because run_test.sh copies outputs to
    # shared storage for the GH Actions runner to upload. Use the outputs/
    # path consistently to avoid double-counting.
    json_files = glob.glob(
        str(
            results_dir
            / "upstream-pytorch-*/outputs/upstream_pytorch_artifacts"
            / "json_reports/test_results_*.json"
        )
    )

    if not json_files:
        logger.warning(f"No JSON files found under {results_dir}")
        return {"test_rows": [], "stats": {}, "durations": {}, "total_stats": {}}

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

    # Aggregate counts per class — dynamically track all statuses
    stats = defaultdict(lambda: defaultdict(int))  # {class: {status: count}}
    durations = defaultdict(float)  # {class: total_duration}
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
            status = row.get("status") or "unknown"
            stats[cls][status] += 1
            all_statuses.add(status)
            with contextlib.suppress(ValueError):
                durations[cls] += float(row.get("duration(sec)", 0))
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
        "total_stats": dict(total_stats),
        "sorted_statuses": sorted_statuses,
    }


def write_github_summary(result: dict) -> None:
    """Write a GitHub Actions job summary (markdown table)."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    stats = result["stats"]
    durations = result["durations"]
    sorted_statuses = result["sorted_statuses"]
    total_stats = result["total_stats"]

    with open(summary_file, "a") as f:
        f.write("## PT Upstream Test Results\n\n")
        header = (
            f"| test_class | {' | '.join(sorted_statuses)} "
            "| total | pass_rate(%) | duration(sec) |\n"
        )
        f.write(header)
        f.write("|---" * (len(sorted_statuses) + 4) + "|\n")

        totals = [0] * len(sorted_statuses)
        total_duration = 0.0
        for cls in sorted(stats.keys()):
            counts = [stats[cls][s] for s in sorted_statuses]
            pass_rate = _calc_pass_rate(stats[cls])
            f.write(
                f"| {cls} | {' | '.join(str(c) for c in counts)} | "
                f"{sum(counts)} | {pass_rate} | {round(durations[cls], 3)} |\n"
            )
            totals = [t + c for t, c in zip(totals, counts, strict=False)]
            total_duration += durations[cls]

        total_pass_rate = _calc_pass_rate(total_stats)
        f.write(
            f"| **TOTAL** | {' | '.join(str(t) for t in totals)} | "
            f"**{sum(totals)}** | **{total_pass_rate}** | **{round(total_duration, 3)}** |\n"
        )


SKIPPED_STATUSES = ("skipped", "skipped_on_retry")


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
    """Concatenate per-node benchmark_results.jsonl files into one."""
    jsonl_files = glob.glob(
        str(
            results_dir
            / "upstream-pytorch-*/outputs/upstream_pytorch_artifacts"
            / "benchmark_results.jsonl"
        )
    )
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


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/upstream-results")
    run_timestamp = os.environ.get("TEST_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = results_dir / f"upstream_pytorch_summary_{run_timestamp}.csv"
    all_tests_csv = results_dir / f"upstream_pytorch_all_tests_aggregated_{run_timestamp}.csv"
    failures_csv = results_dir / f"upstream_pytorch_failures_{run_timestamp}.csv"
    flaky_csv = results_dir / f"upstream_pytorch_flaky_{run_timestamp}.csv"
    skipped_csv = results_dir / f"upstream_pytorch_skipped_{run_timestamp}.csv"
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
    merge_benchmark_jsonl(results_dir, benchmark_jsonl)

    # Print the summary CSV to stdout for log visibility
    print("=" * 60)
    print("  PT Upstream Test Results (Aggregated)")
    print("=" * 60)
    with open(summary_csv) as f:
        sys.stdout.write(f.read())
    print("=" * 60)

    write_github_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
