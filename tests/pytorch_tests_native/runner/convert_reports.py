#!/usr/bin/env python3
"""Convert pytest JSON reports to CSV format for upstream PyTorch tests.

Used for two purposes:

1. Per-node CSV generation — called at the end of ``run_upstream_tests.sh``
   after pytest completes. Writes ``csv_reports/*.csv`` alongside
   ``json_reports/*.json`` so each per-node artifact contains both JSON and
   CSV forms of the test results.

2. Shared helpers — imported by ``aggregate_upstream_results.py`` which
   reuses ``extract_test_data_from_json`` for the aggregation pass.

Usage (standalone):
    python3 convert_reports.py <output_dir>

    Walks ``<output_dir>/json_reports/test_results_*.json`` and produces one
    CSV per JSON file in ``<output_dir>/csv_reports/``. Safe to call even if
    no JSON files exist (returns 0).
"""

import csv
import glob
import json
import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def extract_test_data_from_json(json_file: str) -> list[dict]:
    """Extract test data from a pytest JSON report.

    Args:
        json_file: Path to pytest JSON report file.

    Returns:
        List of dictionaries with keys:
        test_class, test_method, status, duration(sec), failure_reason, timestamp.
    """
    try:
        with open(json_file) as f:
            data = json.load(f)

        test_rows = []
        for test in data.get("tests", []):
            nodeid = test.get("nodeid", "")
            test_class = ""
            test_method = ""

            if "::" in nodeid:
                parts = nodeid.split("::")
                if len(parts) >= 3:
                    test_class = parts[1]
                    test_method = parts[2]
                elif len(parts) == 2:
                    test_class = parts[0].split("/")[-1].replace(".py", "")
                    test_method = parts[1]

            setup_duration = test.get("setup", {}).get("duration", 0)
            call_duration = test.get("call", {}).get("duration", 0)
            teardown_duration = test.get("teardown", {}).get("duration", 0)
            total_duration = setup_duration + call_duration + teardown_duration

            # Extract failure reason for failed and xfailed tests
            failure_reason = ""
            outcome = test.get("outcome", "")
            if outcome in ("failed", "xfailed"):
                call_info = test.get("call", {})
                failure_reason = call_info.get("crash", {}).get("message", "")
                if not failure_reason and "longrepr" in call_info:
                    failure_reason = str(call_info["longrepr"]).split("\n")[-1]
                failure_reason = failure_reason[:500]

            test_rows.append(
                {
                    "test_class": test_class,
                    "test_method": test_method,
                    "status": outcome,
                    "duration(sec)": round(total_duration, 3),
                    "failure_reason": failure_reason,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return test_rows
    except Exception as e:
        logger.error(f"Error processing {json_file}: {e}")
        return []


def json_to_csv(json_path: str, csv_path: str) -> bool:
    """Convert a single pytest JSON report to CSV.

    Args:
        json_path: Path to input JSON report.
        csv_path: Path to output CSV file.

    Returns:
        True if CSV was written (even if empty), False on error.
    """
    rows = extract_test_data_from_json(json_path)
    if not rows:
        logger.info(f"No rows extracted from {json_path} — writing empty CSV")
        fieldnames = [
            "test_class",
            "test_method",
            "status",
            "duration(sec)",
            "failure_reason",
            "timestamp",
        ]
    else:
        fieldnames = list(rows[0].keys())

    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Wrote {len(rows)} rows to {csv_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing CSV {csv_path}: {e}")
        return False


def convert_output_dir(output_dir: str) -> int:
    """Convert all pytest JSON reports in <output_dir>/json_reports/ to CSV.

    Writes one CSV per JSON to <output_dir>/csv_reports/ with the same base
    filename (``test_results_<name>.json`` → ``test_results_<name>.csv``).

    Args:
        output_dir: The --output-dir passed to run_upstream_tests.sh.

    Returns:
        Exit code (0 = success, >0 = one or more failures).
    """
    json_dir = os.path.join(output_dir, "json_reports")
    csv_dir = os.path.join(output_dir, "csv_reports")

    if not os.path.isdir(json_dir):
        logger.warning(f"No json_reports directory at {json_dir} — nothing to convert")
        return 0

    json_files = sorted(glob.glob(os.path.join(json_dir, "test_results_*.json")))
    if not json_files:
        logger.info(f"No test_results_*.json files in {json_dir}")
        return 0

    logger.info(f"Converting {len(json_files)} JSON report(s) to CSV in {csv_dir}")

    failures = 0
    for json_path in json_files:
        csv_basename = os.path.basename(json_path).replace(".json", ".csv")
        csv_path = os.path.join(csv_dir, csv_basename)
        if not json_to_csv(json_path, csv_path):
            failures += 1

    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: convert_reports.py <output_dir>", file=sys.stderr)
        return 2
    return convert_output_dir(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
