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
        nodeid, test_class, test_method, status, duration(sec),
        failure_reason, timestamp.
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

            # test_id: short nodeid for easy re-running, e.g.
            # "test_ops.py::TestOpsNeuron::test_add_cpu_float32"
            # Strips the leading path up to and including "/test/".
            test_id = nodeid.split("/test/")[-1] if "/test/" in nodeid else nodeid

            test_rows.append(
                {
                    "nodeid": nodeid,
                    "test_id": test_id,
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


# Status assigned when a main-run failure passes on the retry pass.
# Kept distinct from "passed" so dashboards can track retry-dependent
# flakiness as a signal of infrastructure issues (NRT worker crashes,
# resource contention, etc.). Counted as a pass by _calc_pass_rate.
PASSED_ON_RETRY_STATUS = "passed_on_retry"


def _on_retry_status(retry_status: str) -> str:
    """Map a retry outcome to its ``*_on_retry`` variant.

    Every non-failing retry outcome gets an ``_on_retry`` suffix so
    dashboards can distinguish "test was always skipped" from "test
    was a worker-crash victim that retry revealed as skipped".

    In pass-rate math each ``*_on_retry`` status counts the same as
    its base status (e.g. ``skipped_on_retry`` is excluded from the
    denominator just like ``skipped``).
    """
    return f"{retry_status}_on_retry"


def apply_retry_outcomes(
    main_rows: list[dict],
    retry_rows: list[dict],
) -> list[dict]:
    """Use retry outcomes as authoritative for main-run failures/errors.

    The main run reports ``failed`` for any test that was assigned to a
    pytest-xdist worker that crashed (NRT segfault, OOM, etc.), even
    though the test itself never ran. The retry re-executes those
    tests in a fresh worker, producing the real outcome.

    For main rows with status ``failed`` or ``error``, when the retry
    reports a different outcome for the same nodeid, the retry result
    is authoritative:

    - ``passed`` / ``xpassed`` on retry → :data:`PASSED_ON_RETRY_STATUS`
      (flakiness signal; still counted as a pass in pass-rate math)
    - ``skipped`` / ``xfailed`` on retry → that status directly
      (the main "failure" was a worker-crash artifact; the test was
      correctly skip/xfail-marked all along, so no flakiness signal
      is warranted — these become the test's real status)
    - ``failed`` / ``error`` on retry → keep the main row unchanged
      (genuine failure; do not swallow the original reason)

    Retry never overrides a passing main-run result.

    Only main-run entries are returned — retry entries are used purely
    to look up the authoritative outcome. The retry JSON contains only
    the previously-failing subset (via ``pytest --last-failed``), so
    including its rows would double-count tests.

    Args:
        main_rows: Rows extracted from the main-run JSON report.
        retry_rows: Rows extracted from the retry JSON report.

    Returns:
        Copy of main_rows with statuses updated per the rules above.
    """
    # Build fast lookup from retry results by nodeid. If a nodeid
    # appears more than once (shouldn't normally happen since retry
    # runs the same unique nodeids), last one wins — this is
    # intentional to prefer the most recent observation.
    retry_by_nodeid: dict[str, dict] = {
        row["nodeid"]: row for row in retry_rows if row.get("nodeid")
    }

    promoted = 0
    out: list[dict] = []
    for row in main_rows:
        nodeid = row.get("nodeid", "")
        status = row.get("status", "")
        if status in ("failed", "error") and nodeid in retry_by_nodeid:
            retry_row = retry_by_nodeid[nodeid]
            retry_status = retry_row.get("status", "")

            # If retry still reports a failure (failed/error), the main
            # row is the genuine result — keep it unchanged.
            if retry_status in ("failed", "error"):
                out.append(row)
                continue

            # Any other retry outcome (passed, xpassed, skipped, xfailed)
            # means the main-run "failure" was a worker-crash artifact.
            # Assign the *_on_retry variant so dashboards can distinguish
            # "always was X" from "worker-crash victim revealed as X".
            new_row = dict(row)
            new_row["status"] = _on_retry_status(retry_status)
            retry_reason = retry_row.get("failure_reason", "")
            original_reason = row.get("failure_reason", "")
            marker = f"[retry {retry_status}]"
            if retry_reason and original_reason:
                new_row["failure_reason"] = (
                    f"{marker} {retry_reason} (originally: {original_reason})"
                )
            elif retry_reason:
                new_row["failure_reason"] = f"{marker} {retry_reason}"
            elif original_reason:
                new_row["failure_reason"] = f"{marker} originally: {original_reason}"
            else:
                new_row["failure_reason"] = marker
            out.append(new_row)
            promoted += 1
            continue
        out.append(row)

    if promoted:
        logger.info(f"Promoted {promoted} main-run failures to *_on_retry statuses")
    return out


def json_to_csv(json_path: str, csv_path: str, retry_json_path: str | None = None) -> bool:
    """Convert a single pytest JSON report to CSV.

    Args:
        json_path: Path to input JSON report.
        csv_path: Path to output CSV file.
        retry_json_path: Optional companion retry JSON whose results are
            merged into the main rows (see :func:`apply_retry_outcomes`).

    Returns:
        True if CSV was written (even if empty), False on error.
    """
    rows = extract_test_data_from_json(json_path)
    if retry_json_path and os.path.exists(retry_json_path):
        retry_rows = extract_test_data_from_json(retry_json_path)
        rows = apply_retry_outcomes(rows, retry_rows)

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
        # Strip internal-only columns from the output.
        # ``test_id`` carries the full test identifier; ``test_class``
        # and ``test_method`` are redundant (derivable from test_id).
        fieldnames = [k for k in rows[0] if k not in ("nodeid", "test_class", "test_method")]

    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Wrote {len(rows)} rows to {csv_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing CSV {csv_path}: {e}")
        return False


def convert_output_dir(output_dir: str) -> int:
    """Convert all pytest JSON reports in <output_dir>/json_reports/ to CSV.

    Writes one CSV per main JSON to <output_dir>/csv_reports/ with the same
    base filename (``test_results_<name>.json`` → ``test_results_<name>.csv``).
    If a ``test_results_<name>_retry.json`` sibling exists, its outcomes
    are merged into the main CSV via :func:`apply_retry_outcomes` — the
    retry JSON itself is not written as a standalone CSV to avoid
    double-counting.

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

    all_json_files = sorted(glob.glob(os.path.join(json_dir, "test_results_*.json")))
    if not all_json_files:
        logger.info(f"No test_results_*.json files in {json_dir}")
        return 0

    # Separate main vs retry reports. Retry reports end in ``_retry.json``.
    main_json_files = [p for p in all_json_files if not p.endswith("_retry.json")]
    logger.info(
        f"Converting {len(main_json_files)} main JSON report(s) to CSV in {csv_dir} "
        f"({len(all_json_files) - len(main_json_files)} retry report(s) will be merged)"
    )

    failures = 0
    for json_path in main_json_files:
        csv_basename = os.path.basename(json_path).replace(".json", ".csv")
        csv_path = os.path.join(csv_dir, csv_basename)
        retry_json_path = json_path.replace(".json", "_retry.json")
        if not os.path.exists(retry_json_path):
            retry_json_path = None
        if not json_to_csv(json_path, csv_path, retry_json_path=retry_json_path):
            failures += 1

    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: convert_reports.py <output_dir>", file=sys.stderr)
        return 2
    return convert_output_dir(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
