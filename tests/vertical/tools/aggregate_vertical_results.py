#!/usr/bin/env python3
"""Convert vertical coverage CSV(s) into dashboard parquet.

This mirrors ``tests/pytorch_tests_native/runner/aggregate_upstream_results.py``:
it reads the aggregated coverage CSV that the vertical suite writes, then emits
typed parquet under a partitioned ``custom-test-artifacts`` tree. The workflow
uploads that tree with the ``custom-test-artifacts`` artifact name, and the
DevOps mirror copies it to S3 verbatim, where Athena reads it.

The tool is credential-free. It runs after the pytest run, in the same job.

Layout (one file per table per run), matching the upstream convention. Eager
and torch_compile are separate tables (op_coverage_eager /
op_coverage_torch_compile), the same split the dashboard already reads:

    custom-test-artifacts/vertical_ops/<table>/branch=<branch>/dt=<date>/run_<id>.parquet

Usage:
    python tests/vertical/tools/aggregate_vertical_results.py \
        --csv target_ops_coverage_20260908_120000.csv \
        --suite target_ops_coverage \
        --output-dir /tmp/vertical-results
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_vertical_parquet import (
    _get_run_metadata,
    read_coverage_csv,
    write_op_coverage_parquet,
    write_run_metrics_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Top-level dir the DevOps mirror copies to S3 verbatim.
ARTIFACT_ROOT = "custom-test-artifacts"
# Suite family under the artifact root -- keeps vertical tables separate from
# the upstream_pytorch tables in the same bucket.
TABLE_FAMILY = "vertical_ops"

# Map the coverage-CSV suite tag to the dashboard table variant. Each suite
# writes its own table (op_coverage_eager / op_coverage_torch_compile), so the
# existing dashboard queries keep working.
_SUITE_VARIANT = {
    "target_ops_coverage": "eager",
    "target_ops_compile_coverage": "torch_compile",
}


def _partition_path(root: Path, table: str, branch: str, run_id: str, run_date: str) -> Path:
    """Build the ``<table>/branch=.../dt=.../run_<id>.parquet`` partition path."""
    return (
        root
        / TABLE_FAMILY
        / table
        / f"branch={branch}"
        / f"dt={run_date}"
        / f"run_{run_id}.parquet"
    )


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True, help="Aggregated coverage CSV to convert.")
    parser.add_argument(
        "--suite",
        required=True,
        help=(
            "Coverage suite tag. Selects the dashboard table variant: "
            "target_ops_coverage -> eager, target_ops_compile_coverage -> torch_compile."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/vertical-results",
        help="Directory that receives the custom-test-artifacts tree.",
    )
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("Coverage CSV not found: %s", csv_path)
        return 1

    variant = _SUITE_VARIANT.get(args.suite)
    if variant is None:
        logger.error("Unknown suite %r. Pass one of %s.", args.suite, sorted(_SUITE_VARIANT))
        return 1

    rows = read_coverage_csv(csv_path)
    if not rows:
        logger.warning("No rows in %s; nothing to write.", csv_path)
        return 0

    metadata = _get_run_metadata()
    branch = metadata["branch"]
    run_id = metadata["run_id"]
    run_date = datetime.now().strftime("%Y-%m-%d")
    parquet_root = Path(args.output_dir) / ARTIFACT_ROOT

    write_run_metrics_parquet(
        rows,
        _partition_path(parquet_root, f"run_metrics_{variant}", branch, run_id, run_date),
    )
    write_op_coverage_parquet(
        rows,
        _partition_path(parquet_root, f"op_coverage_{variant}", branch, run_id, run_date),
    )
    logger.info(
        "Wrote parquet for %d ops (suite=%s, variant=%s) under %s",
        len(rows),
        args.suite,
        variant,
        parquet_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
