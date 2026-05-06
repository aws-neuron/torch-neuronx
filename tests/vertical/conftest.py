"""Pytest configuration for vertical tests.

Aggregates coverage results from test_target_ops_coverage.py → target_ops_coverage_<ts>.csv
"""

import csv
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

_COVERAGE_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_coverage_metrics"
)

COVERAGE_FIELDS = [
    "op_name",
    "op_group",
    "status",
    "supported_by",
    "input_source",
    "input_count",
    "neuron_primitive_ops",
    "fallback_primitive_ops",
    "time_to_first_ms",
    "exec_avg_ms",
    "exec_min_ms",
    "exec_max_ms",
    "error",
    "last_updated",
    "correctness_test",
]


def pytest_addoption(parser):
    parser.addoption(
        "--model-shapes-only",
        action="store_true",
        help="For ops with model-traced shapes, test only those (skip builtin shapes)",
    )
    parser.addoption(
        "--shape-registry",
        default=None,
        help="Path to JSON shape registry file to load model-traced shapes",
    )


def pytest_configure(config):
    if config.getoption("--model-shapes-only", default=False):
        os.environ["NEURON_MODEL_SHAPES_ONLY"] = "1"
    registry_path = config.getoption("--shape-registry", default=None)
    if registry_path:
        os.environ["NEURON_SHAPE_REGISTRY"] = str(registry_path)


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return
    if _COVERAGE_DIR.exists():
        for f in _COVERAGE_DIR.glob("*.json"):
            f.unlink()
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):
        return
    if not _COVERAGE_DIR.exists():
        return

    by_op: dict[str, dict] = {}
    for fp in sorted(_COVERAGE_DIR.glob("*.json")):
        try:
            row = json.loads(fp.read_text())
            by_op[row.get("op_name", "")] = row
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")

    rows = sorted(by_op.values(), key=lambda r: r.get("op_name", ""))
    if not rows:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(f"target_ops_coverage_{ts}.csv")
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COVERAGE_FIELDS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: str(row.get(k, "")).replace("\n", " ").replace("\r", "")
                    for k in COVERAGE_FIELDS
                }
            )

    by_status = Counter(r.get("status", "") for r in rows)
    total = len(rows)
    supported = by_status.get("supported", 0) + by_status.get("variant_covered", 0)
    pending = by_status.get("pending_registry", 0)
    print(
        f"\nCoverage: {output} ({total} ops, {supported}/{total} supported, "
        f"{pending} pending_registry)"
    )
    print(f"  {dict(by_status.most_common())}")

    for fp in _COVERAGE_DIR.glob("*.json"):
        fp.unlink()
