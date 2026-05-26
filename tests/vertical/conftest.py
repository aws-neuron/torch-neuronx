"""Pytest configuration for vertical tests.

Aggregates coverage results:
  test_target_ops_coverage.py → target_ops_coverage_<ts>.csv
  test_target_ops_compile_coverage.py → target_ops_compile_coverage_<ts>.csv
"""

import os

# NEURON_LAUNCH_BLOCKING must be set BEFORE any torch_neuronx import
os.environ.setdefault("NEURON_LAUNCH_BLOCKING", "1")

import csv
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Per-worker cache dirs: avoid rmtree/write races when running with pytest-xdist
# at -n 64. pytest-xdist sets PYTEST_XDIST_WORKER=gw<N> in each worker subprocess
# before conftest is imported, so the value is available before torch_neuronx reads
# these env vars at module load. Non-xdist runs fall back to "main".
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")
_COVERAGE_CACHE_DIR = Path(tempfile.gettempdir()) / f"neuron_coverage_cache_{_WORKER_ID}"
os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] = str(_COVERAGE_CACHE_DIR / "hlo")
os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = str(_COVERAGE_CACHE_DIR / "neff")

# Pin each xdist worker to a distinct logical NeuronCore. Without this, every
# worker tries to grab the same core and all but one fail with
# "Failed to initialize Neuron Runtime: status code 1" during the first
# `.to('neuron')` call. trn2.48xlarge exposes 64 logical NeuronCores, which
# matches the typical -n 64 launch. A user-provided NEURON_RT_VISIBLE_CORES
# takes precedence (respect explicit overrides).
if _WORKER_ID.startswith("gw") and "NEURON_RT_VISIBLE_CORES" not in os.environ:
    try:
        _core_idx = int(_WORKER_ID[2:])
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(_core_idx)
    except ValueError:
        pass

_COVERAGE_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_coverage_metrics"
)

_COMPILE_COVERAGE_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_compile_coverage_metrics"
)

COVERAGE_FIELDS = [
    "op_name",
    "op_group",
    "status",
    "supported_by",
    "input_source",
    "input_count",
    "input_element_count",
    "neuron_primitive_ops",
    "fallback_primitive_ops",
    "time_to_first_ms",
    "exec_avg_ms",
    "exec_min_ms",
    "exec_max_ms",
    "t_roofline_ms",
    "t_compute_ms",
    "t_memory_ms",
    "bound",
    "roofline_ratio",
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
    parser.addoption(
        "--max-shapes-per-op",
        default=10,
        type=int,
        help=(
            "Cap shape-registry entries per op to the top-N by max tensor "
            "element count. Model-traced registries can carry 500-1000 "
            "entries per op; capping here keeps a single op's coverage run "
            "tractable. Pass 0 (or a negative value) to disable the cap."
        ),
    )


def pytest_configure(config):
    if config.getoption("--model-shapes-only", default=False):
        os.environ["NEURON_MODEL_SHAPES_ONLY"] = "1"
    registry_path = config.getoption("--shape-registry", default=None)
    if registry_path:
        os.environ["NEURON_SHAPE_REGISTRY"] = str(registry_path)
    os.environ["NEURON_MAX_SHAPES_PER_OP"] = str(
        config.getoption("--max-shapes-per-op", default=10)
    )


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return
    if _COVERAGE_DIR.exists():
        for f in _COVERAGE_DIR.glob("*.json"):
            f.unlink()
    if _COMPILE_COVERAGE_DIR.exists():
        for f in _COMPILE_COVERAGE_DIR.glob("*.json"):
            f.unlink()
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def pytest_sessionfinish(session, exitstatus):
    # Per-worker: remove this worker's cache dir to avoid /tmp bloat across runs.
    # Ignore errors — the main process (no workerinput) also runs this.
    import shutil

    shutil.rmtree(_COVERAGE_CACHE_DIR, ignore_errors=True)

    if hasattr(session.config, "workerinput"):
        return

    _aggregate_metrics(_COVERAGE_DIR, "target_ops_coverage")
    _aggregate_metrics(_COMPILE_COVERAGE_DIR, "target_ops_compile_coverage")


def _aggregate_metrics(metrics_dir: Path, prefix: str):
    """Aggregate JSON results from a metrics directory into a CSV."""
    if not metrics_dir.exists():
        return

    by_op: dict[str, dict] = {}
    for fp in sorted(metrics_dir.glob("*.json")):
        try:
            row = json.loads(fp.read_text())
            by_op[row.get("op_name", "")] = row
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")

    rows = sorted(by_op.values(), key=lambda r: r.get("op_name", ""))
    if not rows:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(f"{prefix}_{ts}.csv")
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

    for fp in metrics_dir.glob("*.json"):
        fp.unlink()
