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
import sys
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

# Disable torch.compile fallback for accurate lowering results
os.environ["TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION"] = "1"

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

# test_dynamic_aten_ops.py drops per-op JSON row-lists here.
_DYNAMIC_METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_dynamic_metrics"
)

_COLLECTIVES_METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_collectives_metrics"
)

# test_dispatch_latency.py drops one per-op JSON row here.
_DISPATCH_METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_dispatch_metrics"
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
    "ttf_slowest_ms",
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
    # Appended (not inserted) so existing column positions are unchanged for any
    # positional CSV consumer. Set by _reliable_roofline_category in the eager
    # coverage test; blank in the compile CSV (unused there). Drives the verifier's
    # roofline gate: only "trustworthy" categories are gated.
    "roofline_category",
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
        help="JSON shape registry file to load model-traced shapes from",
    )
    parser.addoption(
        "--no-cache-key-shapes",
        action="store_true",
        default=False,
        help=(
            "Skip the committed cache-key model shapes (register_cache_key_shapes in "
            "tests/vertical/internal/neuron_shape_defaults_generated.py). The shapes "
            "load by default. Regenerate them with tools/refresh_cache_key_shapes.py."
        ),
    )
    parser.addoption(
        "--traced-model",
        default=None,
        help=(
            "Comma-separated accuracy smoke test subset(s) to enable, e.g. "
            "qwen3_235b_ep64_tp4. An unknown name is an error rather than an empty selection."
        ),
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
    parser.addoption(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic shapes (pass dynamic=True to torch.compile)",
    )
    parser.addoption(
        "--stablehlo-executor",
        action="store_true",
        default=False,
        help="Use the StableHLO executor backend for torch.compile",
    )
    parser.addoption(
        "--hilo-infergoldens",
        action="store_true",
        default=False,
        help="Use the hilo frontend + XLAInferGoldens executor (requires neuronx-cc)",
    )
    parser.addoption(
        "--preserve-hilo-workdir",
        action="store_true",
        default=False,
        help="Preserve the neuronx-cc working directory for hilo/XLAInferGoldens runs",
    )
    parser.addoption(
        "--ops",
        default=None,
        help=(
            "Comma-separated list of EXACT op names to run, e.g. "
            "'aten::abs,aten::add.Tensor'. When unset, runs all target ops. "
            "Useful for re-running a subset (e.g. the int64 hack list) "
            "without -k expression escaping issues."
        ),
    )


def pytest_configure(config):
    if config.getoption("--model-shapes-only", default=False):
        os.environ["NEURON_MODEL_SHAPES_ONLY"] = "1"
    registry_path = config.getoption("--shape-registry", default=None)
    if registry_path:
        os.environ["NEURON_SHAPE_REGISTRY"] = str(registry_path)
    no_cache_key_shapes = config.getoption("--no-cache-key-shapes", default=False)
    if no_cache_key_shapes:
        os.environ["NEURON_CACHE_KEY_SHAPES"] = "0"
    traced_model = config.getoption("--traced-model", default=None)
    if traced_model:
        os.environ["NEURON_TRACED_MODEL"] = str(traced_model)
    os.environ["NEURON_MAX_SHAPES_PER_OP"] = str(
        config.getoption("--max-shapes-per-op", default=10)
    )
    if config.getoption("--hilo-infergoldens", default=False):
        os.environ["TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS"] = "1"
        os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
    elif config.getoption("--stablehlo-executor", default=False):
        os.environ["TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR"] = "1"
        os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)
    else:
        os.environ.pop("TORCH_NEURONX_DYNAMO_STABLEHLO_EXECUTOR", None)
        os.environ.pop("TORCH_NEURONX_DYNAMO_HILO_INFERGOLDENS", None)
    if config.getoption("--preserve-hilo-workdir", default=False):
        os.environ["TORCH_NEURONX_PRESERVE_HILO_WORKDIR"] = "1"
    if config.getoption("--dynamic", default=False):
        os.environ["NEURON_COMPILE_COVERAGE_DYNAMIC"] = "1"

    ops_filter = config.getoption("--ops", default=None)
    if ops_filter:
        os.environ["NEURON_OPS_FILTER"] = ops_filter


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return
    if _COVERAGE_DIR.exists():
        for f in _COVERAGE_DIR.glob("*.json"):
            f.unlink()
    if _COMPILE_COVERAGE_DIR.exists():
        for f in _COMPILE_COVERAGE_DIR.glob("*.json"):
            f.unlink()
    if _DYNAMIC_METRICS_DIR.exists():
        for f in _DYNAMIC_METRICS_DIR.glob("*.json"):
            f.unlink()
    if _COLLECTIVES_METRICS_DIR.exists():
        for f in _COLLECTIVES_METRICS_DIR.glob("*.json"):
            f.unlink()
    if _DISPATCH_METRICS_DIR.exists():
        for f in _DISPATCH_METRICS_DIR.glob("*.json"):
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
    _aggregate_dispatch_metrics(_DISPATCH_METRICS_DIR, "dispatch_latency")
    _aggregate_dynamic_metrics(_DYNAMIC_METRICS_DIR, "dynamic_aten_ops")
    _aggregate_dynamic_metrics(_COLLECTIVES_METRICS_DIR, "dynamic_collectives")


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


def _aggregate_dispatch_metrics(metrics_dir: Path, prefix: str):
    """Aggregate per-op dispatch-latency JSON rows into one CSV.

    Uses the benchmark's own writer so the schema stays in one place
    (``internal/dispatch_timing.DISPATCH_FIELDS``).
    """
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

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from internal.dispatch_timing import write_csv

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_csv(rows, str(Path(f"{prefix}_{ts}.csv")))

    for fp in metrics_dir.glob("*.json"):
        fp.unlink()


def _aggregate_dynamic_metrics(metrics_dir: Path, prefix: str):
    """Merge per-cell JSON rows (one file per unit per xdist worker) into one
    timestamped ``<prefix>_<ts>.csv``."""

    if not metrics_dir.exists():
        return

    rows: list[dict] = []
    for fp in sorted(metrics_dir.glob("*.json")):
        try:
            payload = json.loads(fp.read_text())
        except Exception as e:
            logger.warning(f"Failed to read {fp}: {e}")
            continue
        rows.extend(payload if isinstance(payload, list) else [payload])

    if not rows:
        return

    fields = list(rows[0].keys())
    lead = fields[:4]
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in lead))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(f"{prefix}_{ts}.csv")
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(
                {k: str(row.get(k, "")).replace("\n", " ").replace("\r", "") for k in fields}
            )
    logger.info(f"wrote {len(rows)} rows -> {output}")

    for fp in metrics_dir.glob("*.json"):
        fp.unlink()
