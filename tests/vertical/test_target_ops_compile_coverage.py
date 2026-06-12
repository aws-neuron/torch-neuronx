"""Target ops compile coverage test using torch.compile(backend='neuron').

Tests each target ATen op through the torch.compile path to verify it can be
traced and executed by the Neuron dynamo backend.

Usage:
    python -m pytest tests/vertical/test_target_ops_compile_coverage.py -n 32 --tb=no -q
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

import torch_neuronx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from internal import NeuronOpInfo
from internal.common import (
    STATUS_RANK as _STATUS_RANK,
)
from internal.common import (
    allocate_to_device as _allocate_to_device,
)
from internal.common import (
    build_params as _build_params_for,
)
from internal.common import (
    classify_op as _classify_op,
)
from internal.common import (
    count_input_elems as _count_input_elems,
)
from internal.common import (
    get_neuron_op_db,
)
from internal.common import (
    probe_output_meta as _probe_output_meta,
)
from internal.common import (
    reset_cache_dirs as _reset_cache_dirs,
)
from internal.neuron_op_info import _resolve_op
from internal.roofline import compute_roofline

logger = logging.getLogger(__name__)

NUM_EXEC_ITERATIONS = 3
METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_compile_coverage_metrics"
)

# Max chars of an exception string stored in the CSV ``error`` column
ERROR_MAXLEN = int(os.environ.get("NEURON_COVERAGE_ERROR_MAXLEN", "1000"))

# Ops for which the CPU correctness reference is too expensive or fails to
# run on CPU.
_OPS_SKIP_CPU_CORRECTNESS: tuple[str, ...] = ("convolution_backward",)

# Ops that hang torch.compile due to 0d tensor inputs; skip entirely.
_OPS_SKIP_COMPILE_HANG: tuple[str, ...] = (
    "linspace",
    "logspace",
)

if "NEURON_COVERAGE_TIMESTAMP" not in os.environ:
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
_SESSION_TIMESTAMP = os.environ["NEURON_COVERAGE_TIMESTAMP"]

# Build the op database (shared cached helper; see internal.common).
neuron_op_db = get_neuron_op_db()


def _build_params():
    return _build_params_for(neuron_op_db)


def _save_compile_result(result: dict):
    result["last_updated"] = _SESSION_TIMESTAMP
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    filepath = (
        METRICS_DIR / f"{result['op_name'].replace('::', '_').replace('.', '_')}_{worker_id}.json"
    )
    with open(filepath, "w") as f:
        json.dump(result, f)


def _run_one_sample_compile(op, op_name: str, classification: str, sample) -> dict:
    """Run a single sample through torch.compile(backend='neuron')."""
    out = {
        "status": "graph_break",
        "supported_by": classification,
        "ttf_ms": None,
        "exec_times": [],
        "executed": [],
        "fallback": [],
        "error": "",
        "correctness": "",
        "t_roofline_ms": None,
        "t_compute_ms": None,
        "t_memory_ms": None,
        "input_elems": _count_input_elems(sample),
    }

    sample.kwargs.pop("requires_grad", None)

    # Compute roofline from CPU-side inputs before device allocation
    input_tensors = []
    for t in (sample.input, *sample.args):
        if isinstance(t, torch.Tensor):
            input_tensors.append(t)
        elif isinstance(t, list | tuple):
            input_tensors.extend(x for x in t if isinstance(x, torch.Tensor))
    dtype = input_tensors[0].dtype if input_tensors else torch.float32
    # Probe output shape/dtype via FakeTensorMode / meta (no kernel run,
    # no memory allocation). Empty list on failure → roofline uses input-size
    # estimates.
    if any(pat in op_name for pat in _OPS_SKIP_CPU_CORRECTNESS):
        output_tensors = []
    else:
        output_tensors = _probe_output_meta(op, sample)
    roofline = compute_roofline(op_name, input_tensors, output_tensors=output_tensors, dtype=dtype)
    if roofline:
        out["t_roofline_ms"] = roofline["t_roofline_ms"]
        out["t_compute_ms"] = roofline["t_compute_ms"]
        out["t_memory_ms"] = roofline["t_memory_ms"]

    # CPU-oracle pre-flight: the shape registry is built from compile-cache keys,
    # which encode tensor shapes/dtypes but NOT scalar runtime values — so a real
    # `clamp(x, 0.0, 6.0)` is reconstructed as `clamp(x, None, None)`, a `str` dim
    # arrives as an int, etc. This logic checks if sample is valid on CPU.
    if not any(pat in op_name for pat in _OPS_SKIP_CPU_CORRECTNESS):
        try:
            cpu_args = tuple(_allocate_to_device(a, "cpu") for a in (sample.input, *sample.args))
            cpu_kwargs = {k: _allocate_to_device(v, "cpu") for k, v in sample.kwargs.items()}
            with torch.no_grad():
                op(*cpu_args, **cpu_kwargs)
        except Exception as e:
            exc_str = str(e)
            is_cpu_gap = (
                ("backend" in exc_str and "Could not run" in exc_str)
                or "not implemented for" in exc_str
                or "does not support" in exc_str
                or "with arguments from the 'CPU' backend" in exc_str
            )
            if not is_cpu_gap:
                clean = exc_str.replace(chr(10), " ")
                out["status"] = "invalid_sample"
                out["error"] = f"invalid sample (CPU oracle): {clean[:ERROR_MAXLEN]}"
                return out

    try:
        neuron_args = tuple(_allocate_to_device(a, "neuron") for a in (sample.input, *sample.args))
        neuron_kwargs = {k: _allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()}
    except Exception as e:
        out["error"] = f"allocation failed: {str(e)[:ERROR_MAXLEN]}"
        out["status"] = "failed_lowering"
        return out

    # Wrap op in a function for torch.compile
    def op_fn(*args, **kwargs):
        return op(*args, **kwargs)

    try:
        compiled_fn = torch.compile(op_fn, backend="neuron", fullgraph=True)
    except Exception as e:
        out["error"] = f"compile failed: {str(e)[:ERROR_MAXLEN]}"
        out["status"] = "failed_lowering"
        return out

    # Everything below compiles-and-runs once and reuses `compiled_fn` across
    # TTF + timing + correctness. Dynamo's guard cache keeps the trace warm
    # across subsequent calls with the same shapes/dtypes, so exec_*_ms
    # measures steady-state execution (not cold compile + execution). A single
    # torch._dynamo.reset() at function exit keeps the per-op cache entries
    # from accumulating across the ~1360 target ops in a worker.
    try:
        try:
            # Ensure a cold compile for time_to_first_ms — clear the worker's
            # HLO+NEFF cache so no previously-compiled primitive from a sibling
            # op in this worker leaks into this op's first compile.
            _reset_cache_dirs(
                os.environ.get("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache"),
                os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache"),
            )
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_fn(*neuron_args, **neuron_kwargs)
            torch_neuronx.synchronize()
            out["ttf_ms"] = (time.perf_counter() - start) * 1000
            err_class = None
        except Exception as e:
            exc_str = str(e)
            out["error"] = exc_str.replace("\n", " ").replace("\r", "")[:ERROR_MAXLEN]
            # Dynamo raises "Unsupported" on graph break (with fullgraph=True);
            # everything else is a real compile/runtime failure.
            err_class = "graph_break" if "Unsupported" in exc_str else "failed_lowering"

        # Single status decision.
        if err_class == "graph_break":
            out["status"] = "graph_break"
        elif err_class == "failed_lowering":
            out["status"] = "failed_lowering"
        else:
            out["status"] = "supported"

        # supported_by mirrors classification (set on dict init). Upgrade the
        # "unknown" case to "compiled" when torch.compile succeeded via a
        # decomposition/variant (no direct PU1 or composite kernel).
        if out["status"] == "supported" and classification == "unknown":
            out["supported_by"] = "compiled"

        if err_class is not None:
            return out  # skip warmup/timing/correctness on error

        # NUM_EXEC_ITERATIONS + 1 calls; first is an untimed warmup.
        # compiled_fn is reused so dynamo's cache stays warm (no recompile).
        for i in range(NUM_EXEC_ITERATIONS + 1):
            torch_neuronx.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_fn(*neuron_args, **neuron_kwargs)
            torch_neuronx.synchronize()
            if i > 0:  # skip warmup
                out["exec_times"].append((time.perf_counter() - start) * 1000)

        # Correctness check — allocate FRESH neuron args so in-place ops
        # (aten::add_, aten::sub_, aten::_foreach_add_, etc.) compare against an
        # unmutated input. Reuse compiled_fn (dynamo hits the warm trace on
        # matching shapes/dtypes).
        if any(pat in op_name for pat in _OPS_SKIP_CPU_CORRECTNESS):
            out["correctness"] = "skip_slow_cpu"
        else:
            try:
                cpu_args = tuple(
                    _allocate_to_device(a, "cpu") for a in (sample.input, *sample.args)
                )
                cpu_kwargs = {k: _allocate_to_device(v, "cpu") for k, v in sample.kwargs.items()}
                fresh_neuron_args = tuple(
                    _allocate_to_device(a, "neuron") for a in (sample.input, *sample.args)
                )
                fresh_neuron_kwargs = {
                    k: _allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()
                }
                with torch.no_grad():
                    cpu_out = op(*cpu_args, **cpu_kwargs)
                    neuron_out = compiled_fn(*fresh_neuron_args, **fresh_neuron_kwargs)
                torch.testing.assert_close(
                    _allocate_to_device(neuron_out, "cpu"),
                    cpu_out,
                    rtol=1e-2,
                    atol=1e-3,
                    equal_nan=True,
                )
                out["correctness"] = "pass"
            except Exception as e:
                exc_str = str(e)
                # CPU-side limitation (not a Neuron bug) — mark skip_no_cpu.
                is_cpu_gap = (
                    ("backend" in exc_str and "Could not run" in exc_str)
                    or "not implemented for" in exc_str
                    or "does not support" in exc_str
                )
                out["correctness"] = "skip_no_cpu" if is_cpu_gap else "fail"
                if not out["error"]:
                    clean = exc_str.replace(chr(10), " ").replace(chr(13), "")
                    out["error"] = f"correctness: {clean[:ERROR_MAXLEN]}"
    finally:
        torch._dynamo.reset()

    return out


def _aggregate_compile_results(result: dict, per_shape: list[dict], classification: str) -> None:
    """Fold per-shape compile results into a single CSV row."""
    # invalid_sample shapes are untestable. The op
    # is invalid_sample ONLY if EVERY shape is invalid.
    valid = [r for r in per_shape if r["status"] != "invalid_sample"]
    ranked = valid if valid else per_shape
    worst = max(ranked, key=lambda r: _STATUS_RANK.get(r["status"], 99))
    result["status"] = worst["status"]
    # supported_by mirrors op classification; per-shape entries may have
    # upgraded "unknown" to "compiled" when compilation succeeded via a
    # decomposition — pick that up if any shape recorded it.
    if any(r.get("supported_by") == "compiled" for r in per_shape):
        result["supported_by"] = "compiled"
    else:
        result["supported_by"] = classification
    result["error"] = worst["error"]

    ttfs = [r["ttf_ms"] for r in per_shape if r["ttf_ms"] is not None]
    if ttfs:
        result["time_to_first_ms"] = f"{min(ttfs):.2f}"

    all_times = [t for r in per_shape for t in r["exec_times"]]
    if all_times:
        result["exec_avg_ms"] = f"{sum(all_times) / len(all_times):.3f}"
        result["exec_min_ms"] = f"{min(all_times):.3f}"
        result["exec_max_ms"] = f"{max(all_times):.3f}"

    # Roofline: pick the roofline for the shape that produced the slowest execution
    slowest_shape = max(
        (r for r in per_shape if r["exec_times"]),
        key=lambda r: max(r["exec_times"]),
        default=None,
    )
    if slowest_shape and slowest_shape.get("t_roofline_ms"):
        result["t_roofline_ms"] = f"{slowest_shape['t_roofline_ms']:.4f}"
        result["t_compute_ms"] = f"{slowest_shape['t_compute_ms']:.4f}"
        result["t_memory_ms"] = f"{slowest_shape['t_memory_ms']:.4f}"
        result["bound"] = (
            "compute" if slowest_shape["t_compute_ms"] > slowest_shape["t_memory_ms"] else "memory"
        )
        exec_max = float(result.get("exec_max_ms", 0))
        roofline = slowest_shape["t_roofline_ms"]
        if roofline > 0 and exec_max > 0:
            result["roofline_ratio"] = f"{exec_max / roofline:.2f}"

    # Total input element count of the slowest-executing shape
    if slowest_shape and slowest_shape.get("input_elems"):
        result["input_element_count"] = str(slowest_shape["input_elems"])

    # Correctness is reported only when aggregated status is ``supported``;
    # otherwise a lucky-shape pass would contradict the overall failure.
    correctness = {r["correctness"] for r in per_shape} - {""}
    if "fail" in correctness:
        result["correctness_test"] = "fail"
        if not result["error"]:
            for r in per_shape:
                if r["correctness"] == "fail" and r["error"]:
                    result["error"] = r["error"]
                    break
    elif result["status"] == "supported":
        if "pass" in correctness:
            result["correctness_test"] = "pass"
        elif "skip_no_cpu" in correctness:
            result["correctness_test"] = "skip_no_cpu"


class TestTargetOpsCompileCoverage:
    @pytest.mark.parametrize("op_info", _build_params())
    def test_op_compile(self, op_info: NeuronOpInfo):
        op_name = op_info.op_name
        if any(pat in op_name for pat in _OPS_SKIP_COMPILE_HANG):
            pytest.skip(f"{op_name}: skipped (hangs with 0d tensor inputs)")
        classification = _classify_op(op_name)

        result = {
            "op_name": op_name,
            "op_group": op_info.op_group,
            "status": "unresolved",
            "supported_by": classification,
            "input_source": "",
            "input_count": "",
            "input_element_count": "",
            "neuron_primitive_ops": "",
            "fallback_primitive_ops": "",
            "time_to_first_ms": "",
            "exec_avg_ms": "",
            "exec_min_ms": "",
            "exec_max_ms": "",
            "error": "",
            "last_updated": "",
            "correctness_test": "",
            "t_roofline_ms": "",
            "t_compute_ms": "",
            "t_memory_ms": "",
            "bound": "",
            "roofline_ratio": "",
        }

        if not op_info.op_resolved:
            result["error"] = "op not found"
            result["status"] = "unresolved"
            _save_compile_result(result)
            pytest.skip(f"{op_name}: op not found")

        try:
            samples = list(op_info.sample_inputs("cpu", torch.float32, requires_grad=False))
        except Exception as e:
            exc_str = str(e)
            result["error"] = f"sample_inputs failed: {exc_str[:ERROR_MAXLEN]}"
            from internal.common import classify_error as _cls_err

            err_class = _cls_err(exc_str)
            if err_class == "failed_lowering":
                # Real Neuron failure raised during sample construction.
                result["status"] = "failed_lowering"
            else:
                # unsupported / pending_registry / unknown — test couldn't
                # produce a valid sample to compile and run.
                result["status"] = "no_input_sample"
                result["input_source"] = "pending_registry"
            _save_compile_result(result)
            pytest.skip(f"{op_name}: {result['status']}")

        if not samples:
            result["error"] = "no shapes registered"
            result["status"] = "no_input_sample"
            result["input_source"] = "pending_registry"
            _save_compile_result(result)
            pytest.skip(f"{op_name}: pending_registry")

        entries = op_info._registry_entries()
        if entries:
            has_model = any(s != "builtin" for e in entries for s in e.sources)
            result["input_source"] = (
                "shape_registry_model" if has_model else "shape_registry_default"
            )
        else:
            result["input_source"] = "pytorch_db"
        result["input_count"] = str(len(samples))
        op = _resolve_op(op_name)

        per_shape = [_run_one_sample_compile(op, op_name, classification, s) for s in samples]
        _aggregate_compile_results(result, per_shape, classification)
        _save_compile_result(result)
