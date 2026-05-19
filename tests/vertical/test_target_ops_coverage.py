"""Target ops coverage + correctness test using NeuronOpInfo.

Usage:
    python -m pytest tests/vertical/test_target_ops_coverage.py -n 32 --tb=no -q
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

# Add testing module to path
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
    classify_error as _classify_error,
)
from internal.common import (
    classify_op as _classify_op,
)
from internal.common import (
    flatten_tensors as _flatten_tensors,  # re-export for tests/vertical/test_probe.py
)
from internal.common import (
    get_neuron_op_db,
)
from internal.common import (
    has_pu1 as _has_pu1,
)
from internal.common import (
    is_composite as _is_composite,
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
    / "neuron_coverage_metrics"
)

# Session timestamp — set by conftest.py pytest_sessionstart
if "NEURON_COVERAGE_TIMESTAMP" not in os.environ:
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
_SESSION_TIMESTAMP = os.environ["NEURON_COVERAGE_TIMESTAMP"]

# Build the op database (shared cached helper; see internal.common)
neuron_op_db = get_neuron_op_db()


def _save_result(result: dict):
    result["last_updated"] = _SESSION_TIMESTAMP
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    filepath = (
        METRICS_DIR / f"{result['op_name'].replace('::', '_').replace('.', '_')}_{worker_id}.json"
    )
    with open(filepath, "w") as f:
        json.dump(result, f)


def _build_params():
    return _build_params_for(neuron_op_db)


def _run_one_sample(op, op_name: str, classification: str, sample) -> dict:
    """Run a single shape end-to-end. Returns a per-shape result dict.

    Keys: status, supported_by, ttf_ms, exec_times (list[float]),
          executed, fallback, error, correctness ('', 'pass', or 'fail').
    """
    out = {
        "status": "missing",
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
    }

    sample.kwargs.pop("requires_grad", None)

    # Compute roofline from CPU-side inputs before device allocation
    # Handle List[Tensor] inputs (foreach ops) by unwrapping lists
    input_tensors = []
    for t in (sample.input, *sample.args):
        if isinstance(t, torch.Tensor):
            input_tensors.append(t)
        elif isinstance(t, list | tuple):
            input_tensors.extend(x for x in t if isinstance(x, torch.Tensor))
    dtype = input_tensors[0].dtype if input_tensors else torch.float32
    # Probe output shape/dtype via FakeTensorMode / meta (no real kernel run,
    # no memory allocation). Works for ops without a CPU kernel (most
    # Neuron-only ops have meta kernels registered by PyTorch). On failure,
    # compute_roofline falls back to input-size estimates.
    output_tensors = _probe_output_meta(op, sample)
    roofline = compute_roofline(op_name, input_tensors, output_tensors=output_tensors, dtype=dtype)
    if roofline:
        out["t_roofline_ms"] = roofline["t_roofline_ms"]
        out["t_compute_ms"] = roofline["t_compute_ms"]
        out["t_memory_ms"] = roofline["t_memory_ms"]

    try:
        neuron_args = tuple(_allocate_to_device(a, "neuron") for a in (sample.input, *sample.args))
        neuron_kwargs = {k: _allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()}
    except Exception as e:
        out["error"] = f"allocation failed: {str(e)[:100]}"
        out["status"] = "failed_lowering"
        return out

    torch_neuronx.synchronize()
    torch_neuronx.clear_op_tracking()

    try:
        # Ensure a cold compile for time_to_first_ms — clear the worker's
        # HLO+NEFF cache contents so no previously-compiled primitive from a
        # sibling op leaks into this op's first invocation.
        _reset_cache_dirs(
            os.environ.get("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache"),
            os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache"),
        )

        start = time.perf_counter()
        with torch.no_grad():
            _ = op(*neuron_args, **neuron_kwargs)
        torch_neuronx.synchronize()
        out["ttf_ms"] = (time.perf_counter() - start) * 1000
    except Exception as e:
        exc_str = str(e)
        out["error"] = exc_str.replace("\n", " ").replace("\r", "")[:200]
        out["executed"] = sorted(set(torch_neuronx.get_executed_ops()))
        out["fallback"] = sorted(set(torch_neuronx.get_fallback_ops()))
        err_class = _classify_error(exc_str)
        if err_class == "pending_registry":
            out["status"] = (
                "supported" if classification in ("registered", "composite") else "missing"
            )
        else:
            out["status"] = err_class
        return out

    out["executed"] = sorted(set(torch_neuronx.get_executed_ops()))
    out["fallback"] = sorted(set(torch_neuronx.get_fallback_ops()))

    has_cpu_roundtrip = "neuron::copy::neuron_to_cpu" in out["executed"]
    if not out["fallback"] and not has_cpu_roundtrip:
        out["status"] = "supported"
        if _has_pu1(op_name):
            out["supported_by"] = "registered"
        elif _is_composite(op_name):
            out["supported_by"] = "composite"
        else:
            out["supported_by"] = "variant_covered"
    else:
        out["status"] = "missing"

    # NUM_EXEC_ITERATIONS + 1 iterations; discard the first (warmup) so
    # exec_{avg,min,max}_ms reflect steady-state execution.
    for i in range(NUM_EXEC_ITERATIONS + 1):
        torch_neuronx.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = op(*neuron_args, **neuron_kwargs)
        torch_neuronx.synchronize()
        if i > 0:  # skip warmup
            out["exec_times"].append((time.perf_counter() - start) * 1000)

    if out["status"] == "supported":
        try:
            cpu_args = tuple(_allocate_to_device(a, "cpu") for a in (sample.input, *sample.args))
            cpu_kwargs = {k: _allocate_to_device(v, "cpu") for k, v in sample.kwargs.items()}
            fresh_neuron_args = tuple(
                _allocate_to_device(a, "neuron") for a in (sample.input, *sample.args)
            )
            fresh_neuron_kwargs = {
                k: _allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()
            }
            with torch.no_grad():
                cpu_out = op(*cpu_args, **cpu_kwargs)
                neuron_out = op(*fresh_neuron_args, **fresh_neuron_kwargs)
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
            # CPU-side limitation (not a Neuron bug) — mark skip_no_cpu so the
            # dashboard doesn't falsely flag the op as a Neuron regression.
            #   - "Could not run ... backend":   no CPU kernel at all
            #   - "not implemented for":         CPU kernel exists but rejects the dtype
            #                                    (e.g. "lshift_cpu not implemented for 'Float'")
            #   - "does not support":            CPU kernel limitation on output shape/dtype
            #                                    (e.g. "signbit does not support non-boolean
            #                                    outputs")
            is_cpu_gap = (
                ("backend" in exc_str and "Could not run" in exc_str)
                or "not implemented for" in exc_str
                or "does not support" in exc_str
            )
            out["correctness"] = "skip_no_cpu" if is_cpu_gap else "fail"
            if not out["error"]:
                out["error"] = (
                    f"correctness: {exc_str.replace(chr(10),' ').replace(chr(13),'')[:200]}"
                )

    return out


def _aggregate_into(result: dict, per_shape: list[dict], classification: str) -> None:
    """Fold per-shape results into a single CSV row.

    - status: worst-case via _STATUS_RANK
    - perf: min/avg/max across all iterations of all shapes
    - correctness: fail if any shape failed, else pass if any passed
    - neuron/fallback ops: union across shapes
    """
    worst = max(per_shape, key=lambda r: _STATUS_RANK.get(r["status"], 99))
    result["status"] = worst["status"]
    result["supported_by"] = (
        "registered"
        if _has_pu1(result["op_name"])
        else "composite"
        if _is_composite(result["op_name"])
        else classification
    )

    # Bad-input short-circuit: when every shape hit the pending_registry
    # classification in _run_one_sample, each per-shape was promoted to
    # status="supported" via dispatch-key classification but still carries the
    # bad-input error and no real execution timings. That (and only that) is
    # what we want to surface as input_source=pending_registry for the
    # dashboard. Ops that legitimately execute and then failed_lowering /
    # missing must keep their real input_source.
    if all(r["status"] == "supported" and r["error"] and not r["exec_times"] for r in per_shape):
        result["input_source"] = "pending_registry"

    result["error"] = worst["error"]
    result["neuron_primitive_ops"] = "; ".join(
        sorted({x for r in per_shape for x in r["executed"]})
    )
    result["fallback_primitive_ops"] = "; ".join(
        sorted({x for r in per_shape for x in r["fallback"]})
    )

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

    # Correctness is reported only when the aggregated op status is
    # ``supported``. If the op as a whole failed (some shape hit
    # failed_lowering / missing), a stray "pass" from a lucky shape is
    # misleading; the error column already conveys the failure. A "fail"
    # from any shape still wins over "pass" so real regressions are visible.
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


class TestTargetOpsCoverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["NEURON_LAUNCH_BLOCKING"] = "1"
        yield
        os.environ.pop("NEURON_LAUNCH_BLOCKING", None)

    @pytest.mark.parametrize("op_info", _build_params())
    def test_op(self, op_info: NeuronOpInfo):
        op_name = op_info.op_name
        classification = _classify_op(op_name)

        result = {
            "op_name": op_name,
            "op_group": op_info.op_group,
            "status": "missing",
            "supported_by": classification,
            "input_source": "",
            "input_count": "",
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
            result["status"] = "missing"
            _save_result(result)
            pytest.skip(f"{op_name}: op not found")

        try:
            samples = list(op_info.sample_inputs("cpu", torch.float32, requires_grad=False))
        except Exception as e:
            exc_str = str(e)
            result["error"] = f"sample_inputs failed: {exc_str[:200]}"
            err_class = _classify_error(exc_str)
            if err_class in ("failed_lowering", "missing"):
                # A real Neuron/dispatch failure happened while building the
                # sample — surface it as that, not as "no input available".
                result["status"] = err_class
            else:
                result["status"] = (
                    "no_input_sample"
                    if classification in ("registered", "composite")
                    else "missing"
                )
                result["input_source"] = "pending_registry"
            _save_result(result)
            pytest.skip(f"{op_name}: {result['status']}")

        # Filter out scalar/0-d inputs for topk — triggers compiler crash
        if "topk" in op_name:
            samples = [
                s
                for s in samples
                if (isinstance(s.input, torch.Tensor) and s.input.dim() > 0)
                or isinstance(s.input, list | tuple)
            ]

        if not samples:
            result["error"] = "no shapes registered"
            result["status"] = (
                "no_input_sample" if classification in ("registered", "composite") else "missing"
            )
            result["input_source"] = "pending_registry"
            _save_result(result)
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

        # Run every registered shape, aggregate across shapes.
        per_shape = [_run_one_sample(op, op_name, classification, s) for s in samples]
        _aggregate_into(result, per_shape, classification)
        _save_result(result)

        all_fallback = sorted({f for r in per_shape for f in r["fallback"]})
        if all_fallback:
            logger.warning(f"{op_name}: CPU fallback ops: {all_fallback}")
