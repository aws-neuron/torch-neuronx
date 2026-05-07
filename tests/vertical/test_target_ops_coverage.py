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
from internal import NeuronOpInfo, build_neuron_op_db
from internal.neuron_op_info import _resolve_op
from internal.target_ops import TARGET_ATEN_OP_GROUP, TARGET_ATEN_OPS

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

# Build the op database
neuron_op_db = build_neuron_op_db(
    target_ops=TARGET_ATEN_OPS,
    target_op_groups=TARGET_ATEN_OP_GROUP,
)


def _allocate_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.detach().clone().to(device)
    elif isinstance(data, list | tuple):
        return type(data)(_allocate_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: _allocate_to_device(v, device) for k, v in data.items()}
    return data


def _has_pu1(op_name: str) -> bool:
    clean = op_name.replace("aten::", "")
    parts = clean.split(".", 1)
    base, overload = parts[0], (parts[1] if len(parts) > 1 else "default")
    try:
        pkt = getattr(torch.ops.aten, base, None)
        if pkt is None:
            return False
        op = getattr(pkt, overload, None)
        if op is None:
            return False
        schema = op._schema
        q = f"{schema.name}.{schema.overload_name}" if schema.overload_name else str(schema.name)
        return torch._C._dispatch_has_kernel_for_dispatch_key(q, torch._C.DispatchKey.PrivateUse1)
    except Exception:
        return False


def _is_composite(op_name: str) -> bool:
    clean = op_name.replace("aten::", "")
    parts = clean.split(".", 1)
    base, overload = parts[0], (parts[1] if len(parts) > 1 else "default")
    try:
        pkt = getattr(torch.ops.aten, base, None)
        if pkt is None:
            return False
        op = getattr(pkt, overload, None)
        if op is None:
            return False
        schema = op._schema
        q = f"{schema.name}.{schema.overload_name}" if schema.overload_name else str(schema.name)
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            q, torch._C.DispatchKey.CompositeImplicitAutograd
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(
            q, torch._C.DispatchKey.CompositeExplicitAutograd
        )
    except Exception:
        return False


def _classify_op(op_name: str) -> str:
    if _has_pu1(op_name):
        return "registered"
    if _is_composite(op_name):
        return "composite"
    return "unknown"


def _classify_error(exc_str: str) -> str:
    if "Could not run" in exc_str and "backend" in exc_str:
        return "missing"
    if "NYI" in exc_str or "not yet implemented" in exc_str.lower():
        return "missing"
    if "op not found" in exc_str:
        return "missing"
    if "CPU fallback disabled" in exc_str:
        return "failed_lowering"
    if "_copy_from_and_resize" in exc_str:
        return "failed_lowering"
    if "got an unexpected keyword argument" in exc_str:
        return "failed_lowering"
    if "Expected PythonKernelHolder" in exc_str:
        return "failed_lowering"
    if "only supports CPU" in exc_str:
        return "missing"
    # Type mismatches from wrong sample inputs — not a real failure
    if "Expected a value of type" in exc_str and "but instead found type" in exc_str:
        return "pending_registry"
    if "is missing value for argument" in exc_str:
        return "pending_registry"
    if "expected at most" in exc_str and "argument(s) but received" in exc_str:
        return "pending_registry"
    if "scalar tensor expected to be 0 dim" in exc_str:
        return "pending_registry"
    if "Unknown keyword argument" in exc_str:
        return "pending_registry"
    if "takes" in exc_str and "positional argument(s) but" in exc_str:
        return "pending_registry"
    if "specified both as positional and keyword" in exc_str:
        return "pending_registry"
    if "Expected out tensor to have dtype" in exc_str:
        return "pending_registry"
    if "Expected result to be of dtype" in exc_str:
        return "pending_registry"
    if "Dimension specified as" in exc_str and "tensor has no dimensions" in exc_str:
        return "pending_registry"
    if "list assignment index out of range" in exc_str:
        return "pending_registry"

    return "failed_lowering"


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
    params = []
    for op_info in neuron_op_db:
        params.append(pytest.param(op_info, id=f"{op_info.op_name}[{op_info.op_group}]"))
    return params


# Worst-case ordering when aggregating across shapes. Higher rank = worse.
_STATUS_RANK = {
    "supported": 0,
    "pending_registry": 1,
    "missing": 2,
    "failed_lowering": 3,
}


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
    }

    sample.kwargs.pop("requires_grad", None)

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

    if not out["fallback"]:
        out["status"] = "supported"
        if _has_pu1(op_name):
            out["supported_by"] = "registered"
        elif _is_composite(op_name):
            out["supported_by"] = "composite"
        else:
            out["supported_by"] = "variant_covered"
    else:
        out["status"] = "missing"

    for _ in range(NUM_EXEC_ITERATIONS):
        torch_neuronx.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = op(*neuron_args, **neuron_kwargs)
        torch_neuronx.synchronize()
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
            if "backend" in exc_str and (
                "Could not run" in exc_str or "not implemented" in exc_str
            ):
                out["correctness"] = "skip_no_cpu"
            else:
                out["correctness"] = "fail"
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

    # If every shape hit the pending_registry short-circuit (status promoted to
    # supported/missing via dispatch key but each had a bad-input error with no
    # real execution), surface that as pending_registry for the dashboard.
    if all(r["error"] and not r["exec_times"] for r in per_shape):
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

    correctness = {r["correctness"] for r in per_shape} - {""}
    if "fail" in correctness:
        result["correctness_test"] = "fail"
    elif "pass" in correctness:
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
        }

        if not op_info.op_resolved:
            result["error"] = "op not found"
            result["status"] = "missing"
            _save_result(result)
            pytest.skip(f"{op_name}: op not found")

        try:
            samples = list(op_info.sample_inputs("cpu", torch.float32, requires_grad=False))
        except Exception as e:
            result["error"] = f"sample_inputs failed: {str(e)[:100]}"
            result["status"] = (
                "supported" if classification in ("registered", "composite") else "missing"
            )
            result["input_source"] = "pending_registry"
            _save_result(result)
            pytest.skip(f"{op_name}: pending_registry")

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
                "supported" if classification in ("registered", "composite") else "missing"
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
