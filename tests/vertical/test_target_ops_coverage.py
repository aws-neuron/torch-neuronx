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
    CompileTimeoutError,
    compile_time_limit,
    get_neuron_op_db,
    kill_compile_subprocesses,
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
    count_input_elems as _count_input_elems,
)
from internal.common import (
    flatten_tensors as _flatten_tensors,  # re-export for tests/vertical/test_probe.py
)
from internal.common import (
    gate_enabled as _gate_enabled,
)
from internal.common import (
    probe_output_meta as _probe_output_meta,
)
from internal.common import (
    reset_cache_dirs as _reset_cache_dirs,
)
from internal.neuron_op_info import _resolve_op
from internal.roofline import compute_roofline
from known_coverage_gaps import EAGER_GAPS

logger = logging.getLogger(__name__)


def _reliable_roofline_category(category: str, op, op_name: str) -> str:
    """Return the roofline category, downgraded to "unknown" when the number
    can't be trusted to gate on.

    Every op that matches no explicit handler falls through to the "elementwise"
    bucket in roofline.py, but that formula (1 read + 1 write, O(1) ops/elem) is
    only correct for genuinely pointwise ops. Anything else that lands there
    (weight-norm, pooling, cdist, scatter/gather, loss, resample, ...) gets a
    confidently-wrong roofline. We keep "elementwise" only when PyTorch tags the
    op ``pointwise`` (authoritative, self-maintaining) or it is a ``_foreach_``
    op (elementwise by construction); otherwise it becomes "unknown" so the
    verifier records the ratio as advisory instead of gating on it.
    """
    if category != "elementwise":
        return category or ""
    try:
        if torch.Tag.pointwise in op.tags:
            return "elementwise"
    except (AttributeError, TypeError):
        # Some op objects don't expose a usable ``tags`` attribute; treat as
        # untagged and fall through to the checks below (ends at non-gating
        # "unknown").
        pass
    if op_name.startswith("aten::_foreach_"):
        return "elementwise"
    return "unknown"


NUM_EXEC_ITERATIONS = 3
METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_coverage_metrics"
)

# Ops for which the CPU correctness reference is too expensive or fails to
# run on CPU.
#   - _weight_norm_interface_backward: the harness fills saved_norms with
#     independent randn, violating the saved_norms==‖saved_v‖ precondition; the
#     1/norm**3 term then diverges and the compare is meaningless. Correctness is
#     covered by upstream gradcheck + unit/e2e tests; the lowering is verified
#     exact vs CPU with consistent inputs. (Substring match also covers the .out
#     overload; it does NOT match the forward "aten::_weight_norm_interface".)
_OPS_SKIP_CPU_CORRECTNESS: tuple[str, ...] = (
    "convolution_backward",
    "aten::_weight_norm_interface_backward",
)

# Session timestamp — set by conftest.py pytest_sessionstart
if "NEURON_COVERAGE_TIMESTAMP" not in os.environ:
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
_SESSION_TIMESTAMP = os.environ["NEURON_COVERAGE_TIMESTAMP"]

# Build the op database (shared cached helper; see internal.common)
neuron_op_db = get_neuron_op_db()


# ── Local hacks: int64 dtype override for selected ops when running with
#                --shape-registry pointing at a model-trace JSON ─────────────
#
# The shape registry stores tensor shapes; dtype is determined per-shape
# entry by the schema-driven sample builder, which defaults to
# ``torch.float32`` unless the registry entry overrides it. For a small set
# of ops we want to verify behavior with ``int64`` inputs instead — without
# rewriting every JSON entry to add a ``dtype: long`` annotation. Listing the
# op_name here forces all floating-point tensors in the sample (including
# ``out=`` kwargs populated by ``_with_out_kwargs``) to be regenerated as
# ``torch.int64`` randints in [0, 100) right before allocation to Neuron.
#
# Gate: only applies when ``--shape-registry`` is provided (i.e.
# ``NEURON_SHAPE_REGISTRY`` env var is set by conftest); the no-shape-registry
# code path is unchanged. Keeps this file self-contained — no need to touch
# the registry data files for a one-off dtype probe.
_INT64_DTYPE_OPS = frozenset(
    {
        "aten::abs",
        "aten::add.Tensor",
        "aten::add.out",
        "aten::amax",
        "aten::amax.out",
        "aten::bitwise_and.Tensor_out",
        "aten::fill_.Scalar",
        "aten::floor_divide",
        "aten::ge.Scalar_out",
        "aten::gt.Scalar_out",
        "aten::lt.Scalar_out",
        "aten::ne.Scalar_out",
        "aten::neg",
        "aten::sub.Tensor",
        "aten::sum",
        "aten::sum.dim_IntList",
        "aten::sum.out",
        "aten::zeros",
    }
)
_INT64_HACK_ENABLED = bool(os.environ.get("NEURON_SHAPE_REGISTRY"))


def _cast_sample_to_int64(sample):
    """Return a new SampleInput with floating-point tensors replaced by
    int64 randints in [0, 100). Non-float tensors and non-tensor values
    pass through unchanged. Mirrors the shape/device/list-nesting layout
    of the original.
    """
    from torch.testing._internal.opinfo.core import SampleInput

    def cast(v):
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            return torch.randint(0, 100, v.shape, device=v.device, dtype=torch.int64)
        if isinstance(v, list):
            return [cast(x) for x in v]
        if isinstance(v, tuple):
            return tuple(cast(x) for x in v)
        if isinstance(v, dict):
            return {k: cast(x) for k, x in v.items()}
        return v

    return SampleInput(
        cast(sample.input),
        args=tuple(cast(a) for a in sample.args),
        kwargs={k: cast(v) for k, v in sample.kwargs.items()},
    )


def _save_result(result: dict):
    result["last_updated"] = _SESSION_TIMESTAMP
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    filepath = (
        METRICS_DIR / f"{result['op_name'].replace('::', '_').replace('.', '_')}_{worker_id}.json"
    )
    with open(filepath, "w") as f:
        json.dump(result, f)


# Eager statuses that mean a shipped kernel did not run on Neuron.
_EAGER_BLOCKING = frozenset({"cpu_fallback", "failed_lowering"})


def _gate_check(op_name: str, status: str, error: str) -> None:
    """Merge-queue gate: fail on a registered eager kernel that regressed.

    The eager gate covers only a registered kernel (a shipped PrivateUse1 op).
    A known gap is on the allowlist, so it reports xfail, not a failure.
    """
    if _classify_op(op_name) != "registered" or status not in _EAGER_BLOCKING:
        return
    if op_name in EAGER_GAPS:
        pytest.xfail(f"known eager gap: {EAGER_GAPS[op_name]}")
    pytest.fail(f"{op_name}: eager regression ({status}). {error}", pytrace=False)


def _build_params():
    db = neuron_op_db
    ops_filter = os.environ.get("NEURON_OPS_FILTER", "").strip()
    if ops_filter:
        wanted = {op.strip() for op in ops_filter.split(",") if op.strip()}
        db = [info for info in db if info.op_name in wanted]
    return _build_params_for(db)


def _run_one_sample(op, op_name: str, classification: str, sample) -> dict:
    """Run a single shape end-to-end. Returns a per-shape result dict.

    Keys: status, supported_by, ttf_ms, exec_times (list[float]),
          executed, fallback, error, correctness ('', 'pass', or 'fail').
    """
    out = {
        "status": "cpu_fallback",
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
        "roofline_category": "",
        "input_elems": _count_input_elems(sample),
    }

    sample.kwargs.pop("requires_grad", None)

    # Local hack: force int64 inputs for a curated set of ops when running
    # against a model-trace JSON (--shape-registry). See _INT64_DTYPE_OPS.
    if _INT64_HACK_ENABLED and op_name in _INT64_DTYPE_OPS:
        sample = _cast_sample_to_int64(sample)

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
    if any(pat in op_name for pat in _OPS_SKIP_CPU_CORRECTNESS):
        output_tensors = []
    else:
        output_tensors = _probe_output_meta(op, sample)
    roofline = compute_roofline(op_name, input_tensors, output_tensors=output_tensors, dtype=dtype)
    if roofline:
        out["t_roofline_ms"] = roofline["t_roofline_ms"]
        out["t_compute_ms"] = roofline["t_compute_ms"]
        out["t_memory_ms"] = roofline["t_memory_ms"]
        out["roofline_category"] = _reliable_roofline_category(
            roofline.get("category", ""), op, op_name
        )

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
        # Eager runs lazily: the neuronx-cc compile fires at synchronize, so the
        # time limit must cover it too, not just the op call.
        with torch.no_grad(), compile_time_limit():
            _ = op(*neuron_args, **neuron_kwargs)
            torch_neuronx.synchronize()
        out["ttf_ms"] = (time.perf_counter() - start) * 1000
        err_class = None
    except CompileTimeoutError as e:
        # A single shape ran past the compile limit. Kill the leftover
        # neuronx-cc subprocess so the worker frees its core, then move on.
        kill_compile_subprocesses()
        out["error"] = str(e)
        err_class = "compile_timeout"
    except Exception as e:
        exc_str = str(e)
        out["error"] = exc_str.replace("\n", " ").replace("\r", "")[:5000]
        err_class = _classify_error(exc_str)

    # Always capture executed/fallback — same source for both the success and
    # error paths, so a single decision block below can reason about them.
    out["executed"] = sorted(set(torch_neuronx.get_executed_ops()))
    out["fallback"] = sorted(set(torch_neuronx.get_fallback_ops()))
    has_cpu_roundtrip = "neuron::copy::neuron_to_cpu" in out["executed"]

    # Single status decision based on err_class + fallback + roundtrip.
    if err_class == "pending_registry":
        # Schema-incompatible sample; op itself may well work on Neuron.
        out["status"] = "no_input_sample"
        out["input_source"] = "pending_registry"
    elif err_class == "unsupported":
        # No kernel on Neuron → eager path falls back to CPU.
        out["status"] = "cpu_fallback"
    elif err_class == "compile_timeout":
        out["status"] = "compile_timeout"
    elif err_class == "failed_lowering":
        out["status"] = "failed_lowering"
    elif out["fallback"] or has_cpu_roundtrip:
        out["status"] = "cpu_fallback"
    else:
        out["status"] = "runs_on_neuron"

    # supported_by mirrors classification (already set on dict init). If an
    # "unknown"-classified op still executed cleanly on Neuron — default
    # overload has no kernel but a sibling variant does — upgrade the label.
    if out["status"] == "runs_on_neuron" and classification == "unknown":
        out["supported_by"] = "variant_covered"

    if err_class is not None:
        return out  # skip warmup/timing/correctness on error

    # NUM_EXEC_ITERATIONS + 1 iterations; discard the first (warmup) so
    # exec_{avg,min,max}_ms reflect steady-state execution.
    #
    # Guard the loop with compile_time_limit. The ttf block above bounds the
    # cold compile, but a warmup call here can still trigger a compile (a cache
    # miss or a decomposition variant). Without the guard a runaway compile
    # hangs the worker for the rest of the run. The watchdog kills the
    # neuronx-cc subprocess, so the wrapped call raises.
    try:
        for i in range(NUM_EXEC_ITERATIONS + 1):
            torch_neuronx.synchronize()
            start = time.perf_counter()
            with torch.no_grad(), compile_time_limit():
                _ = op(*neuron_args, **neuron_kwargs)
                torch_neuronx.synchronize()
            if i > 0:  # skip warmup
                out["exec_times"].append((time.perf_counter() - start) * 1000)
    except CompileTimeoutError as e:
        kill_compile_subprocesses()
        out["error"] = str(e)
        out["status"] = "compile_timeout"
        out["exec_times"] = []
        return out

    if out["status"] == "runs_on_neuron":
        # Some ops have CPU references that burn through hours of compute
        # on large shapes (see _OPS_SKIP_CPU_CORRECTNESS). Skip the assert
        # but keep Neuron exec timings and the overall "runs_on_neuron"
        # status intact.
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
    # supported_by mirrors op classification; per-shape entries already set
    # "variant_covered" on the rare success-by-sibling-overload case — pick
    # that up if any shape recorded it.
    if any(r.get("supported_by") == "variant_covered" for r in per_shape):
        result["supported_by"] = "variant_covered"
    else:
        result["supported_by"] = classification

    # Propagate input_source=pending_registry when any per-shape recorded it
    # (set directly in _run_one_sample for pending_registry err_class).
    if any(r.get("input_source") == "pending_registry" for r in per_shape):
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
        result["roofline_category"] = slowest_shape.get("roofline_category", "")
        exec_max = float(result.get("exec_max_ms", 0))
        roofline = slowest_shape["t_roofline_ms"]
        if roofline > 0 and exec_max > 0:
            result["roofline_ratio"] = f"{exec_max / roofline:.2f}"

    # Total input element count of the slowest-executing shape
    if slowest_shape and slowest_shape.get("input_elems"):
        result["input_element_count"] = str(slowest_shape["input_elems"])

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
    elif result["status"] == "runs_on_neuron":
        if "pass" in correctness:
            result["correctness_test"] = "pass"
        elif "skip_no_cpu" in correctness:
            result["correctness_test"] = "skip_no_cpu"


class TestTargetOpsCoverage:
    @pytest.mark.parametrize("op_info", _build_params())
    def test_op(self, op_info: NeuronOpInfo):
        op_name = op_info.op_name
        classification = _classify_op(op_name)

        result = {
            "op_name": op_name,
            "op_group": op_info.op_group,
            "status": "cpu_fallback",
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
            "roofline_category": "",
        }

        if not op_info.op_resolved:
            result["error"] = "op not found"
            result["status"] = "unresolved"
            _save_result(result)
            pytest.skip(f"{op_name}: op not found")

        try:
            samples = list(op_info.sample_inputs("cpu", torch.float32, requires_grad=False))
        except Exception as e:
            exc_str = str(e)
            result["error"] = f"sample_inputs failed: {exc_str[:200]}"
            err_class = _classify_error(exc_str)
            if err_class == "unsupported":
                # No Neuron kernel; eager path would fall back to CPU.
                result["status"] = "cpu_fallback"
            elif err_class == "failed_lowering":
                # Real Neuron failure raised during sample construction.
                result["status"] = "failed_lowering"
            else:
                # pending_registry or unknown: we never built a valid sample.
                result["status"] = "no_input_sample"
                result["input_source"] = "pending_registry"
            _save_result(result)
            pytest.skip(f"{op_name}: {result['status']}")

        # Clamp zeros in divisor args for remainder — integer division by zero
        # causes the compiled NEFF to hang on Neuron hardware.
        if "remainder" in op_name:
            for s in samples:
                for a in s.args:
                    if isinstance(a, torch.Tensor) and a.dtype in (torch.int32, torch.int64):
                        a.clamp_(min=1)

        if not samples:
            result["error"] = "no shapes registered"
            result["status"] = "no_input_sample"
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

        if _gate_enabled():
            _gate_check(op_name, result["status"], result["error"])

        all_fallback = sorted({f for r in per_shape for f in r["fallback"]})
        if all_fallback:
            logger.warning(f"{op_name}: CPU fallback ops: {all_fallback}")
