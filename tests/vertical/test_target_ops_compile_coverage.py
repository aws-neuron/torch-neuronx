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
    COMPILE_TIMEOUT_S,
    CompileTimeoutError,
    compile_time_limit,
    get_neuron_op_db,
    kill_compile_subprocesses,
)
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
    compile_supported_ops as _compile_supported_ops,
)
from internal.common import (
    count_input_elems as _count_input_elems,
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
from internal.common import (
    sample_inputs_have_nan as _sample_inputs_have_nan,
)
from internal.neuron_op_info import _resolve_op
from internal.roofline import compute_roofline
from known_coverage_gaps import COMPILE_GAPS

logger = logging.getLogger(__name__)


def _use_dynamic() -> bool:
    return os.environ.get("NEURON_COMPILE_COVERAGE_DYNAMIC") == "1"


NUM_EXEC_ITERATIONS = 3
METRICS_DIR = (
    Path(os.environ.get("NEURON_COVERAGE_METRICS_DIR", tempfile.gettempdir()))
    / "neuron_compile_coverage_metrics"
)

# Max chars of an exception string stored in the CSV ``error`` column
ERROR_MAXLEN = int(os.environ.get("NEURON_COVERAGE_ERROR_MAXLEN", "1000"))

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

# Ops that return multiple outputs where only a subset is mathematically unique;
# the rest are valid-but-implementation-defined intermediates (e.g. an LU
# factorization's L/U/pivots are non-unique — many valid pivotings give the same
# determinant). For these, correctness is checked only on the listed output
# indices. Keyed by aten op name (without overload), matched by substring.
_OPS_CORRECTNESS_OUTPUT_INDICES: dict[str, tuple[int, ...]] = {
    # _linalg_det returns (det, LU, pivots); only the determinant is unique.
    "_linalg_det": (0,),
    # mode returns (values, indices); the value is unique but index may not
    # match CPU unstable sort
    "aten::mode": (0,),
}

# Skip reasons for _OPS_SKIP_COMPILE_HANG.
_REASON_0D_HANG = "hangs with 0d tensor inputs"
_REASON_REGRESSION_TIMEOUT = "regression causing tests to timeout"

# Ops that hang torch.compile; skip entirely.
_OPS_SKIP_COMPILE_HANG: dict[str, str] = {
    "linspace": _REASON_0D_HANG,
    "logspace": _REASON_0D_HANG,
    "aten::_softmax_backward_data": _REASON_0D_HANG,
}

# Ops that hang or segfault under dynamic shapes; skip when --dynamic is active.
# - _foreach_* ops: each variant compiles N sub-graphs sequentially, taking 50-120 min
# - native_dropout, addr, cumprod_backward: triggers extremely long compilation (120+ min)
# - range: hangs with dynamic scalar inputs
_OPS_SKIP_COMPILE_DYNAMIC: tuple[str, ...] = (
    "_foreach_",
    "native_dropout",
    "range",
    "addr",
    "cumprod_backward",
    "_linalg_det",
    "linalg_slogdet",
    "exp.out",
    "embedding_dense_backward",
    "rrelu_with_noise_backward",
)


# Curated status overrides from manual triage of the coverage CSV (see
# failed_lowerings.md). Keyed by exact aten op name (incl. overload) → (status,
# detail). Ops listed here are NOT executed on device: the test emits the given
# status with the detail written into the ``error`` column, then skips. This is
# an xfail-style registry — a curated op will not auto-detect if Neuron later
# gains support, so prune an entry once the underlying gap is closed.
_OPS_STATUS_OVERRIDE: dict[str, tuple[str, str]] = {
    # Data-dependent output shape — cannot trace through a static graph.
    "aten::_unique2": ("graph_break", "data dependent shape"),
    "aten::bincount": ("graph_break", "data dependent shape"),
    "aten::masked_select": ("graph_break", "data dependent shape"),
    "aten::masked_select.out": ("graph_break", "data dependent shape"),
    "aten::nonzero": ("graph_break", "data dependent shape"),
    "aten::nonzero.out": ("graph_break", "data dependent shape"),
    "aten::unique_consecutive": ("graph_break", "data dependent shape"),
    "aten::linalg_lstsq.out": ("graph_break", "data dependent shape"),
    # Upstream torch bug (functionalization assert on in-place fused optimizer).
    "aten::_fused_adam_": ("graph_break", "torch functionalization bug"),
    "aten::_fused_adamw_": ("graph_break", "torch functionalization bug"),
    # Upstream torch bug. The .out meta kernel forwards out0/out1 to the
    # functional op, which rejects them (TypeError before the Neuron backend).
    "aten::grid_sampler_3d_backward.out": ("graph_break", "upstream .out meta rejects out0 kwarg"),
    # Unsupported tracing constructs.
    "aten::random_.from": ("graph_break", "unsupported"),
    "aten::sym_storage_offset": ("graph_break", "unsupported"),
    # No named tensor support
    "aten::all.dimname_out": ("graph_break", "no named tensor support"),
    "aten::any.dimname_out": ("graph_break", "no named tensor support"),
    "aten::cat.names_out": ("graph_break", "no named tensor support"),
    "aten::cumprod.dimname_out": ("graph_break", "no named tensor support"),
    "aten::cumsum.dimname_out": ("graph_break", "no named tensor support"),
    "aten::full.names_out": ("graph_break", "no named tensor support"),
    "aten::gather.dimname_out": ("graph_break", "no named tensor support"),
    "aten::index_add.dimname": ("graph_break", "no named tensor support"),
    "aten::index_copy.dimname": ("graph_break", "no named tensor support"),
    "aten::index_copy_.dimname": ("graph_break", "no named tensor support"),
    "aten::index_fill.Dimname_Scalar": ("graph_break", "no named tensor support"),
    "aten::index_fill.Dimname_Tensor": ("graph_break", "no named tensor support"),
    "aten::index_fill_.Dimname_Scalar": ("graph_break", "no named tensor support"),
    "aten::index_fill_.Dimname_Tensor": ("graph_break", "no named tensor support"),
    "aten::index_select.dimname_out": ("graph_break", "no named tensor support"),
    "aten::kthvalue.dimname_out": ("graph_break", "no named tensor support"),
    "aten::logcumsumexp.dimname_out": ("graph_break", "no named tensor support"),
    "aten::logsumexp.names_out": ("graph_break", "no named tensor support"),
    "aten::max.names_dim_max": ("graph_break", "no named tensor support"),
    "aten::mean.names_out": ("graph_break", "no named tensor support"),
    "aten::median.names_dim_values": ("graph_break", "no named tensor support"),
    "aten::min.names_dim_min": ("graph_break", "no named tensor support"),
    "aten::mode.dimname_out": ("graph_break", "no named tensor support"),
    "aten::nanmedian.names_dim_values": ("graph_break", "no named tensor support"),
    "aten::norm.names_dtype_out": ("graph_break", "no named tensor support"),
    "aten::norm.names_out": ("graph_break", "no named tensor support"),
    "aten::ones.names_out": ("graph_break", "no named tensor support"),
    "aten::prod.Dimname_out": ("graph_break", "no named tensor support"),
    "aten::rand.generator_with_names_out": ("graph_break", "no named tensor support"),
    "aten::rand.names_out": ("graph_break", "no named tensor support"),
    "aten::randn.generator_with_names_out": ("graph_break", "no named tensor support"),
    "aten::randn.names_out": ("graph_break", "no named tensor support"),
    "aten::std.correction_names_out": ("graph_break", "no named tensor support"),
    "aten::std.names_out": ("graph_break", "no named tensor support"),
    "aten::sum.DimnameList_out": ("graph_break", "no named tensor support"),
    "aten::var.correction_names_out": ("graph_break", "no named tensor support"),
    "aten::var.names_out": ("graph_break", "no named tensor support"),
    "aten::zeros.names_out": ("graph_break", "no named tensor support"),
}


# Above this many input elements, the float32 CPU oracle's own reduction loses
# enough precision (a float32 sum drops small terms once the partial exceeds
# 2**24 ~= 16.7M) that an accurate device result reads as a mismatch. Only above
# this threshold do we bother recomputing a float64 reference.
_FLOAT64_ORACLE_MIN_ELEMS = 10_000_000


def _input_elem_count(args, kwargs) -> int:
    """Largest single tensor element count across args/kwargs (incl. tensor lists)."""
    biggest = 0
    for v in (*args, *kwargs.values()):
        items = v if isinstance(v, tuple | list) else (v,)
        for t in items:
            if torch.is_tensor(t):
                biggest = max(biggest, t.numel())
    return biggest


def _filter_outputs(out, indices: tuple[int, ...] | None):
    """Select only the given output indices from a tuple/list output, or pass through."""
    if indices is not None and isinstance(out, tuple | list):
        return tuple(out[i] for i in indices)
    return out


def _correctness_output_indices(op_name: str) -> tuple[int, ...] | None:
    """Return the output indices to compare for correctness, or None for all."""
    for pat, idxs in _OPS_CORRECTNESS_OUTPUT_INDICES.items():
        if pat in op_name:
            return idxs
    return None


def _is_rng_op(op) -> bool:
    """Return True if the op reads the global RNG state.

    An RNG op gives different values on the device and on the CPU oracle, so a
    value comparison is not meaningful. Detect the op with PyTorch's own
    ``nondeterministic_seeded`` tag, which marks each seeded random op.
    """
    try:
        return torch.Tag.nondeterministic_seeded in op.tags
    except (AttributeError, TypeError):
        return False


def _matches_float64_oracle(
    op, cpu_args, cpu_kwargs, neuron_cpu, *, rtol, atol, output_indices=None
) -> bool:
    """Re-check a failed correctness comparison against a float64 CPU oracle."""

    def _to_f64(x):
        if torch.is_tensor(x) and x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return x.double()
        if isinstance(x, tuple | list):
            return type(x)(_to_f64(e) for e in x)
        return x

    def _close(a, b):
        try:
            torch.testing.assert_close(
                a, b, rtol=rtol, atol=atol, equal_nan=True, check_dtype=False
            )
            return True
        except AssertionError:
            return False

    try:
        f64_args = tuple(_to_f64(a) for a in cpu_args)
        f64_kwargs = {k: _to_f64(v) for k, v in cpu_kwargs.items()}
        with torch.no_grad():
            cpu_f32 = op(*cpu_args, **cpu_kwargs)
            cpu_f64 = op(*f64_args, **f64_kwargs)

        cpu_f32 = _filter_outputs(cpu_f32, output_indices)
        cpu_f64 = _filter_outputs(cpu_f64, output_indices)
        neuron_cpu = _filter_outputs(neuron_cpu, output_indices)

        flat32 = list(cpu_f32) if isinstance(cpu_f32, tuple | list) else [cpu_f32]
        flat64 = list(cpu_f64) if isinstance(cpu_f64, tuple | list) else [cpu_f64]
        nflat = list(neuron_cpu) if isinstance(neuron_cpu, tuple | list) else [neuron_cpu]

        # Is the float32 oracle itself unreliable here? Compare it to float64.
        # If they agree, the oracle is fine -> do not rescue (return False).
        oracle_reliable = all(
            _close(a.double() if torch.is_tensor(a) else a, b)
            for a, b in zip(flat32, flat64, strict=False)
        )
        if oracle_reliable:
            return False

        # Oracle is the inaccurate side: accept the device iff it matches float64.
        return all(
            _close(n.double() if torch.is_tensor(n) else n, b)
            for n, b in zip(nflat, flat64, strict=False)
        )
    except Exception:
        return False


if "NEURON_COVERAGE_TIMESTAMP" not in os.environ:
    os.environ["NEURON_COVERAGE_TIMESTAMP"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
_SESSION_TIMESTAMP = os.environ["NEURON_COVERAGE_TIMESTAMP"]

# Build the op database (shared cached helper; see internal.common).
neuron_op_db = get_neuron_op_db()


def _build_params():
    db = neuron_op_db
    ops_filter = os.environ.get("NEURON_OPS_FILTER", "").strip()
    if ops_filter:
        wanted = {op.strip() for op in ops_filter.split(",") if op.strip()}
        db = [info for info in db if info.op_name in wanted]
    return _build_params_for(db)


def _save_compile_result(result: dict):
    result["last_updated"] = _SESSION_TIMESTAMP
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    filepath = (
        METRICS_DIR / f"{result['op_name'].replace('::', '_').replace('.', '_')}_{worker_id}.json"
    )
    with open(filepath, "w") as f:
        json.dump(result, f)


# Compile statuses that mean an expected-supported op did not lower. A compile
# timeout is watchdog/load dependent, so it does not gate.
_COMPILE_BLOCKING = frozenset({"failed_lowering", "graph_break"})


def _gate_check_compile(op_name: str, status: str, error: str) -> None:
    """Merge-queue gate: fail on a compile-supported op that regressed.

    "Supported" is static: a registered or composite op, or an op the backend
    lowers through a decomposition. A known gap is on the allowlist, so it
    reports xfail, not a failure.
    """
    if status not in _COMPILE_BLOCKING:
        return
    expected = _classify_op(op_name) in ("registered", "composite") or (
        op_name in _compile_supported_ops()
    )
    if not expected:
        return
    if op_name in COMPILE_GAPS:
        pytest.xfail(f"known compile gap: {COMPILE_GAPS[op_name]}")
    pytest.fail(f"{op_name}: compile regression ({status}). {error}", pytrace=False)


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
        compiled_fn = torch.compile(op_fn, backend="neuron", fullgraph=True, dynamic=_use_dynamic())
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
            # Ensure a cold compile for time_to_first_ms. Clear the worker's
            # HLO cache so no previously-compiled primitive from a sibling op
            # in this worker leaks into this op's first compile. When the
            # persistent NEFF cache is active (PERSISTENT_STORAGE_BUCKET set),
            # keep the shared NEFF dir: a seeding coverage run accumulates
            # NEFFs and a gate run reads them warm. The NEFF key comes from the
            # HLO content, so wiping the HLO dir does not stop the reuse. Perf
            # never sets the bucket, so it still wipes the NEFF dir and keeps
            # time_to_first_ms cold.
            _reset_dirs = [os.environ.get("TORCH_NEURONX_HLO_CACHE_DIR", "/tmp/hlo_cache")]
            if not os.environ.get("PERSISTENT_STORAGE_BUCKET"):
                _reset_dirs.append(
                    os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")
                )
            _reset_cache_dirs(*_reset_dirs)
            start = time.perf_counter()
            with torch.no_grad(), compile_time_limit():
                _ = compiled_fn(*neuron_args, **neuron_kwargs)
                # torch.compile(backend="neuron") runs lazily: the neuronx-cc
                # compile fires at synchronize, so the time limit must cover it
                # too, not just the trace call.
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
            out["error"] = exc_str.replace("\n", " ").replace("\r", "")[:ERROR_MAXLEN]
            # Dynamo raises "Unsupported" on graph break (with fullgraph=True);
            # everything else is a real compile/runtime failure.
            err_class = "graph_break" if "Unsupported" in exc_str else "failed_lowering"

        # Single status decision.
        if err_class == "graph_break":
            out["status"] = "graph_break"
        elif err_class == "compile_timeout":
            out["status"] = "compile_timeout"
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
        #
        # Guard the loop with compile_time_limit. backend="neuron" runs the
        # neuronx-cc compile on a background thread, and synchronize() waits on
        # the device kernels, not on that compile. So the cold compile can land
        # here in the warmup call, not in the ttf block above. Without the guard
        # a runaway compile hangs the worker for the rest of the run. The
        # watchdog kills the neuronx-cc subprocess, so the wrapped call raises.
        try:
            for i in range(NUM_EXEC_ITERATIONS + 1):
                torch_neuronx.synchronize()
                start = time.perf_counter()
                with torch.no_grad(), compile_time_limit():
                    _ = compiled_fn(*neuron_args, **neuron_kwargs)
                    torch_neuronx.synchronize()
                if i > 0:  # skip warmup
                    out["exec_times"].append((time.perf_counter() - start) * 1000)
        except CompileTimeoutError as e:
            kill_compile_subprocesses()
            out["error"] = str(e)
            out["status"] = "compile_timeout"
            out["exec_times"] = []
            return out

        # Correctness check — allocate FRESH neuron args so in-place ops
        # (aten::add_, aten::sub_, aten::_foreach_add_, etc.) compare against an
        # unmutated input. Reuse compiled_fn (dynamo hits the warm trace on
        # matching shapes/dtypes).
        if any(pat in op_name for pat in _OPS_SKIP_CPU_CORRECTNESS):
            out["correctness"] = "skip_slow_cpu"
        elif _is_rng_op(op):
            out["correctness"] = "skip_accuracy_check_rng"
        elif _sample_inputs_have_nan(sample):
            out["correctness"] = "skip_nan"
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
                neuron_cpu = _allocate_to_device(neuron_out, "cpu")
                output_indices = _correctness_output_indices(op_name)
                cpu_cmp = _filter_outputs(neuron_cpu, output_indices)
                ref_cmp = _filter_outputs(cpu_out, output_indices)
                try:
                    torch.testing.assert_close(
                        cpu_cmp, ref_cmp, rtol=1e-2, atol=1e-3, equal_nan=True
                    )
                    out["correctness"] = "pass"
                except AssertionError:
                    # For large reductions the float32 CPU oracle is inaccurate
                    # Recheck against a float64 oracle
                    if _input_elem_count(
                        cpu_args, cpu_kwargs
                    ) >= _FLOAT64_ORACLE_MIN_ELEMS and _matches_float64_oracle(
                        op,
                        cpu_args,
                        cpu_kwargs,
                        neuron_cpu,
                        rtol=1e-2,
                        atol=1e-3,
                        output_indices=output_indices,
                    ):
                        out["correctness"] = "pass"
                    else:
                        raise
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
        result["ttf_slowest_ms"] = f"{max(ttfs):.2f}"

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
        elif "skip_accuracy_check_rng" in correctness:
            result["correctness_test"] = "skip_accuracy_check_rng"
        elif "skip_nan" in correctness:
            result["correctness_test"] = "skip_nan"


class TestTargetOpsCompileCoverage:
    @pytest.mark.parametrize("op_info", _build_params())
    def test_op_compile(self, op_info: NeuronOpInfo):
        op_name = op_info.op_name
        for pat, skip_reason in _OPS_SKIP_COMPILE_HANG.items():
            if pat in op_name:
                pytest.skip(f"{op_name}: skipped ({skip_reason})")
        if _use_dynamic() and any(pat in op_name for pat in _OPS_SKIP_COMPILE_DYNAMIC):
            pytest.skip(f"{op_name}: skipped (hangs/segfaults with dynamic shapes)")
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
            "ttf_slowest_ms": "",
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

        override = _OPS_STATUS_OVERRIDE.get(op_name)
        if override is not None:
            status, detail = override
            result["status"] = status
            result["error"] = detail
            _save_compile_result(result)
            pytest.skip(f"{op_name}: {status} ({detail})")

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

        # Persist a provisional compile_timeout row before the compile. A hard
        # abort (see compile_time_limit) ends the process without running Python
        # again, so this row records the timeout. The success path overwrites it.
        result["status"] = "compile_timeout"
        result["error"] = f"compile exceeded {COMPILE_TIMEOUT_S}s (hard abort)"
        _save_compile_result(result)

        per_shape = [_run_one_sample_compile(op, op_name, classification, s) for s in samples]
        _aggregate_compile_results(result, per_shape, classification)
        _save_compile_result(result)

        if _gate_enabled():
            _gate_check_compile(op_name, result["status"], result["error"])
