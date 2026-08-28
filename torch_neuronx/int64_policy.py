"""Int64 execution-mode policy for Neuron.

A single env var, ``TORCH_NEURONX_INT64_MODE``, selects how int64 tensors are
handled during Neuron dispatch:

  - ``native`` (default): per-op cascade. A natively-supported op runs as S64
    on the compiler (GPSIMD); a non-native op is demoted on device via the
    packed view (low 32 bits); matmul/conv (and other unsupported ops) fall
    back to CPU. The "best available path per op".
  - ``packed``: force every int64 op through the packed view (low-word
    on-device extraction). Non-packable ops (matmul/conv) fall back to CPU.
  - ``cpu_bounce``: cast int64→int32 via a CPU roundtrip for every op (op still
    runs on device in int32). Safe but slow.
  - ``cpu_fallback``: run every int64 op *entirely on CPU* (no Neuron
    execution). The whole op executes on the host and the result is copied
    back. Differs from ``cpu_bounce``, which only does the int64→int32 cast on
    CPU and still runs the op on device.
  - ``native_strict``: S64 only for natively-supported ops; everything else
    falls back to CPU. No precision loss.

Two layers:
  * ``Int64Mode`` / ``get_int64_mode()`` / ``is_*_int64_mode()`` — the storage
    and execution mode (used by the packed-view paths and cast policy).
  * ``Int64Policy`` — the per-op native dispatch decision used by ``native`` /
    ``native_strict`` modes (which ops pass S64, which downcast, which CPU
    fall back) plus the compiler flags that gate them.
"""

from __future__ import annotations

import functools
import logging
import os
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)
_warned_ops: set[str] = set()


class Int64Mode(Enum):
    """How int64 tensors are handled when dispatched to Neuron.

    String values match the ``TORCH_NEURONX_INT64_MODE`` env var. The C++
    ``Int64Mode`` enum in ``DispatchUtils.h`` mirrors these with stable integer
    codes (cpu_bounce=0, native=1, packed=2, native_strict=3, cpu_fallback=4).
    """

    CPU_BOUNCE = "cpu_bounce"
    NATIVE = "native"
    PACKED = "packed"
    NATIVE_STRICT = "native_strict"
    CPU_FALLBACK = "cpu_fallback"


class Int64Decision(str, Enum):
    """Per-op routing decision (mirrors the C++ Int64Decision enum)."""

    NATIVE = "native"
    DOWNCAST = "downcast"
    UNSUPPORTED = "unsupported"


@lru_cache(maxsize=1)
def get_int64_mode() -> Int64Mode:
    """Read the int64 execution mode from environment. Cached at first call.

    Resolution order:
      1. TORCH_NEURONX_INT64_MODE
         (native | native_strict | packed | cpu_bounce | cpu_fallback)
      2. Default: native (per-op cascade)
    """
    mode_str = os.environ.get("TORCH_NEURONX_INT64_MODE")
    if mode_str is not None:
        mode_str = mode_str.strip().lower()
        try:
            return Int64Mode(mode_str)
        except ValueError:
            valid = ", ".join(m.value for m in Int64Mode)
            raise ValueError(
                f"Invalid TORCH_NEURONX_INT64_MODE={mode_str!r}. Valid: {valid}"
            ) from None

    return Int64Mode.NATIVE


def is_packed_int64_mode() -> bool:
    """Return True if int64 should use packed physical storage."""
    return get_int64_mode() == Int64Mode.PACKED


def is_native_int64_mode() -> bool:
    """Return True if int64 passes through to the compiler as S64 (native)."""
    return get_int64_mode() == Int64Mode.NATIVE


def is_native_strict_int64_mode() -> bool:
    """Return True if mode is native_strict (S64 for native ops, else CPU)."""
    return get_int64_mode() == Int64Mode.NATIVE_STRICT


def is_cpu_bounce_int64_mode() -> bool:
    """Return True if int64 uses CPU roundtrip cast (op still runs on device)."""
    return get_int64_mode() == Int64Mode.CPU_BOUNCE


def is_cpu_fallback_int64_mode() -> bool:
    """Return True if every int64 op runs entirely on CPU (no Neuron execution)."""
    return get_int64_mode() == Int64Mode.CPU_FALLBACK


def is_s64_passthrough() -> bool:
    """Return True when int64 flows to the compiler as S64 (native or native_strict)."""
    return get_int64_mode() in (Int64Mode.NATIVE, Int64Mode.NATIVE_STRICT)


def native_int64_guard_allows(inputs) -> bool:
    """Central guard for *when* native int64 (S64) is safe to use for an op.

    Native S64 execution deadlocks the compiler for certain input structures, so
    a native-capable op runs as S64 only when its int64 operands match the cases
    the compiler is known to handle; every other case falls back to the packed
    int32[..., 2] path (correct, just low-word for magnitudes > 2**31).

    This is the single place to widen native int64 coverage — as compiler hangs
    are fixed, relax a criterion here (a future PR need only touch this guard).
    Must stay in lockstep with the C++ ``NativeInt64GuardAllows`` in
    Int64Policy.h so the traced graph (fake) and the execution binding agree.

    Current safe criteria (conservative first version):
      1. every int64 tensor operand is contiguous, AND
      2. at most one int64 operand needs broadcasting to the common shape — a
         native S64 binary op with BOTH operands broadcast (e.g. kron's
         outer-product ``mul``) hangs the device.
    """
    import torch

    int64_tensors: list = []

    def _collect(x):
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.int64 and x.dim() > 0:
                int64_tensors.append(x)
        elif isinstance(x, list | tuple):
            for e in x:
                _collect(e)

    for a in inputs:
        _collect(a)

    if not int64_tensors:
        return True
    if any(not t.is_contiguous() for t in int64_tensors):
        return False
    try:
        common = tuple(torch.broadcast_shapes(*[tuple(t.shape) for t in int64_tensors]))
    except Exception:
        return True
    # Dual-broadcast (>=2 int64 operands differ from the common shape) hangs
    # native S64 arithmetic; keep at most one broadcast operand on the native path.
    return sum(1 for t in int64_tensors if tuple(t.shape) != common) < 2


def use_packed_int64_view_for_op(op_name: str, inputs=None) -> bool:
    """Whether ``op_name``'s int64 tensors execute on the packed int32[..., 2]
    physical view rather than as S64.

    True for explicit packed mode (every int64 op) and for the native cascade's
    non-native (DOWNCAST) ops. A NATIVE op returns False (keeps S64) only when
    the input guard allows it (see ``native_int64_guard_allows``); otherwise it
    falls back to the packed view for this invocation. Python mirror of the C++
    ``UsePackedInt64ViewForOp`` — the two must agree so the traced graph's int64
    parameters (faked as int32[..., 2] here) match the physical view the C++
    dispatcher binds at execution.
    """
    if is_packed_int64_mode():
        return True
    if not is_s64_passthrough():
        return False
    if Int64Policy.get().decide(op_name) == Int64Decision.DOWNCAST:
        return True
    # NATIVE-capable op: fall back to the packed view when the input guard
    # rejects this invocation (non-contiguous / dual-broadcast); else keep S64.
    return inputs is not None and not native_int64_guard_allows(inputs)


def reset_int64_mode_cache() -> None:
    """Clear the cached mode. For testing only."""
    get_int64_mode.cache_clear()
    Int64Policy.reset_cache()


# Ops with confirmed compiler int64 (S64) support.
_NATIVE_INT64_OPS: frozenset[str] = frozenset(
    {
        # TensorTensor / TensorScalar arithmetic
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::remainder",
        "aten::neg",
        "aten::abs",
        "aten::sign",
        "aten::maximum",
        "aten::minimum",
        "aten::floor_divide",
        # Comparison — DISABLED: native S64 comparison hangs the device on
        # non-contiguous int64. The hang is driven by the *output* transpose
        # epilogue (inputs are contiguous at the binding), so the input-based
        # native_int64_guard_allows can't gate it — keep these packed until the
        # strided S64 compare+transpose codegen hang is fixed.
        # "aten::eq",
        # "aten::ne",
        # "aten::gt",
        # "aten::ge",
        # "aten::lt",
        # "aten::le",
        # Bitwise
        "aten::bitwise_and",
        "aten::bitwise_or",
        "aten::bitwise_xor",
        "aten::bitwise_not",
        "aten::bitwise_left_shift",
        "aten::bitwise_right_shift",
        # Reduction
        # NOTE: aten::sum is intentionally NOT native. Benchmarks show native
        # int64 reduce_add is catastrophically slow (~130x at 1024^2, ~268x at
        # 4096^2 vs the packed/int32 path), so it's left as DOWNCAST → packed
        # view. Trade-off: packed sums the low 32 bits, so an int64 accumulation
        # that overflows int32 loses precision (use native_strict/cpu_bounce for
        # exact large-magnitude reductions). prod/amax/amin do not show this
        # pathology and stay native.
        "aten::prod",
        "aten::amax",
        "aten::amin",
        # Memset/creation
        "aten::fill",
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::empty",
        # Data movement
        "aten::slice",
        "aten::cat",
        "aten::stack",
        "aten::reshape",
        "aten::expand",
        "aten::contiguous",
        "aten::select",
        "aten::unsqueeze",
        "aten::squeeze",
        "aten::clone",
        "aten::copy_",
        # Transpose
        "aten::permute",
        "aten::transpose",
        # Type conversion
        "aten::to",
        "aten::_to_copy",
        # Select
        "aten::where",
        # Indexing — DISABLED: native S64 gather makes the compiler emit an
        # internal int64 Cast that fails ISA validation (dtype_int64_illegal_check,
        # NCC_IINAR001), a compile-time error. Demoted to the packed int32 path
        # (verified: packed handles it). Re-enable once the S64 gather codegen
        # is fixed.
        # "aten::gather",
        # Embedding
        "aten::embedding",
    }
)

# Ops that accept int64 but would produce garbage on Neuron regardless of
# whether the compiler downcasts (overflow, float-only internals).
_UNSUPPORTED_INT64_OPS: frozenset[str] = frozenset(
    {
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::matmul",
        "aten::linear",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv3d",
    }
)


def _strip_overload(op_name: str) -> str:
    if "::" in op_name:
        ns_op = op_name.split("::", 1)[1]
        base = ns_op.split(".", 1)[0]
        return f"aten::{base}"
    return op_name


# NKI's int64 capability flag.
#
# NKI's kernel language does not yet support int64 (tracked as NKI-151).
# Kernels that take int64 inputs silently lower the values to int32 in
# NKI's own compiler pipeline, producing int32 output with int32-precision
# values. From the int64 policy's perspective this is equivalent to a
# DOWNCAST op — some form of demotion happens, just inside NKI's
# compiler instead of neuronx-cc's LegalizeInt64 pass.
#
# When NKI-151 lands and NKI's language gains int64 support, flip this
# constant to True. Every nki_op-registered op then automatically
# transitions from DOWNCAST behavior to NATIVE behavior with no other
# code changes:
#   - native mode: pass int64 through (NKI handles natively); no warning.
#   - native_strict mode: pass int64 through (op is now NATIVE).
#   - packed / cpu_bounce mode: framework still casts int64↔int32 at the
#     kernel boundary (matches their on-device downcast contract).
#   - cpu_fallback mode: still falls back (mode-driven, not op-driven).
#
# This constant is intentionally a code-level toggle rather than an env
# var: NKI int64 capability is a property of the installed NKI version,
# not a per-run user choice.
_NKI_INT64_NATIVE: bool = False


class Int64Policy:
    """Per-op native int64 dispatch decision for the native / native_strict modes.

    For packed / cpu_bounce modes this policy never forces a whole-op CPU
    fallback — those modes keep their existing on-device handling — so
    ``should_cpu_fallback`` returns False there.
    """

    def __init__(self, mode: Int64Mode):
        self._mode = mode

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get() -> Int64Policy:
        return Int64Policy(get_int64_mode())

    @staticmethod
    def reset_cache() -> None:
        Int64Policy.get.cache_clear()

    @property
    def mode(self) -> Int64Mode:
        return self._mode

    @property
    def passes_s64_to_compiler(self) -> bool:
        """True when int64 tensors flow through to the compiler as S64."""
        return self._mode in (Int64Mode.NATIVE, Int64Mode.NATIVE_STRICT)

    # Back-compat alias used by kernel.py / op_impl.py / etc.
    @property
    def native_int64_enabled(self) -> bool:
        return self.passes_s64_to_compiler

    def decide(self, op_name: str) -> Int64Decision:
        """Per-op decision."""
        if self.should_cpu_fallback(op_name):
            return Int64Decision.UNSUPPORTED
        base = _strip_overload(op_name)
        if self.passes_s64_to_compiler and base in _NATIVE_INT64_OPS:
            return Int64Decision.NATIVE
        return Int64Decision.DOWNCAST

    def should_cpu_fallback(self, op_name: str) -> bool:
        """True if this op should run on CPU instead of Neuron.

        - ``cpu_fallback``: every int64 op runs entirely on CPU.
        - ``native`` / ``native_strict``: only ops steered away from Neuron
          (UNSUPPORTED matmul/conv always; in native_strict any non-native op).
        - ``packed`` / ``cpu_bounce``: never force a whole-op CPU fallback —
          they keep their on-device handling (packed view / int32 cast).

        Callers still gate on the op actually having an int64 input.
        """
        if self._mode == Int64Mode.CPU_FALLBACK:
            return True
        if self._mode not in (Int64Mode.NATIVE, Int64Mode.NATIVE_STRICT):
            return False
        base = _strip_overload(op_name)
        if base in _UNSUPPORTED_INT64_OPS:
            return True
        return self._mode == Int64Mode.NATIVE_STRICT and base not in _NATIVE_INT64_OPS

    def should_warn_downcast(self, op_name: str) -> bool:
        """True if executing this op demotes int64 to int32 on device (native
        cascade + non-native op → packed view, losing the upper 32 bits)."""
        if self._mode != Int64Mode.NATIVE:
            return False
        return self.decide(op_name) == Int64Decision.DOWNCAST

    def get_compiler_flags(self, op_name: str) -> list[str]:
        """Per-op compiler flags. Non-empty only for NATIVE (S64) ops.

        In the cascade, DOWNCAST ops are demoted to int32 on device via the
        packed view before lowering, so the compiler only ever sees int32 for
        them and needs no int64 flags. Only NATIVE ops keep S64 and pass
        ``--native-int64=enabled`` (public compiler flag; the next SDK release
        flips it to the default). Mirror of
        ``KernelExecution.cpp::GetXLAAdditionalArgs``.
        """
        decision = self.decide(op_name)
        if decision != Int64Decision.NATIVE:
            # DOWNCAST ops are demoted to int32 on device via the packed view
            # before lowering, so the compiler sees int32 and needs no int64
            # flags. UNSUPPORTED ops never reach the compiler (CPU fallback).
            return []
        full_cast = os.environ.get("TORCH_NEURONX_INT64_FULL_CAST") == "1"
        native_flag = "--native-int64=disabled" if full_cast else "--native-int64=enabled"
        return [native_flag, "--implicit-integer-downcast=all"]

    def warn_if_needed(self, op_name: str) -> None:
        """Emit a one-time warning for ops demoted to int32 on device."""
        if not self.should_warn_downcast(op_name):
            return
        base = _strip_overload(op_name)
        if base in _warned_ops:
            return
        _warned_ops.add(base)
        logger.warning(
            "Int64 precision loss: '%s' is not natively supported; it runs on "
            "the low 32 bits via the packed view (upper bits truncated). Set "
            "TORCH_NEURONX_INT64_MODE=native_strict or =cpu_bounce to opt out.",
            base,
        )


def get_native_int64_op_set() -> frozenset[str]:
    return _NATIVE_INT64_OPS


def get_unsupported_int64_op_set() -> frozenset[str]:
    return _UNSUPPORTED_INT64_OPS


def init_cpp_policy() -> None:
    """Initialize the C++ int64 policy op tables from the Python op sets.

    The active mode is read independently on the C++ side from
    TORCH_NEURONX_INT64_MODE (DispatchUtils.h GetInt64Mode), so only the op
    tables are pushed here.
    """
    try:
        from torch_neuronx import _C

        if hasattr(_C, "_init_int64_policy"):
            _C._init_int64_policy(
                list(_NATIVE_INT64_OPS),
                list(_UNSUPPORTED_INT64_OPS),
            )
    except (ImportError, AttributeError) as exc:
        logger.debug("Skipping C++ int64 policy initialization: %s", exc)


def set_int64_mode_for_testing(mode: Int64Mode | str) -> None:
    """TEST ONLY: pin the int64 mode for the current process, syncing every layer.

    Switches the env var, clears the Python lru caches (so ``get_int64_mode`` and
    ``Int64Policy`` re-read it), and overrides the C++ mode storage + re-pushes the
    op tables. This is what lets the pytest ``int64_mode`` marker exercise each
    mode within a single process. Never called by production code.
    """
    if isinstance(mode, str):
        mode = Int64Mode(mode)
    os.environ["TORCH_NEURONX_INT64_MODE"] = mode.value
    reset_int64_mode_cache()
    try:
        from torch_neuronx import _C

        if hasattr(_C, "_set_int64_mode_for_testing"):
            _C._set_int64_mode_for_testing(mode.value)
    except (ImportError, AttributeError) as exc:
        logger.debug("Skipping C++ int64 policy initialization: %s", exc)

    init_cpp_policy()
