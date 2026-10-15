"""Int64 execution-mode policy for Neuron.

A single env var, ``TORCH_NEURONX_INT64_MODE``, selects how int64 tensors are
handled during Neuron dispatch:

  - ``native`` (default): per-op cascade. A natively-supported op runs as S64
    on the compiler (GPSIMD); a non-native op is demoted on device via the
    packed view (low 32 bits); the ops in ``_UNSUPPORTED_INT64_OPS`` (for
    example matmul and linear) fall back to CPU. Its conv entries do not match:
    convolution reaches dispatch as ``aten::convolution``, so it runs on device
    through the packed view. The "best available path per op".
  - ``packed``: force every int64 op through the packed view (low-word
    on-device extraction). The policy sends no int64 op to CPU in this mode.
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
    """Define int64 execution modes for Neuron dispatch.

    Note:
        String values match ``TORCH_NEURONX_INT64_MODE``. The mirrored C++
        ``Int64Mode`` uses stable codes: ``cpu_bounce=0``, ``native=1``,
        ``packed=2``, ``native_strict=3``, and ``cpu_fallback=4``.
    """

    CPU_BOUNCE = "cpu_bounce"
    NATIVE = "native"
    PACKED = "packed"
    NATIVE_STRICT = "native_strict"
    CPU_FALLBACK = "cpu_fallback"


class Int64Decision(str, Enum):
    """Define per-op routing decisions shared with the C++ policy.

    Note:
        Values mirror the C++ ``Int64Decision`` enum.
    """

    NATIVE = "native"
    DOWNCAST = "downcast"
    UNSUPPORTED = "unsupported"


@lru_cache(maxsize=1)
def get_int64_mode() -> Int64Mode:
    """Read and cache the int64 execution mode from the environment.

    Returns:
        Int64Mode: The normalized ``TORCH_NEURONX_INT64_MODE`` value, or
        ``Int64Mode.NATIVE`` when the variable is unset.

    Raises:
        ValueError: If the configured value is not a recognized mode.

    Note:
        The first call caches the result. Supported values are ``native``,
        ``native_strict``, ``packed``, ``cpu_bounce``, and ``cpu_fallback``.
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
    """Check whether the active int64 mode is packed.

    Returns:
        bool: ``True`` when the active mode is ``packed``.
    """
    return get_int64_mode() == Int64Mode.PACKED


def is_native_int64_mode() -> bool:
    """Check whether the active int64 mode is native.

    Returns:
        bool: ``True`` when the active mode is ``native``.
    """
    return get_int64_mode() == Int64Mode.NATIVE


def is_native_strict_int64_mode() -> bool:
    """Check whether the active int64 mode is native strict.

    Returns:
        bool: ``True`` when the active mode is ``native_strict``.
    """
    return get_int64_mode() == Int64Mode.NATIVE_STRICT


def is_cpu_bounce_int64_mode() -> bool:
    """Check whether int64 uses a CPU round-trip cast.

    Returns:
        bool: ``True`` when casts occur on CPU but the operation runs on a Neuron device.
    """
    return get_int64_mode() == Int64Mode.CPU_BOUNCE


def is_cpu_fallback_int64_mode() -> bool:
    """Check whether every int64 operation uses CPU fallback.

    Returns:
        bool: ``True`` when no int64 operation executes on a Neuron device.
    """
    return get_int64_mode() == Int64Mode.CPU_FALLBACK


def is_s64_passthrough() -> bool:
    """Check whether the active mode permits S64 compiler passthrough.

    Returns:
        bool: ``True`` when the active mode is ``native`` or ``native_strict``.
    """
    return get_int64_mode() in (Int64Mode.NATIVE, Int64Mode.NATIVE_STRICT)


def use_packed_int64_view_for_op(op_name: str) -> bool:
    """Check whether an operation uses the packed int64 physical view.

    Args:
        op_name (str): Operation name to classify.

    Returns:
        bool: ``True`` in packed mode or for a native-mode downcast operation;
            ``False`` otherwise.

    Note:
        This mirrors C++ ``UsePackedInt64ViewForOp`` so fake traced parameters use
        the same ``int32[..., 2]`` physical view bound by the C++ dispatcher.
    """
    if is_packed_int64_mode():
        return True
    if not is_s64_passthrough():
        return False
    return Int64Policy.get().decide(op_name) == Int64Decision.DOWNCAST


def reset_int64_mode_cache() -> None:
    """Clear the cached Python int64 mode and policy.

    Note:
        This helper is intended only for tests.
    """
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
        # Comparison
        "aten::eq",
        "aten::ne",
        "aten::gt",
        "aten::ge",
        "aten::lt",
        "aten::le",
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
        "aten::eye",
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
        # Embedding - DISABLED. Native S64 embedding gather reads out of
        # bounds at runtime (NRT 1006). The packed int32 view races
        # cross-stream under HostCC and gives an intermittent NaN. Route it
        # to a CPU int32 index roundtrip via DowncastArgs
        # (kCpuBounceInt64Ops) instead. Re-enable once the cross-stream
        # int64 hazard has a proper fix.
        # "aten::embedding",
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
        "aten::random_",  # covers .from/.to; int64 RNG is slower on-device than CPU fallback
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
    """Route int64 operations under a selected execution mode.

    Args:
        mode (Int64Mode): Execution mode used for per-operation decisions.

    Note:
        In ``packed`` and ``cpu_bounce`` modes, this policy never forces whole-op
        CPU fallback; those modes retain their existing on-device handling.
    """

    def __init__(self, mode: Int64Mode):
        self._mode = mode

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get() -> Int64Policy:
        """Return the cached process-wide int64 policy.

        Returns:
            Int64Policy: Policy initialized from the cached active execution mode.
        """
        return Int64Policy(get_int64_mode())

    @staticmethod
    def reset_cache() -> None:
        """Clear the cached process-wide int64 policy."""
        Int64Policy.get.cache_clear()

    @property
    def mode(self) -> Int64Mode:
        """Return the policy execution mode.

        Returns:
            Int64Mode: Mode captured when this policy was created.
        """
        return self._mode

    @property
    def passes_s64_to_compiler(self) -> bool:
        """Check whether the mode permits S64 compiler passthrough.

        Returns:
            bool: ``True`` when the mode is ``native`` or ``native_strict``.
        """
        return self._mode in (Int64Mode.NATIVE, Int64Mode.NATIVE_STRICT)

    # Back-compat alias used by kernel.py / op_impl.py / etc.
    @property
    def native_int64_enabled(self) -> bool:
        """Check whether native int64 compiler handling is enabled.

        Returns:
            bool: The value of ``passes_s64_to_compiler``.

        Note:
            This property is a backward-compatible alias used by dispatch call sites.
        """
        return self.passes_s64_to_compiler

    def decide(self, op_name: str) -> Int64Decision:
        """Classify the int64 routing decision for an operation.

        Args:
            op_name (str): Operation name, optionally including an overload suffix.

        Returns:
            Int64Decision: ``UNSUPPORTED`` for CPU fallback, ``NATIVE`` for a listed
            native op in an S64 mode, or ``DOWNCAST`` otherwise.
        """
        if self.should_cpu_fallback(op_name):
            return Int64Decision.UNSUPPORTED
        base = _strip_overload(op_name)
        if self.passes_s64_to_compiler and base in _NATIVE_INT64_OPS:
            return Int64Decision.NATIVE
        return Int64Decision.DOWNCAST

    def should_cpu_fallback(self, op_name: str) -> bool:
        """Determine whether an int64 operation should use CPU fallback.

        Args:
            op_name (str): Operation name, optionally including an overload suffix.

        Returns:
            bool: ``True`` for every op in ``cpu_fallback`` mode, unsupported ops in
            native modes, and non-native ops in ``native_strict`` mode. ``packed`` and
            ``cpu_bounce`` modes never force whole-op CPU fallback.

        Note:
            Callers must separately verify that the operation has an int64 input.
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
        """Determine whether an operation needs a precision-loss warning.

        Args:
            op_name (str): Operation name to classify.

        Returns:
            bool: ``True`` for a ``DOWNCAST`` operation in ``native`` mode, where the
            packed view drops the upper 32 bits.
        """
        if self._mode != Int64Mode.NATIVE:
            return False
        return self.decide(op_name) == Int64Decision.DOWNCAST

    def get_compiler_flags(self, op_name: str) -> list[str]:
        """Return int64 compiler flags for an operation.

        Args:
            op_name (str): Operation name to classify.

        Returns:
            list[str]: Empty for ``DOWNCAST`` or ``UNSUPPORTED`` operations; otherwise
                contains the native-int64 and implicit-downcast flags.

        Note:
            Downcast inputs use the packed view in ``packed`` and ``native`` modes, but
            use a CPU round trip in ``cpu_bounce`` mode; unsupported operations never
            reach the compiler. ``TORCH_NEURONX_INT64_FULL_CAST=1`` sets the native-int64
            flag to ``disabled``; otherwise native operations use
            ``--native-int64=enabled``.
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
        """Emit a one-time warning for an int64 downcast operation.

        Args:
            op_name (str): Operation name to classify after removing its overload.

        Note:
            Each base op warns once per process when ``native`` mode runs it through
            the packed low-32-bit view. The warning identifies ``native_strict`` and
            ``cpu_bounce`` as opt-out modes.
        """
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
    """Return operations with confirmed native S64 compiler support.

    Returns:
        frozenset[str]: Base aten operation names that may execute natively.
    """
    return _NATIVE_INT64_OPS


def get_unsupported_int64_op_set() -> frozenset[str]:
    """Return operations routed to CPU fallback in native int64 modes.

    Returns:
        frozenset[str]: Base aten operation names routed away from Neuron in native modes.
    """
    return _UNSUPPORTED_INT64_OPS


def init_cpp_policy() -> None:
    """Initialize C++ int64 operation tables from the Python sets.

    Note:
        C++ reads the active mode independently from ``TORCH_NEURONX_INT64_MODE``;
        this function pushes only native and unsupported operation tables. An
        unavailable ``_C`` module is logged at debug level; an absent binding is
        ignored.
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
    """Set the process int64 mode across Python and C++ test layers.

    Args:
        mode (Int64Mode or str): Mode to store in ``TORCH_NEURONX_INT64_MODE``.

    Raises:
        ValueError: If a string does not identify a valid ``Int64Mode``.

    Note:
        This test-only helper updates the environment, clears both Python LRU
        caches, overrides C++ mode storage when available, and re-pushes the
        operation tables. Production code never calls it.
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
