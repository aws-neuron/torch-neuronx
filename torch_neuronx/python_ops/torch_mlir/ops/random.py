"""Random operations using on-device NKI RNG."""

import logging
import warnings

import numpy as np
import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type_name
from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

if _has_nki_rng:
    from torch_neuronx.python_ops.nki_kernels.rng_state import _generate_raw

from ..operation_registry import Preprocessing, register_aten

logger = logging.getLogger(__name__)

_GENERATOR_WARNING = (
    "[Neuron] [WARNING] Custom generator is not supported with on-device NKI RNG. "
    "Falling back to CPU for this call. "
    "To use on-device RNG, call without the generator argument."
)


def _dtype_name(dtype):
    """Return the c10::ScalarType name PyTorch uses in error messages.

    Reuses the shared mapping in ``torch_neuronx.dtype_utils`` (e.g. float32 ->
    "Float", float16 -> "Half") so this matches CPU's messages; falls back to a
    capitalized name only for dtypes the shared table does not cover.
    """
    return torch_dtype_to_scalar_type_name(dtype) or str(dtype).rsplit(".", 1)[-1].capitalize()


def _fmt_bound(value):
    """Format a bound the way PyTorch does in errors: drop the .0 for whole floats."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _validate_uniform_inputs(self, from_=0.0, to=1.0, generator=None, **kwargs):
    """Validate uniform_ inputs, matching CPU error types and messages."""
    if not self.is_floating_point() and not self.is_complex():
        raise NotImplementedError(
            f"\"check_uniform_bounds\" not implemented for '{_dtype_name(self.dtype)}'"
        )
    if from_ > to:
        raise RuntimeError(
            f"uniform_ expects to return a [from, to) range, but found "
            f"from={_fmt_bound(from_)} > to={_fmt_bound(to)}"
        )
    # ATen also rejects a span wider than the dtype can represent, e.g.
    # uniform_(float_min, float_max) on float32 (test_uniform_from_to).
    max_val = torch.finfo(self.dtype).max
    if to - from_ > max_val:
        raise RuntimeError(
            f"uniform_ expects to-from <= std::numeric_limits<{_dtype_name(self.dtype)}>::max(), "
            f"but found to={_fmt_bound(to)} and from={_fmt_bound(from_)} "
            f"which result in to-from to exceed the limit"
        )
    _check_fp_bounds_warn(self.dtype, from_, to)


# Largest integer each floating dtype represents exactly. Past this a uniform
# generator loses integer resolution, so ATen warns (see check_from_to_in_range
# in aten/src/ATen/native/DistributionTemplates.h).
_FP_INT_LIMIT = {
    torch.float64: 2**53,
    torch.float32: 2**24,
    torch.float16: 2**11,
    torch.bfloat16: 2**8,
}

# ATen's random_ with to=None caps the upper bound at each dtype's largest
# exactly-representable integer (same reference as above).
_DEFAULT_TO_VAL = {
    torch.bool: 2**1,
    torch.uint8: 2**8,
    torch.int8: 2**7,
    torch.int16: 2**15,
    torch.int32: 2**31,
    # ATen uses INT64_MAX; cap at 2**62 so `to - from` still fits in int64.
    torch.int64: 2**62,
    torch.float16: 2**11,
    torch.bfloat16: 2**8,
    torch.float32: 2**24,
    torch.float64: 2**53,
}

# Neuron legalizes int64 to int32 and rejects constants outside the int32 signed
# range (NCC_ESFH001), so random_ bounds are clamped to this window.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# Unsigned dtypes beyond uint8 that torch.iinfo may reject on some builds.
_UINT_BITS = {}
for _name, _bits in (("uint16", 16), ("uint32", 32), ("uint64", 64)):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _UINT_BITS[_dt] = _bits
        # Same rule as the signed types: cap at the largest exactly-representable
        # value, and keep uint64 within int64 so `to - from` does not overflow.
        _DEFAULT_TO_VAL.setdefault(_dt, 2 ** min(_bits, 62))


def _dtype_value_range(dtype):
    """Inclusive [min, max] integer range representable by ``dtype``.

    ``torch.iinfo``/``finfo`` reject some dtypes (e.g. the unsigned types
    beyond uint8 on older builds), so fall back to an explicit table.
    """
    if dtype == torch.bool:
        return 0, 1
    i64 = torch.iinfo(torch.int64)
    if dtype.is_floating_point:
        try:
            info = torch.finfo(dtype)
            return int(max(info.min, i64.min)), int(min(info.max, i64.max))
        except (TypeError, RuntimeError):
            return i64.min, i64.max
    try:
        info = torch.iinfo(dtype)
        return info.min, info.max
    except (TypeError, RuntimeError):
        bits = _UINT_BITS.get(dtype)
        if bits is not None:
            return 0, 2**bits - 1
        return i64.min, i64.max


def _check_fp_bounds_warn(dtype, from_, to):
    """Warn when a bound exceeds the dtype's exactly-representable integer range.

    Matches ATen: floating dtypes only, message text asserted on by
    ``test_random_from_to`` / ``test_uniform_from_to``.
    """
    limit = _FP_INT_LIMIT.get(dtype)
    if limit is None:
        return
    if not -limit <= from_ <= limit:
        warnings.warn(f"from is out of bounds for {_dtype_name(dtype)}", UserWarning, stacklevel=3)
    if to is not None and not -limit <= to - 1 <= limit:
        warnings.warn(
            f"to - 1 is out of bounds for {_dtype_name(dtype)}", UserWarning, stacklevel=3
        )


def _rand_uniform(ref):
    """Uniform [0, 1) fp32 with ``ref``'s shape, from the on-device RNG if available."""
    if _has_nki_rng:
        return torch.ops.nki_kernels.rand(ref, list(ref.shape), torch.float32)
    return torch.rand(list(ref.shape), dtype=torch.float32, device=ref.device)


def _draw_uniform01(ref, shape):
    """[0, 1) fp32 of ``shape`` from the seeded on-device RNG if available.

    Draws through the ``rng_state.rand`` wrapper (not the raw ``nki_kernels.rand``
    op) so this one leaf legalizes in BOTH the eager register_aten trace and the
    torch.compile fx->StableHLO decomposition path.
    """
    # Use the seeded NKI draw when compiling (trace the custom op -- it legalizes and
    # never recurses) or on a real neuron tensor. Only a real CPU tensor in eager
    # (e.g. the CPU-only decomposition unit tests) takes torch.rand -- which is NOT
    # decomposed in eager, so it cannot recurse into rand_decomp the way it would
    # under the compile backend.
    if _has_nki_rng and (torch.compiler.is_compiling() or ref.device.type != "cpu"):
        from torch_neuronx.python_ops.nki_kernels.rng_state import rand as _wrap_rand

        return _wrap_rand(ref, list(shape), torch.float32)
    return torch.rand(list(shape), dtype=torch.float32, device=ref.device)


def uniform_fill(ref, shape, from_, to, dtype):
    """Fill a shape with uniformly distributed values.

    Args:
        ref (torch.Tensor): Reference tensor supplying the device and tracing context.
        shape (Sequence[int]): Output shape for non-complex output.
        from_ (Number): Inclusive lower bound.
        to (Number): Exclusive upper bound.
        dtype (torch.dtype): Output dtype.

    Returns:
        torch.Tensor: Tensor shaped as ``shape``, or complex ``ref`` after filling its
            real and imaginary components.

    Note:
        Nonempty floating intervals clamp low-precision round-up below ``to``. For a
        complex dtype, ``shape`` is ignored and the components of ``ref`` are filled.
    """
    if dtype.is_complex:
        # Fill the complex output's real view in place: real & imag each get an
        # independent real draw. Avoids the complex64->complex32 cast that drops
        # the imaginary part, and the experimental float16 view_as_complex path.
        real_view = torch.view_as_real(ref)
        real_view.copy_(uniform_fill(real_view, real_view.shape, from_, to, real_view.dtype))
        return ref
    nel = 1
    for _d in shape:
        nel *= int(_d)
    result = (_draw_uniform01(ref, (nel,)) * (to - from_) + from_).to(dtype)
    if dtype.is_floating_point:
        # exclusive upper: clamp any low-precision round-up back below ``to`` at runtime
        # (was the numpy _float_cap_value); no-op for the zero-width to==from case.
        # decrement ``to`` by ~one dtype-ulp arithmetically. torch.nextafter has
        # no Neuron lowering (breaks the torch.compile path: int32/int64 meta
        # mismatch -> invalid StableHLO), so we avoid it. eps is the relative
        # spacing at 1.0 (2**-nmant); the tiny term keeps below<to when to==0.
        _eps = torch.finfo(dtype).eps
        _tiny = torch.finfo(dtype).tiny
        zeros = torch.zeros_like(result)
        to_t = zeros + to
        below = to_t - (torch.abs(to_t) * _eps + _tiny)
        cap = torch.where(to_t > (zeros + from_), below, to_t)
        result = torch.minimum(result, cap)
    return result.reshape(shape)


def _box_muller_standard(ref, draw_shape):
    """N(0, 1) samples of ``draw_shape`` via Box-Muller on two uniform draws."""
    import math as _math

    u1 = torch.clamp(_draw_uniform01(ref, draw_shape), min=1e-7, max=1.0 - 1e-7)
    u2 = _draw_uniform01(ref, draw_shape)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * _math.pi * u2)


def normal_fill(ref, shape, mean, std, dtype):
    """Fill a shape with normally distributed values using Box-Muller.

    Args:
        ref (torch.Tensor): Reference tensor supplying the device and tracing context.
        shape (Sequence[int]): Output shape for non-complex output.
        mean (Number or torch.Tensor): Distribution mean.
        std (Number or torch.Tensor): Distribution standard deviation.
        dtype (torch.dtype): Output dtype.

    Returns:
        torch.Tensor: Tensor shaped as ``shape``, or complex ``ref`` after filling its
            real and imaginary components.

    Note:
        Scalar parameters flatten by element count, while tensor parameters draw at
        ``shape`` for broadcasting. For a complex dtype, ``shape`` is ignored and the
        components of ``ref`` are filled independently.
    """
    if dtype.is_complex:
        real_view = torch.view_as_real(ref)
        real_view.copy_(normal_fill(real_view, real_view.shape, mean, std, real_view.dtype))
        return ref
    if isinstance(mean, torch.Tensor) or isinstance(std, torch.Tensor):
        z = _box_muller_standard(ref, shape)
        return (mean + std * z).to(dtype)
    nel = 1
    for _d in shape:
        nel *= int(_d)
    z = _box_muller_standard(ref, (nel,))
    if std == 1.0 and mean == 0.0:
        return z.reshape(shape).to(dtype)
    return (z * std + mean).reshape(shape).to(dtype)


try:
    import ml_dtypes

    _NP_FLOAT = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bfloat16: ml_dtypes.bfloat16,
    }
except ImportError:  # ml_dtypes optional; without it the bf16 cap falls back to no-clamp
    _NP_FLOAT = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
    }


def _float_cap_value(dtype, to):
    """Largest value of ``dtype`` strictly below ``to``, or ``None`` if none needed.

    In low-precision floats, a value just below ``to`` can round up to ``to``
    (e.g. 32751 -> 32752 in float16); CPU caps at the largest value < to. Kept
    pure-Python (numpy/ml_dtypes) so it stays a trace-time constant -- torch ops
    here (``.item()``/``nextafter``) inject a ``lift_fresh_copy`` that AOTAutograd
    rejects on the compile path. ``None`` for ints or a ``to`` not representable.
    """
    if to is None or not dtype.is_floating_point:
        return None
    npd = _NP_FLOAT.get(dtype)
    if npd is None:
        return None
    to_repr = npd(to)
    if float(to_repr) < to:
        return None
    return float(np.nextafter(to_repr, npd(float("-inf"))))


# Narrow ints computed on an int32 proxy and cast back (see _factory): uint16/32/64
# have no NKI dtype-map entry, and int8/int16/uint8 trip codegen assert NCC_IBCG902
# on the narrow int32->int8/16 convert (Kaena Compiler NCC-9649 / P521512907).
# int64 is not here — it CPU-falls-back via the int64 dispatch policy instead.
_INT32_PROXY_DTYPES = {torch.int8, torch.int16, torch.uint8} | {
    d for d in (getattr(torch, n, None) for n in ("uint16", "uint32", "uint64")) if d is not None
}

# Torch-MLIR rejects a direct int32->fp16/bf16 convert, so fp16/bf16 route through fp32.
_FP32_PROXY_DTYPES = (torch.float16, torch.bfloat16)


def _to_result_dtype(result, dtype):
    """Cast to ``dtype`` for the fill; int->fp16/bf16 goes via fp32 (direct convert rejected)."""
    if dtype in _FP32_PROXY_DTYPES and not result.dtype.is_floating_point:
        result = result.to(torch.float32)
    return result.to(dtype)


def _factory(self, compute_fn, extra_args):
    """Assemble the factory tuple ``(compute_fn, args, kwargs[, post])``.

    ``compute_fn(carrier, *extra_args, out=...)``. Ordinary dtypes carry on ``self``
    in-place; narrow ints carry on a fresh int32 proxy that ``_cast_back`` copies into
    ``self``. fp16/bf16 need no proxy — the compute_fn converts int32->fp32->dtype in-graph.
    """
    if self.dtype in _INT32_PROXY_DTYPES:
        proxy = torch.empty_like(self, dtype=torch.int32)
        real = self

        def _cast_back(res):
            real.copy_(res.to(real.dtype))
            return real

        return compute_fn, (proxy, *extra_args), {"out": proxy}, _cast_back
    return compute_fn, (self, *extra_args), {"out": self}


def _validate_random_inputs(self, from_=None, to=None, generator=None, **kwargs):
    """Validate random_ / random_.from bounds, matching CPU errors and warnings.

    Order matters and is asserted on by ``test_random_from_to``: the empty-range
    check comes first (it wins even when ``from`` is also outside the dtype
    range), then ``from`` bounds, then ``to - 1`` bounds, then the float
    resolution warnings.
    """
    if from_ is None:
        return  # bare random_(): bounds are derived internally, nothing to check
    if to is not None and to <= from_:
        raise RuntimeError(
            f"random_ expects 'from' to be less than 'to', but got "
            f"from={_fmt_bound(from_)} >= to={_fmt_bound(to)}"
        )
    min_val, max_val = _dtype_value_range(self.dtype)
    if not min_val <= from_ <= max_val:
        raise RuntimeError(f"from is out of bounds for {_dtype_name(self.dtype)}")
    if to is not None and not min_val <= to - 1 <= max_val:
        raise RuntimeError(f"to - 1 is out of bounds for {_dtype_name(self.dtype)}")
    _check_fp_bounds_warn(self.dtype, from_, to)


@register_aten(
    ["aten::uniform_"],
    static_argnums=(1, 2),
    validate_static_inputs=_validate_uniform_inputs,
    output_params=("self",),
)
def torch_uniform_(self, from_: float = 0.0, to: float = 1.0, generator=None, **kwargs):
    """Fill a tensor with values from a uniform distribution.

    Args:
        self (torch.Tensor): Destination tensor.
        from_ (float, optional): Inclusive lower bound. Defaults to ``0.0``.
        to (float, optional): Exclusive upper bound. Defaults to ``1.0``.
        generator (torch.Generator or None, optional): Unsupported explicit generator.
            Defaults to ``None``.
        **kwargs (Any): Additional registration arguments, which are ignored.

    Returns:
        torch.Tensor: Uniformly filled tensor.

    Raises:
        NotImplementedError: If ``self`` is neither floating point nor complex.
        RuntimeError: If a generator is supplied or the requested bounds are invalid.
    """
    if generator is not None:
        raise RuntimeError(_GENERATOR_WARNING)
    return uniform_fill(self, self.shape, from_, to, self.dtype)


def _validate_normal_inputs(self, mean=0.0, std=1.0, generator=None, **kwargs):
    """Validate normal_ inputs, matching CPU error types and messages."""
    if not self.is_floating_point() and not self.is_complex():
        raise NotImplementedError(
            f"\"normal_kernel_cpu\" not implemented for '{_dtype_name(self.dtype)}'"
        )
    if std < 0:
        raise RuntimeError(f"normal expects std >= 0.0, but found std {std}")


@register_aten(
    ["aten::normal_"],
    validate_static_inputs=_validate_normal_inputs,
    output_params=("self",),
)
def torch_normal_(self, mean: float = 0.0, std: float = 1.0, generator=None, **kwargs):
    """Fill a tensor with values from a normal distribution.

    Args:
        self (torch.Tensor): Destination tensor.
        mean (float, optional): Distribution mean. Defaults to ``0.0``.
        std (float, optional): Distribution standard deviation. Defaults to ``1.0``.
        generator (torch.Generator | None, optional): Unsupported explicit generator.
            Defaults to ``None``.
        **kwargs (Any): Additional registration arguments, which are ignored.

    Returns:
        torch.Tensor: Normally distributed tensor.

    Raises:
        RuntimeError: If ``generator`` is not ``None``.
    """
    if generator is not None:
        raise RuntimeError(_GENERATOR_WARNING)
    return normal_fill(self, self.shape, mean, std, self.dtype)


@register_aten(
    ["aten::random_"],
    static_argnums=(1, 2, 3),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    validate_static_inputs=_validate_random_inputs,
    output_params=("self",),
)
def torch_random_(self, from_: int | None = None, to: int | None = None, generator=None, **kwargs):
    """Prepare an in-place uniform integer fill.

    Args:
        self (torch.Tensor): Destination tensor.
        from_ (int | None, optional): Inclusive lower bound. Defaults to ``None``.
        to (int | None, optional): Exclusive upper bound. Defaults to ``None``.
        generator (torch.Generator | None, optional): Unsupported explicit generator.
            Defaults to ``None``.
        **kwargs (Any): Additional registration arguments, which are ignored.

    Returns:
        tuple: Preprocessing tuple containing a compute function, positional arguments, output
            keyword arguments, and an optional narrow-integer cast-back function.

    Raises:
        RuntimeError: If an explicit generator is supplied or no default upper bound exists.
    """
    if generator is not None:
        raise RuntimeError(_GENERATOR_WARNING)

    dtype = self.dtype  # real destination dtype, before any int32 widening

    if dtype == torch.bool:
        # Honor explicit bounds: random_(0, 1) can only produce False and
        # random_(1, 2) only True. Only the full [0, 2) range is a coin flip.
        lo = 0 if from_ is None else from_
        hi = 2 if to is None else to
        if hi - lo <= 1:

            def fill_const(self, value, out=None):
                """Fill a Boolean destination with one constant.

                Args:
                    self (torch.Tensor): Tensor supplying shape, dtype, and device.
                    value (bool): Constant value to fill.
                    out (torch.Tensor | None, optional): Unused factory output destination.
                        Defaults to ``None``.

                Returns:
                    torch.Tensor: Constant tensor matching ``self``.
                """
                return torch.full_like(self, value)

            return _factory(self, fill_const, (bool(lo),))

        def coin_flip(self, out=None):
            """Generate a Boolean coin-flip tensor.

            Args:
                self (torch.Tensor): Tensor supplying shape and device.
                out (torch.Tensor | None, optional): Unused factory output destination.
                    Defaults to ``None``.

            Returns:
                torch.Tensor: Boolean tensor whose values compare uniform draws against ``0.5``.
            """
            return _rand_uniform(self) >= 0.5

        return _factory(self, coin_flip, ())

    # Bare random_() is [0, iinfo.max] with from=0 (ATen), so signed dtypes stay
    # non-negative. Fold raw bits into [0, max]; modulus is the real dtype's (int8/16
    # fold correctly). The float path can't cover a 2**31-wide int32 span (overflow).
    if from_ is None and to is None and _has_nki_rng and not dtype.is_floating_point:
        _, dtype_max = _dtype_value_range(dtype)
        modulus = min(dtype_max, _INT32_MAX) + 1
        mask = modulus - 1

        if modulus & mask == 0:
            # dtype_max+1 is a power of two, so fold via bitmask not modulo.
            def fold(self, mask, out=None):
                """Fold raw random bits into a power-of-two integer range.

                Args:
                    self (torch.Tensor): Tensor supplying output shape, dtype, and device.
                    mask (int): Bitmask defining the inclusive upper value.
                    out (torch.Tensor | None, optional): Unused factory output destination.
                        Defaults to ``None``.

                Returns:
                    torch.Tensor: Masked raw values cast to the destination dtype.
                """
                raw = _generate_raw(self, list(self.shape))
                return (raw & mask).to(self.dtype)

            return _factory(self, fold, (mask,))

        def modulo(self, modulus, out=None):
            """Fold raw random values into a non-power-of-two range.

            Args:
                self (torch.Tensor): Tensor supplying output shape, dtype, and device.
                modulus (int): Positive exclusive upper bound.
                out (torch.Tensor | None, optional): Unused factory output destination.
                    Defaults to ``None``.

            Returns:
                torch.Tensor: Absolute raw values reduced modulo ``modulus``.
            """
            raw = _generate_raw(self, list(self.shape))
            return (raw.abs() % modulus).to(self.dtype)

        return _factory(self, modulo, (modulus,))

    # ATen's default lower bound is 0 for every dtype (not iinfo.min): bare
    # random_() is [0, to), so signed dtypes never draw negatives.
    if from_ is None:
        from_ = 0
    if to is None:
        to = _DEFAULT_TO_VAL.get(dtype)
        if to is None:
            raise RuntimeError(f"random_: no default upper bound for dtype {dtype}")

    span = to - from_
    # A full-int32-range span is exactly the raw bits — return them directly (modulo
    # can't express span==2**32, and the float path loses the low bits).
    if _has_nki_rng and from_ <= _INT32_MIN and to - 1 >= _INT32_MAX:

        def raw_full(self, out=None):
            """Generate values spanning the full int32 range.

            Args:
                self (torch.Tensor): Tensor supplying output shape, dtype, and device.
                out (torch.Tensor | None, optional): Unused factory output destination.
                    Defaults to ``None``.

            Returns:
                torch.Tensor: Raw int32 values converted to the destination dtype.
            """
            return _to_result_dtype(_generate_raw(self, list(self.shape)), self.dtype)

        return _factory(self, raw_full, ())

    # Bounded: raw bits % span (ATen's approach). Only valid while span fits int32
    # (Neuron rejects out-of-int32 constants, NCC_ESFH001); wider spans use the float path.
    if _has_nki_rng and _INT32_MIN < span <= _INT32_MAX:
        cap = _float_cap_value(dtype, to)  # eager (reads scalars); passed as a constant

        def bounded(self, from_val, span, cap, out=None):
            """Generate bounded integers with raw-value modulo reduction.

            Args:
                self (torch.Tensor): Tensor supplying output shape, dtype, and device.
                from_val (int): Inclusive lower bound.
                span (int): Width of the requested range.
                cap (Number | None): Optional dtype-representable upper cap.
                out (torch.Tensor | None, optional): Unused factory output destination.
                    Defaults to ``None``.

            Returns:
                torch.Tensor: Values in the requested range, optionally clamped to ``cap``.
            """
            raw = _generate_raw(self, list(self.shape))
            result = _to_result_dtype(raw.abs() % span + from_val, self.dtype)
            return result if cap is None else torch.clamp(result, max=cap)

        return _factory(self, bounded, (from_, span, cap))

    # Fallback (no NKI, or span too wide for int32): scale a float uniform, bounds
    # clamped into the int32 window.
    lo = max(from_, _INT32_MIN)
    hi = min(to, _INT32_MAX)
    cap = _float_cap_value(dtype, to)

    def scaled(self, lo, hi, cap, out=None):
        """Scale uniform floating-point draws into an integer range.

        Args:
            self (torch.Tensor): Tensor supplying output shape, dtype, and device.
            lo (int): Inclusive clamped lower bound.
            hi (int): Exclusive clamped upper bound.
            cap (Number | None): Optional dtype-representable upper cap.
            out (torch.Tensor | None, optional): Unused factory output destination.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Scaled values, a constant when the range is empty, optionally capped.
        """
        if hi <= lo:
            return torch.full_like(self, lo)
        u = _rand_uniform(self)
        result = _to_result_dtype((u * float(hi - lo)).to(torch.int32) + lo, self.dtype)
        return result if cap is None else torch.clamp(result, max=cap)

    return _factory(self, scaled, (lo, hi, cap))


@register_aten(
    ["aten::random_.from"],
    static_argnums=(1, 2, 3),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    validate_static_inputs=_validate_random_inputs,
    output_params=("self",),
)
def torch_random_from(self, from_: int = 0, to: int | None = None, generator=None, **kwargs):
    """Prepare an in-place uniform integer fill with an explicit lower bound.

    Args:
        self (torch.Tensor): Destination tensor.
        from_ (int, optional): Inclusive lower bound. Defaults to ``0``.
        to (int | None, optional): Exclusive upper bound. Defaults to ``None``.
        generator (torch.Generator | None, optional): Unsupported explicit generator.
            Defaults to ``None``.
        **kwargs (Any): Additional registration arguments forwarded to ``torch_random_``.

    Returns:
        tuple: Preprocessing factory returned by ``torch_random_``.

    Raises:
        RuntimeError: If delegated random-fill validation fails.
    """
    return torch_random_(self, from_=from_, to=to, generator=generator, **kwargs)


def _validate_random_to(self, to=None, generator=None, **kwargs):
    """Validate random_.to, whose lower bound is implicitly 0 (reuses the shared check)."""
    _validate_random_inputs(self, from_=0, to=to, generator=generator)


@register_aten(
    ["aten::random_.to"],
    static_argnums=(1, 2, 3),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    validate_static_inputs=_validate_random_to,
    output_params=("self",),
)
def torch_random_to(self, to: int | None = None, generator=None, **kwargs):
    """Prepare an in-place uniform integer fill with a zero lower bound.

    Args:
        self (torch.Tensor): Destination tensor.
        to (int | None, optional): Exclusive upper bound. Defaults to ``None``.
        generator (torch.Generator | None, optional): Unsupported explicit generator.
            Defaults to ``None``.
        **kwargs (Any): Additional registration arguments forwarded to ``torch_random_``.

    Returns:
        tuple: Preprocessing factory returned by ``torch_random_``.

    Raises:
        RuntimeError: If delegated random-fill validation fails.
    """
    return torch_random_(self, from_=0, to=to, generator=generator, **kwargs)
