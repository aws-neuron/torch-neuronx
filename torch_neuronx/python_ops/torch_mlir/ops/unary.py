"""Unary operations."""

import math
import struct

import torch

from ..operation_registry import register_aten
from ..validation import (
    aten_scalar_type_name,
    validate_float_only_dtype,
    validate_round_dtype,
)

# Neuron's hardware exp and cos are inaccurate: measured max rel error on trn2
# in fp32 is 1.0e-05 for exp and 5.9e-05 for cos, vs 8e-08 for sin/tanh (fp32
# eps is 1.2e-07). That is above the 1.3e-6 tolerance the upstream tests use, so
# cosh/sinh/tan below don't call the hardware ops directly -- they reduce the
# argument to a small range and evaluate it accurately there. Both reductions
# rely on argument reduction, which needs the Cody-Waite constant split below.


def _cody_waite_split(constant, parts, keep_bits=8):
    """Split a constant into ``parts`` fp32 chunks that sum to it, each holding
    only its top ``keep_bits`` mantissa bits (the rest zeroed).

    Both reductions below rewrite ``x = n*C + r`` and evaluate on the small
    remainder ``r = x - n*C`` (C = pi for tan, ln2 for exp; the call sites say
    why). ``r`` is only as accurate as ``n*C``: a single fp32 ``C`` has 24
    mantissa bits, so for a large integer ``n`` the product ``n*C`` rounds and
    the low bits of ``r`` -- exactly the bits we care about -- come out wrong.

    Cody-Waite fixes this by splitting ``C = c0 + c1 + ...`` with each ``ci``'s
    low mantissa bits zeroed. Because a zeroed-low chunk has few significant
    bits, ``n*ci`` still lands in 24 bits *exactly* (no rounding) for
    ``|n| < 2**(23 - keep_bits)``; subtracting the chunks one at a time then
    peels the reduction off in exact steps.

    Worked decimal analogue (3 significant digits, C = 3.14159):
        split -> c0 = 3.14, c1 = 0.00159   (c0's low digits zeroed)
        n = 50:  50*C = 157.0795, but 3-digit arithmetic rounds it to 157.
                 50*c0 = 157.0 and 50*c1 = 0.0795 are each exact, so
                 (x - 50*c0) - 50*c1 keeps the 0.0795 a lone 50*C would drop.
    """
    chunks = []
    remainder = float(constant)
    for _ in range(parts - 1):
        raw = struct.unpack("<I", struct.pack("<f", remainder))[0]
        raw &= ~((1 << (23 - keep_bits)) - 1) & 0xFFFFFFFF
        hi = struct.unpack("<f", struct.pack("<I", raw))[0]
        chunks.append(hi)
        remainder -= hi
    chunks.append(remainder)
    return chunks


# The reduction constants are the true math values (from the math library, not
# hand-typed): tan reduces modulo pi because tan has period pi, and exp reduces
# modulo ln2 because exp(k*ln2) = 2**k is an exact power of two to scale by.
# Chunk count = how many exact pieces the largest n needs. tan handles inputs up
# to ~1e20 (n huge), so pi needs 3 chunks to stay exact; exp's n = round(x/ln2)
# is at most ~128 over the finite range, so ln2 needs only 2.
_PI_HI, _PI_MID, _PI_LO = _cody_waite_split(math.pi, 3)
_LN2_HI, _LN2_LO = _cody_waite_split(math.log(2.0), 2)

# sinh's exp form cancels catastrophically near 0; its Taylor series degrades
# past 1. Measured at x=1: Taylor 8.0e-08, exp 1.0e-07.
_SINH_TAYLOR_LIMIT = 1.0

# expm1 near 0: exp(x) - 1 cancels; below this |x| use the Taylor series. Above
# it the reduced exp is accurate. Both stay under the fp32 tolerance at 0.35.
_EXPM1_TAYLOR_LIMIT = 0.35

# expm1 large negative: below this x, exp(x) is 0 to float precision, so
# expm1(x) = -1. Saturate here to skip the range reduction, which returns nan for
# very negative x (2**k underflows to 0 and exp(r) is non-finite).
_EXPM1_NEG_SATURATE = -40.0


def _exp_reduced(x, exp_bias=0):
    """Accurate exp(x) * 2**exp_bias, sidestepping the hardware exp unit.

    Uses the identity exp(x) = 2**k * exp(r), where k = round(x/ln2) and
    r = x - k*ln2 with |r| <= ln2/2. The Taylor series exp(r) = 1 + r + r²/2!
    + r³/3! + ... + r⁷/7! (the polynomial coefficients 1, 1, 1/2, 1/6, 1/24,
    1/120, 1/720, 1/5040 are literally 1/n! for n = 0..7) converges quickly for
    small |r|. The 2**k factor is an exact power-of-two scale. This avoids the
    hardware exp's 1e-05 error, measuring 7e-08 instead.

    exp_bias folds an extra power of two into 2**(k + exp_bias) so that callers
    can absorb a constant multiplier without an extra pass. cosh/sinh use
    exp_bias=-2 (i.e. dividing by 4 before the final combine), because at the
    fp32 edge (x~89.4, cosh still finite on CPU) k = round(89.4/0.693) = 129 and
    a bare 2**129 overflows fp32 (max is ~2**127.99). Biasing by -2 makes the
    intermediate 2**127 which fits. The caller reconstructs the full value by
    multiplying later (see the *2.0 in the cosh/sinh formulas).

    The callers combine exp(|x|)/4 and its reciprocal into cosh/sinh:
        cosh(|x|) = exp(|x|)/2 + exp(-|x|)/2
                  = (exp(|x|)/4)*2 + 1/(exp(|x|)/4) * (1/16) * 2
                  = quarter_e*2 + 0.0625/quarter_e * 2
    where 0.0625 = 1/16 = (1/4)*(1/4) accounts for both the /4 in quarter_e and
    the /4 in the reciprocal's denominator. The same logic gives sinh its sign.
    """
    k = torch.round(x / math.log(2.0))
    # Clamp k to its meaningful range. exp overflows fp32 once k exceeds ~128,
    # so the result is +inf beyond that regardless of k's exact value. A huge
    # input gives a huge k, and 2**k * poly(r) then evaluates to nan instead of
    # +inf. Clamping keeps the finite domain (|k| <= ~129) exact, forces a clean
    # +inf past it, and decouples this reduction from round's large-value
    # handling.
    k = torch.clamp(k, min=-255.0, max=255.0)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    poly = 1.0 + r * (
        1.0 + r * (0.5 + r * (1 / 6 + r * (1 / 24 + r * (1 / 120 + r * (1 / 720 + r / 5040)))))
    )
    return poly * torch.pow(2.0, k + exp_bias)


def _validate_float_promoting_out(x, *args, out=None, **kwargs):
    """Reject an out= tensor whose dtype can't hold the float result.

    Mirrors PyTorch's structured-kernel unsafe-cast guard, which the Neuron
    dispatch path skips (it would otherwise silently cast the result into out).
    These ops promote integer/bool inputs to the default float dtype, so the
    expected out dtype is the input dtype only when it is already floating.
    """
    # Unwrap tuple out= (e.g. max.dim returns (values, indices)); check values only.
    out_tensor = out[0] if isinstance(out, tuple | list) else out
    if out_tensor is None or not isinstance(out_tensor, torch.Tensor):
        return
    # Only check when the input is floating-point: the result is float, and an
    # integer out= cannot hold it. For integer/bool inputs the result dtype is
    # also integer (or stays bool), so any integer out= is acceptable.
    if not x.is_floating_point():
        return
    if out_tensor.is_floating_point():
        return
    expected = x.dtype
    raise RuntimeError(
        f"Expected out tensor to have dtype {expected}, " f"but got {out_tensor.dtype} instead"
    )


@register_aten(
    ["aten::abs", "aten::abs.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_abs(x, out=None):
    """Compute the elementwise absolute value.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Absolute values with negative floating-point zero normalized.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Returns the input tensor itself for unsigned 8-bit data because StableHLO
        absolute value does not support that dtype.
    """
    if x.dtype == torch.uint8:
        return x
    result = torch.abs(x)
    if x.is_floating_point():
        result = torch.where(x == 0, torch.zeros_like(result), result)
    return result


@register_aten(
    ["aten::neg", "aten::neg.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_neg(x, out=None):
    # neuronx-cc fails to compile an int32 negate that feeds the transpose
    # epilogue alone (contiguous input, permuted output). A subtraction from
    # zero compiles, gives the same result, and measures faster than negate
    # for int32. Other dtypes compile with negate.
    """Negate each input element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise negated values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    if x.dtype == torch.int32:
        return 0 - x
    return torch.neg(x)


@register_aten(
    ["aten::sqrt", "aten::sqrt.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sqrt(x, out=None):
    """Compute the elementwise square root.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise square roots.
    """
    return torch.sqrt(x)


@register_aten(
    ["aten::rsqrt", "aten::rsqrt.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_rsqrt(x, out=None):
    """Compute the reciprocal square root of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise reciprocal square roots.
    """
    return torch.rsqrt(x)


@register_aten(
    ["aten::cos", "aten::cos.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_cos(x, out=None):
    """Compute the cosine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise cosine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.cos(x)


@register_aten(
    ["aten::sin", "aten::sin.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sin(x, out=None):
    """Compute the sine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise sine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.sin(x)


@register_aten(
    ["aten::cosh", "aten::cosh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_cosh(x, out=None):
    """Compute the hyperbolic cosine of each element.

    Args:
        x (torch.Tensor): Input tensor, promoted to the default float dtype if needed.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise hyperbolic cosine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Finite ``torch.float32`` and ``torch.float64`` inputs use the reduced exponential formula
        ``(exp(x) + exp(-x)) / 2``. ``torch.float16``, ``torch.bfloat16``, and non-finite inputs use
        the hardware operation to avoid polynomial error or invalid reduction.
    """
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    if x.dtype not in (torch.float32, torch.float64):
        return torch.cosh(x)
    # cosh is even, so reduce |x|: that keeps k positive and leaves overflow as
    # the only failure direction (2**k underflowing to 0 for very negative x
    # would make the reciprocal term inf and the product nan).
    quarter_e = _exp_reduced(x.abs(), exp_bias=-2)
    computed = (quarter_e + 0.0625 / quarter_e) * 2.0
    return torch.where(torch.isfinite(x), computed, torch.cosh(x))


@register_aten(
    ["aten::sinh", "aten::sinh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sinh(x, out=None):
    """Compute the hyperbolic sine of each element.

    Args:
        x (torch.Tensor): Input tensor, promoted to the default float dtype if needed.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise hyperbolic sine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Finite ``torch.float32`` and ``torch.float64`` inputs use a Taylor series for ``abs(x) < 1``
        and the reduced formula ``(exp(x) - exp(-x)) / 2`` otherwise, avoiding near-zero
        cancellation. ``torch.float16``, ``torch.bfloat16``, and non-finite inputs use hardware.
    """
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    if x.dtype not in (torch.float32, torch.float64):
        return torch.sinh(x)
    # sinh is odd, so take the magnitude from |x| and restore the sign, for the
    # same reason as torch_cosh. The series is already odd in x.
    quarter_e = _exp_reduced(x.abs(), exp_bias=-2)
    x2 = x * x
    series = x * (1.0 + x2 * (1 / 6 + x2 * (1 / 120 + x2 * (1 / 5040 + x2 / 362880))))
    large = torch.sign(x) * ((quarter_e - 0.0625 / quarter_e) * 2.0)
    computed = torch.where(x.abs() < _SINH_TAYLOR_LIMIT, series, large)
    return torch.where(torch.isfinite(x), computed, torch.sinh(x))


@register_aten(
    ["aten::tan", "aten::tan.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_tan(x, out=None):
    """Compute the tangent after Cody-Waite argument reduction.

    Args:
        x (torch.Tensor): Input tensor, promoted to the default float dtype if needed.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise tangent values in the promoted input dtype.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        ``torch.float16`` and ``torch.bfloat16`` inputs are reduced modulo pi in
        ``torch.float32`` using a three-part pi split and are cast back afterward;
        ``torch.float32`` and ``torch.float64`` inputs retain their dtype.
    """
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    orig_dtype = x.dtype
    xr = x.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else x
    n = torch.round(xr / math.pi)
    r = ((xr - n * _PI_HI) - n * _PI_MID) - n * _PI_LO
    return torch.tan(r).to(orig_dtype)


@register_aten(
    ["aten::acos", "aten::acos.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_acos(x, out=None):
    """Compute the inverse cosine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse cosine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.acos(x)


@register_aten(
    ["aten::acosh", "aten::acosh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_acosh(x, out=None):
    """Compute the inverse hyperbolic cosine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse hyperbolic cosine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.acosh(x)


@register_aten(
    ["aten::asin", "aten::asin.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_asin(x, out=None):
    """Compute the inverse sine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse sine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.asin(x)


@register_aten(
    ["aten::asinh", "aten::asinh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_asinh(x, out=None):
    """Compute the inverse hyperbolic sine of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse hyperbolic sine values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.asinh(x)


@register_aten(
    ["aten::atan", "aten::atan.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_atan(x, out=None):
    """Compute the inverse tangent of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse tangent values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.atan(x)


@register_aten(
    ["aten::atanh", "aten::atanh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_atanh(x, out=None):
    """Compute the inverse hyperbolic tangent of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise inverse hyperbolic tangent values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.atanh(x)


@register_aten(
    ["aten::exp", "aten::exp.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_exp(x, out=None):
    """Compute the exponential of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise exponential values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.exp(x)


def _expm1_accurate(x):
    """exp(x) - 1 over the accurate reduced exp; Taylor near zero.

    The hardware exp has ~1e-05 relative error, so exp(x) - 1 misses the fp32
    tolerance. Use the range-reduced exp instead (see _exp_reduced, ~7e-08).
    Near zero exp(x) - 1 cancels catastrophically, so small inputs use the
    Taylor series expm1(x) = x + x**2/2! + x**3/3! + ... instead.

    This helper calls no torch.expm1, so the torch.compile decomposition that
    reuses it (expm1_decomp in decompositions.py) does not recurse. Both paths
    share this one implementation. Compute fp16/bf16 in float32 for accuracy.
    """
    # Integer and bool inputs promote to the default float dtype, matching CPU.
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    large = _exp_reduced(x) - 1.0
    series = x * (
        1.0 + x * (0.5 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x * (1 / 720 + x / 5040)))))
    )
    computed = torch.where(x.abs() < _EXPM1_TAYLOR_LIMIT, series, large)
    # Large negative x: exp(x) is 0 to float precision, so expm1(x) = -1. Saturate
    # before the range reduction reaches it: for very negative x, 2**k underflows
    # to 0 while the corrupted reduced argument makes exp(r) non-finite, so
    # _exp_reduced returns 0 * inf = nan. -40 is far above that (exp(-40)~4e-18,
    # negligible for every float dtype) and far below the underflow region.
    computed = torch.where(x < _EXPM1_NEG_SATURATE, torch.full_like(x, -1.0), computed)
    # Non-finite inputs: expm1(+inf)=+inf, expm1(-inf)=-1, expm1(nan)=nan. A
    # finite overflow already yields +inf through 2**k, and nan propagates.
    computed = torch.where(x == float("-inf"), torch.full_like(x, -1.0), computed)
    computed = torch.where(x == float("inf"), torch.full_like(x, float("inf")), computed)
    # A nan input must stay nan. On Neuron the large-negative `<` compare above can
    # read a nan as true, so the saturate can turn nan into -1. Restore nan last.
    computed = torch.where(torch.isnan(x), x, computed)
    if orig_dtype in (torch.float16, torch.bfloat16):
        computed = computed.to(orig_dtype)
    return computed


@register_aten(
    ["aten::expm1", "aten::expm1.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_expm1(x, out=None):
    """Compute one less than the exponential of each element.

    Args:
        x (torch.Tensor): Input tensor, promoted to the default float dtype if needed.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise values of ``exp(x) - 1``.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Uses the shared range-reduced decomposition, a Taylor series near zero, and negative
        saturation. Float16 and bfloat16 inputs are computed in float32 and cast back.
    """
    # Call the aten op so make_fx reuses the registered expm1 decomposition
    # (expm1_decomp -> _expm1_accurate); both paths share the one helper.
    return torch.ops.aten.expm1(x)


def _validate_expm1_inplace(self):
    """Reject an integer or bool in-place expm1_.

    expm1 promotes an integer input to float, but the in-place variant must
    store the result back into self. The Neuron dispatch path skips PyTorch's
    structured-kernel guard, so check here. CPU raises the same error.
    """
    if not self.is_floating_point():
        expected = torch.get_default_dtype()
        raise RuntimeError(
            f"result type {expected} can't be cast to the desired output type {self.dtype}"
        )


@register_aten(["aten::expm1_"], validate_static_inputs=_validate_expm1_inplace)
def torch_expm1_(self):
    """Compute the registered in-place exponential-minus-one operation.

    Args:
        self (torch.Tensor): Floating-point tensor updated by the registered operation.

    Returns:
        torch.Tensor: The mutated ``self`` tensor.

    Raises:
        RuntimeError: If ``self`` has a Boolean or integer dtype.

    Note:
        Uses the same range-reduced implementation as ``torch_expm1``.
    """
    # Same as torch_expm1: call the aten op so make_fx reuses the registered
    # expm1 decomposition. The in-place path writes the result back into self.
    return torch.ops.aten.expm1(self)


@register_aten(
    ["aten::log", "aten::log.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_log(x, out=None):
    """Compute the natural logarithm of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise natural logarithms.
    """
    return torch.log(x)


@register_aten(["aten::log2", "aten::log2.out", "aten::log2_"])
def torch_log2(x, out=None):
    """Compute the base-2 logarithm of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise base-2 logarithms.

    Note:
        The registered ``log2_`` variant updates ``x`` in place.
    """
    return torch.log2(x)


@register_aten(["aten::log10", "aten::log10.out", "aten::log10_"])
def torch_log10(x, out=None):
    """Compute the base-10 logarithm of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise base-10 logarithms.

    Note:
        The registered ``log10_`` variant updates ``x`` in place.
    """
    return torch.log10(x)


@register_aten(["aten::log1p", "aten::log1p.out", "aten::log1p_"])
def torch_log1p(x, out=None):
    """Compute the natural logarithm of one plus each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise values of ``log(1 + x)``; integer and Boolean inputs
            use the default floating-point dtype.

    Note:
        Computes ``torch.float16`` and ``torch.bfloat16`` inputs in
        ``torch.float32`` before casting back to match CPU semantics. The registered
        ``log1p_`` variant updates ``x`` in place.
    """
    # Upcast to float32 for reduced-precision types to match CPU semantics.
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.log1p(x.float()).to(x.dtype)
    return torch.log1p(x)


@register_aten(
    ["aten::erf", "aten::erf.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_erf(x, out=None):
    # Cast non-floating-point inputs (e.g. bool, int) to float,
    # matching PyTorch CPU behavior where erf promotes to float.
    """Compute the error function of each element.

    Args:
        x (torch.Tensor): Input tensor, promoted to ``torch.float32`` if non-floating-point.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise error-function values with NaNs preserved.
    """
    if not x.is_floating_point():
        x = x.to(torch.float32)
    # Handle NaN case: erf(NaN) should return NaN
    result = torch.erf(x)
    return torch.where(torch.isnan(x), x, result)


@register_aten(
    ["aten::erfinv", "aten::erfinv.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_erfinv(y, out=None):
    # Cast non-floating-point inputs (e.g. bool, int) to float,
    # matching PyTorch CPU behavior where erfinv promotes to float.
    """Approximate the inverse error function elementwise.

    Args:
        y (torch.Tensor): Input tensor, promoted to ``torch.float32`` if non-floating-point.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Piecewise rational approximations, with NaN outside
        ``[-1, 1]`` and signed infinity at the endpoints.
    """
    if not y.is_floating_point():
        y = y.to(torch.float32)

    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]

    y_abs = torch.abs(y)
    result = torch.where(
        torch.isnan(y) | (y_abs > 1.0), y * float("nan"), torch.sign(y) * float("inf")
    )

    mask_valid = y_abs < 1.0
    z_low = y * y
    num_low = ((a[3] * z_low + a[2]) * z_low + a[1]) * z_low + a[0]
    dem_low = (((b[3] * z_low + b[2]) * z_low + b[1]) * z_low + b[0]) * z_low + 1.0
    x_low = y * num_low / dem_low

    z_high = torch.sqrt(-torch.log((1.0 - y_abs) / 2.0))
    num_high = ((c[3] * z_high + c[2]) * z_high + c[1]) * z_high + c[0]
    dem_high = (d[1] * z_high + d[0]) * z_high + 1.0
    x_high = torch.sign(y) * num_high / dem_high

    result = torch.where(mask_valid & (y_abs <= 0.7), x_low, result)
    result = torch.where(mask_valid & (y_abs > 0.7), x_high, result)

    return result


@register_aten(
    ["aten::ceil", "aten::ceil.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_ceil(self, out=None):
    """Compute the ceiling of each element.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise ceiling values.

    Raises:
        RuntimeError: If ``self`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.ceil(self)


def _validate_floor(self, out=None):
    """Reject an out= tensor whose dtype doesn't match the input.

    Mirrors PyTorch's structured-kernel unsafe-cast guard, which the Neuron
    dispatch path skips (it would otherwise silently cast the result into out).
    floor preserves the input dtype (unlike acos et al., which promote to float),
    so out must have the same dtype as the input. The message is spelled like
    CPU (e.g. 'Long', not 'torch.int64').
    """
    if isinstance(self, torch.Tensor) and self.dtype is torch.bool:
        raise RuntimeError("\"floor\" not implemented for 'Bool'")
    if out is not None and out.dtype != self.dtype:
        raise RuntimeError(
            f"Found dtype {aten_scalar_type_name(out.dtype)} but expected "
            f"{aten_scalar_type_name(self.dtype)}"
        )


@register_aten(
    ["aten::floor", "aten::floor.out", "aten::floor_"],
    validate_static_inputs=_validate_floor,
)
def torch_floor(self, out=None):
    """Compute the floor of each element.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise floors, or ``self`` itself for supported integer data.

    Raises:
        RuntimeError: If ``self`` is Boolean or ``out`` has a different dtype from
            ``self``.

    Note:
        Passes through every value with ``abs(x) >= 2**23`` to avoid StableHLO
        clamping on the ``torch.float32`` path. Restores NaNs because neuronx-cc floor otherwise
        returns ``2**31``; infinities are handled by the large-value branch. The
        registered ``floor_`` variant updates ``self`` in place.
    """
    if not self.is_floating_point():
        return self
    result = torch.floor(self)
    large = torch.abs(self) >= 8388608.0  # 2^23
    result = torch.where(large, self, result)
    return torch.where(torch.isnan(self), self, result)


@register_aten(
    ["aten::trunc", "aten::trunc.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_trunc(x, out=None):
    """Truncate each element toward zero.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise truncated values.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.trunc(x)


@register_aten(
    ["aten::sign", "aten::sign.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sign(x, out=None):
    """Compute the sign of each element.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise signs.
    """
    return torch.sign(x)


@register_aten(
    ["aten::angle", "aten::angle.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_angle(x, out=None):
    """Compute the phase angle of each real input element.

    Args:
        x (torch.Tensor): Real input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise phase angles; real inputs yield zero for nonnegative values,
            pi for negative values, and NaN for NaN inputs.

    Raises:
        RuntimeError: If ``x`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Torch-MLIR lacks StableHLO legalization for ``aten::angle``. The shared
        ``angle_decomp`` decomposition handles eager ``make_fx`` and
        ``torch.compile`` paths before this handler lowers ``torch.angle``.
    """
    return torch.angle(x)


@register_aten(
    ["aten::reciprocal", "aten::reciprocal_", "aten::reciprocal.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_reciprocal(a, out=None):
    """Compute the reciprocal of each element.

    Args:
        a (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise reciprocals.

    Raises:
        RuntimeError: If ``a`` is floating-point and ``out`` has a non-floating-point dtype.

    Note:
        Promotes integer and Boolean inputs to the default float dtype because the
        Torch-MLIR StableHLO lowering accepts only floating-point reciprocal
        operands. This also supports integer ``__rdiv__`` and ``__rtruediv__``.
        The registered ``reciprocal_`` variant updates ``a`` in place.
    """
    if not a.is_floating_point() and not a.is_complex():
        a = a.to(torch.get_default_dtype())
    return torch.reciprocal(a)


@register_aten(
    ["aten::exp2", "aten::exp2.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_exp2(self, out=None):
    """Compute two raised to each input element.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise base-2 exponentials.

    Raises:
        RuntimeError: If ``self`` is floating-point and ``out`` has a non-floating-point dtype.
    """
    return torch.ops.aten.exp2(self)


@register_aten(["aten::exp2_"])
def torch_exp2_(self):
    """Compute the registered in-place base-2 exponential.

    Args:
        self (torch.Tensor): Input tensor updated by the registered operation.

    Returns:
        torch.Tensor: The mutated ``self`` tensor.
    """
    return torch.ops.aten.exp2(self)


@register_aten(
    ["aten::frac", "aten::frac.out", "aten::frac_"],
    validate_static_inputs=lambda *args, **kw: (
        validate_float_only_dtype(*args, **kw),
        _validate_float_promoting_out(*args, **kw),
    ),
)
def torch_frac(self, out=None):
    """Compute the fractional portion of each element.

    Args:
        self (torch.Tensor): Floating-point input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Fractional portions of the input values.

    Raises:
        RuntimeError: If ``self`` has a Boolean or integer dtype, or floating-point
            ``self`` is paired with a non-floating-point ``out``.

    Note:
        The registered ``frac_`` variant updates ``self`` in place.
    """
    return torch.ops.aten.frac(self)


@register_aten(
    ["aten::round", "aten::round.out", "aten::round_"],
    validate_static_inputs=lambda *args, **kw: (
        validate_round_dtype(*args, **kw),
        _validate_float_promoting_out(*args, **kw),
    ),
)
def torch_round(self, out=None):
    """Round each element to the nearest integer value.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise rounded values.

    Raises:
        RuntimeError: If ``self`` has Boolean dtype, or floating-point ``self`` is
            paired with a non-floating-point ``out``.

    Note:
        The registered ``round_`` variant updates ``self`` in place.
    """
    return torch.ops.aten.round(self)


@register_aten(
    ["aten::round.decimals", "aten::round.decimals_out", "aten::round_.decimals"],
    static_argnames=("decimals",),
    validate_static_inputs=lambda *args, **kw: (
        validate_float_only_dtype(*args, **kw),
        _validate_float_promoting_out(*args, **kw),
    ),
)
def torch_round_decimals(self, decimals, out=None):
    """Round each element to a fixed number of decimal places.

    Args:
        self (torch.Tensor): Floating-point input tensor.
        decimals (int): Number of decimal places to retain.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise rounded values.

    Raises:
        RuntimeError: If ``self`` has a Boolean or integer dtype, or floating-point
            ``self`` is paired with a non-floating-point ``out``.

    Note:
        The registered ``round_.decimals`` variant updates ``self`` in place.
    """
    return torch.ops.aten.round(self, decimals=decimals)


@register_aten(
    ["aten::logit", "aten::logit.out"],
    static_argnames=("eps",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_logit(self, eps=None, out=None):
    """Compute the logit of each element.

    Args:
        self (torch.Tensor): Input tensor.
        eps (float, optional): Clamping epsilon. Defaults to ``None``.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise logit values.
    """
    return torch.ops.aten.logit(self, eps)


@register_aten(["aten::logit_"], static_argnames=("eps",))
def torch_logit_(self, eps=None):
    """Compute the registered in-place logit.

    Args:
        self (torch.Tensor): Input tensor updated by the registered operation.
        eps (float, optional): Clamping epsilon. Defaults to ``None``.

    Returns:
        torch.Tensor: The mutated ``self`` tensor.
    """
    return torch.ops.aten.logit(self, eps)


@register_aten(
    ["aten::logit_backward", "aten::logit_backward.grad_input"],
    output_params=("grad_input",),
    static_argnames=("eps",),
)
def torch_logit_backward(grad_output, self, eps=None, grad_input=None):
    """Compute the gradient of the logit operation.

    Args:
        grad_output (torch.Tensor): Upstream gradient tensor.
        self (torch.Tensor): Forward input tensor.
        eps (float, optional): Forward clamping epsilon. Defaults to ``None``.
        grad_input (torch.Tensor, optional): Gradient destination. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.
    """
    return torch.ops.aten.logit_backward.default(grad_output, self, eps)


@register_aten(
    ["aten::special_i1e", "aten::special_i1e.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_special_i1e(self, out=None):
    """Compute the exponentially scaled modified Bessel function I1.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Destination tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise exponentially scaled modified Bessel values.
    """
    return torch.ops.aten.special_i1e(self)
