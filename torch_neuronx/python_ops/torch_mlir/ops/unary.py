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
    """Reject an out= tensor that cannot hold a floating-point result.

    Raises only when the input is a floating-point type AND the out= tensor is
    a non-floating dtype.

    Accepts *args/**kwargs so it can serve as validate_static_inputs for any
    op regardless of how many positional or keyword arguments it has.
    For ops that return (values, indices), out is a tuple — only the values
    tensor is checked.
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
    """StableHlo.abs does not support unsigned dtype

    Reference: https://openxla.org/stablehlo/spec#abs_uint8_decomp
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
    return torch.neg(x)


@register_aten(
    ["aten::sqrt", "aten::sqrt.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sqrt(x, out=None):
    """Square root."""
    return torch.sqrt(x)


@register_aten(
    ["aten::rsqrt", "aten::rsqrt.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_rsqrt(x, out=None):
    return torch.rsqrt(x)


@register_aten(
    ["aten::cos", "aten::cos.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_cos(x, out=None):
    return torch.cos(x)


@register_aten(
    ["aten::sin", "aten::sin.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sin(x, out=None):
    return torch.sin(x)


@register_aten(
    ["aten::cosh", "aten::cosh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_cosh(x, out=None):
    """Hyperbolic cosine as (e**x + e**-x)/2 over the reduced exp.

    Measured max rel error 1.1e-07 in fp32 vs 9.3e-06 for the hardware cosh.
    fp16/bf16 keep the hardware op: the polynomial's multiply-adds accumulate
    more error in an 8-11 bit mantissa (1.5e-02 bf16) than the single op does.
    Non-finite inputs also defer to it, since the reduction turns -inf into nan.
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
    """Hyperbolic sine: Taylor near 0, else (e**x - e**-x)/2 over the reduced exp.

    The exp form cancels catastrophically as x approaches 0 (4.6e-02 at 1e-6),
    so small inputs use the series instead. Measured max rel error 1.0e-07 in
    fp32 vs 1.7e-05 for the hardware sinh. fp16/bf16 and non-finite inputs keep
    the hardware op, for the reasons in torch_cosh.
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
    """Tangent with the argument reduced into (-pi/2, pi/2) first.

    The hardware reduces x mod pi in single precision, so error grows with |x|:
    8.1e-02 over [-pi, pi] and 4.2e-01 over [1e3, 1e5]. Reducing first keeps it
    at 3.3e-06 / 4.6e-04. The reduction runs in fp32 (upcasting fp16/bf16) so
    the split-pi subtraction stays exact; casting back matches torch's own fp16
    tan on every measured sample.
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
    """Inverse cosine (arccos) — elementwise transcendental."""
    return torch.acos(x)


@register_aten(
    ["aten::acosh", "aten::acosh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_acosh(x, out=None):
    """Inverse hyperbolic cosine — elementwise transcendental."""
    return torch.acosh(x)


@register_aten(
    ["aten::asin", "aten::asin.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_asin(x, out=None):
    """Inverse sine (arcsin) — elementwise transcendental."""
    return torch.asin(x)


@register_aten(
    ["aten::asinh", "aten::asinh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_asinh(x, out=None):
    """Inverse hyperbolic sine — elementwise transcendental."""
    return torch.asinh(x)


@register_aten(
    ["aten::atan", "aten::atan.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_atan(x, out=None):
    """Inverse tangent (arctan) — elementwise transcendental."""
    return torch.atan(x)


@register_aten(
    ["aten::atanh", "aten::atanh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_atanh(x, out=None):
    """Inverse hyperbolic tangent — elementwise transcendental."""
    return torch.atanh(x)


@register_aten(
    ["aten::exp", "aten::exp.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_exp(x, out=None):
    return torch.exp(x)


@register_aten(
    ["aten::log", "aten::log.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_log(x, out=None):
    return torch.log(x)


@register_aten(["aten::log2", "aten::log2.out", "aten::log2_"])
def torch_log2(x, out=None):
    """Base-2 logarithm — elementwise."""
    return torch.log2(x)


@register_aten(["aten::log10", "aten::log10.out", "aten::log10_"])
def torch_log10(x, out=None):
    """Base-10 logarithm — elementwise."""
    return torch.log10(x)


@register_aten(["aten::log1p", "aten::log1p.out", "aten::log1p_"])
def torch_log1p(x, out=None):
    """Natural logarithm of (1 + x) — elementwise."""
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
    return torch.ceil(self)


def _validate_floor(self, out=None):
    """Reject bool inputs and mismatched out= dtype for floor.

    bool is rejected because floor is not defined for bool dtype. The out=
    dtype must match the input (floor preserves the input dtype, unlike
    float-promoting ops).
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
    """Floor — elementwise, identity on non-floating-point dtypes.

    Two input classes bypass the HLO floor and pass through unchanged (floor is
    the identity on both):
      - |x| >= 2^23: float32 has no fractional bits, so the value is already an
        integer; the HLO lowering otherwise loses precision (clamps to int32).
      - nan: neuronx-cc's floor returns 2^31 for nan instead of nan. floor(nan)
        is nan, so we restore it with an isnan mask. (inf/-inf are already handled
        correctly by the 2^23 branch, since |inf| >= 2^23.)
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
    return torch.trunc(x)


@register_aten(
    ["aten::sign", "aten::sign.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sign(x, out=None):
    return torch.sign(x)


@register_aten(
    ["aten::angle", "aten::angle.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_angle(x, out=None):
    """Phase angle of a real input.

    torch-mlir has no StableHLO legalization for aten::angle, so it is decomposed
    into where/isnan (0 for x >= 0, pi for x < 0, nan for nan) by
    ``angle_decomp`` in neuron_dynamo_backend/decompositions.py. That decomposition
    table is applied on both the eager (make_fx) and torch.compile paths, so this
    handler just calls torch.angle and lets the shared decomposition lower it.
    """
    return torch.angle(x)


@register_aten(
    ["aten::reciprocal", "aten::reciprocal_", "aten::reciprocal.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_reciprocal(a, out=None):
    """Elementwise reciprocal (1/x).

    reciprocal promotes integer/bool inputs to the default floating-point dtype
    (matching PyTorch's type-promotion semantics). The torch-mlir StableHLO
    lowering of ``AtenReciprocalOp`` only legalizes floating-point operands
    ("only floating-point datatype legalization supported for AtenReciprocalOp"),
    so cast integer/bool inputs up front instead of emitting an illegal integer
    reciprocal. This is what makes integer ``__rdiv__`` / ``__rtruediv__``
    (``1 / int_tensor``) lower correctly.
    """
    if not a.is_floating_point() and not a.is_complex():
        a = a.to(torch.get_default_dtype())
    return torch.reciprocal(a)


@register_aten(
    ["aten::exp2", "aten::exp2.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_exp2(self, out=None):
    return torch.ops.aten.exp2(self)


@register_aten(["aten::exp2_"])
def torch_exp2_(self):
    return torch.ops.aten.exp2(self)


@register_aten(
    ["aten::frac", "aten::frac.out", "aten::frac_"],
    validate_static_inputs=lambda *args, **kw: (
        validate_float_only_dtype(*args, **kw),
        _validate_float_promoting_out(*args, **kw),
    ),
)
def torch_frac(self, out=None):
    return torch.ops.aten.frac(self)


@register_aten(
    ["aten::round", "aten::round.out", "aten::round_"],
    validate_static_inputs=lambda *args, **kw: (
        validate_round_dtype(*args, **kw),
        _validate_float_promoting_out(*args, **kw),
    ),
)
def torch_round(self, out=None):
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
    return torch.ops.aten.round(self, decimals=decimals)


@register_aten(
    ["aten::logit", "aten::logit.out"],
    static_argnames=("eps",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_logit(self, eps=None, out=None):
    return torch.ops.aten.logit(self, eps)


@register_aten(["aten::logit_"], static_argnames=("eps",))
def torch_logit_(self, eps=None):
    return torch.ops.aten.logit(self, eps)


@register_aten(
    ["aten::logit_backward", "aten::logit_backward.grad_input"],
    output_params=("grad_input",),
    static_argnames=("eps",),
)
def torch_logit_backward(grad_output, self, eps=None, grad_input=None):
    return torch.ops.aten.logit_backward.default(grad_output, self, eps)


@register_aten(
    ["aten::special_i1e", "aten::special_i1e.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_special_i1e(self, out=None):
    return torch.ops.aten.special_i1e(self)
