"""Unary operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::abs", "aten::abs.out"])
def torch_abs(x, out=None):
    """StableHlo.abs does not support unsigned dtype

    Reference: https://openxla.org/stablehlo/spec#abs_uint8_decomp
    """
    if x.dtype == torch.uint8:
        return x
    return torch.abs(x)


@register_aten(["aten::neg", "aten::neg.out"])
def torch_neg(x, out=None):
    return torch.neg(x)


@register_aten(["aten::sqrt", "aten::sqrt.out"])
def torch_sqrt(x, out=None):
    """Square root."""
    return torch.sqrt(x)


@register_aten(["aten::rsqrt", "aten::rsqrt.out"])
def torch_rsqrt(x, out=None):
    return torch.rsqrt(x)


@register_aten(["aten::cos", "aten::cos.out"])
def torch_cos(x, out=None):
    return torch.cos(x)


@register_aten(["aten::sin", "aten::sin.out"])
def torch_sin(x, out=None):
    return torch.sin(x)


@register_aten(["aten::exp", "aten::exp.out"])
def torch_exp(x, out=None):
    return torch.exp(x)


@register_aten(["aten::log", "aten::log.out"])
def torch_log(x, out=None):
    return torch.log(x)


@register_aten(["aten::erf", "aten::erf.out"])
def torch_erf(x, out=None):
    # Handle NaN case: erf(NaN) should return NaN
    result = torch.erf(x)
    return torch.where(torch.isnan(x), x, result)


@register_aten(["aten::erfinv", "aten::erfinv.out"])
def torch_erfinv(y, out=None):
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


@register_aten(["aten::ceil", "aten::ceil.out"])
def torch_ceil(self, out=None):
    return torch.ceil(self)


@register_aten(["aten::trunc", "aten::trunc.out"])
def torch_trunc(x, out=None):
    return torch.trunc(x)


@register_aten(["aten::sign", "aten::sign.out"])
def torch_sign(x, out=None):
    return torch.sign(x)


@register_aten(["aten::reciprocal", "aten::reciprocal_", "aten::reciprocal.out"])
def torch_reciprocal(a, out=None):
    return torch.reciprocal(a)
