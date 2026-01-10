"""Comparison operations."""

import torch

from ..operation_registry import register_aten
from .constants import INT_DTYPES


@register_aten(["aten::eq", "aten::eq.Scalar_out", "aten::eq.Tensor_out"])
def torch_eq(x, y, out=None):
    return torch.eq(x, y)


@register_aten(["aten::ne", "aten::ne.Scalar_out", "aten::ne.Tensor_out"])
def torch_ne(x, y, out=None):
    return torch.ne(x, y)


@register_aten(["aten::gt", "aten::gt.Scalar_out", "aten::gt.Tensor_out"])
def torch_gt(x, y, out=None):
    return torch.gt(x, y)


@register_aten(["aten::ge", "aten::ge.Scalar_out", "aten::ge.Tensor_out"])
def torch_ge(x, y, out=None):
    return torch.ge(x, y)


@register_aten(["aten::lt", "aten::lt.Scalar_out", "aten::lt.Tensor_out"])
def torch_lt(x, y, out=None):
    return torch.lt(x, y)


@register_aten(["aten::le", "aten::le.Tensor_out", "aten::le.Scalar_out"])
def torch_le(x, y, out=None):
    return torch.le(x, y, out=out) if out is not None else torch.le(x, y)


@register_aten(
    ["aten::clamp", "aten::clamp.out", "aten::clamp.Tensor", "aten::clamp.Tensor_out"],
)
def torch_clamp(x, min=None, max=None):
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    return torch.clamp(x, min=min, max=max)


@register_aten(
    [
        "aten::clamp_max",
        "aten::clamp_max.out",
        "aten::clamp_max.Tensor",
        "aten::clamp_max.Tensor_out",
    ],
)
def torch_clamp_max(x, max):
    return torch.clamp_max(x, max)


@register_aten(
    [
        "aten::clamp_min",
        "aten::clamp_min.out",
        "aten::clamp_min.Tensor",
        "aten::clamp_min.Tensor_out",
    ],
)
def torch_clamp_min(x, min):
    return torch.clamp_min(x, min)


@register_aten(["aten::isnan"])
def torch_isnan(x):
    return torch.isnan(x)


@register_aten(["aten::isinf"])
def torch_isinf(x):
    return torch.isinf(x)


@register_aten(["aten::isneginf", "aten::isneginf.out"])
def torch_isneginf(x):
    return torch.isneginf(x)


@register_aten(["aten::isfinite"])
def torch_isfinite(x):
    if x.dtype in INT_DTYPES:
        # Integers are always finite
        return torch.ones_like(x, dtype=torch.bool)
    else:
        return torch.isfinite(x)
