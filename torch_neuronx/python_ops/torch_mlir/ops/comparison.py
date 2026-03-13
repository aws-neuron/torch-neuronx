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
    return torch.le(x, y)


@register_aten(
    ["aten::clamp", "aten::clamp.out", "aten::clamp.Tensor", "aten::clamp.Tensor_out"],
)
def torch_clamp(x, min=None, max=None, out=None):
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")

    # Ensure tensor min/max have same dtype as input
    if min is not None and isinstance(min, torch.Tensor):
        min = min.to(x.dtype)
    if max is not None and isinstance(max, torch.Tensor):
        max = max.to(x.dtype)

    return torch.clamp(x, min=min, max=max)


@register_aten(
    [
        "aten::clamp_max",
        "aten::clamp_max.out",
        "aten::clamp_max.Tensor",
        "aten::clamp_max.Tensor_out",
    ],
)
def torch_clamp_max(x, max, out=None):
    # Ensure tensor max has same dtype as input
    if isinstance(max, torch.Tensor):
        max = max.to(x.dtype)
    return torch.clamp_max(x, max)


@register_aten(
    [
        "aten::clamp_min",
        "aten::clamp_min.out",
        "aten::clamp_min.Tensor",
        "aten::clamp_min.Tensor_out",
    ],
)
def torch_clamp_min(x, min, out=None):
    # Ensure tensor min has same dtype as input
    if isinstance(min, torch.Tensor):
        min = min.to(x.dtype)
    return torch.clamp_min(x, min)


@register_aten(["aten::isnan"])
def torch_isnan(x):
    return torch.isnan(x)


@register_aten(["aten::isinf"])
def torch_isinf(x):
    """Integer dtype cannot be inf and should always return False.

    Bypassing the operation here to avoid running stablehlo.abs with unsupported dtypes.
    """
    if x.dtype.is_floating_point or x.dtype.is_complex:
        return torch.isinf(x)
    else:
        return torch.zeros_like(x, dtype=torch.bool)


@register_aten(["aten::isneginf", "aten::isneginf.out"])
def torch_isneginf(x, out=None):
    return torch.isneginf(x)


@register_aten(["aten::isfinite"])
def torch_isfinite(x):
    """
    torch.isfinite is getting decomposed before MLIR in fx graph,
    therefore we wont be able to get stablehlo.isfinite in the HLO
    """
    if x.dtype in INT_DTYPES:
        # Integers are always finite
        return torch.ones_like(x, dtype=torch.bool)
    else:
        # For floating point: finite = not (nan or inf)
        return ~(torch.isnan(x) | torch.isinf(x))
