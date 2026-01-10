"""Reduction operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::sum", "aten::sum.dim_IntList", "aten::sum.IntList_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
)
def torch_sum(self, dim=None, keepdim=False, dtype=None):
    return torch.sum(self, dim, keepdim, dtype=dtype)


@register_aten(
    ["aten::mean", "aten::mean.dim", "aten::mean.out", "aten::mean.dtype_out"],
    static_argnums=(1, 2),
)
def torch_mean(x, dim=None, keepdim=False, dtype=None):
    if dim is None:
        return torch.mean(x, dtype=dtype)
    return torch.mean(x, dim, keepdim, dtype=dtype)


@register_aten(["aten::max", "aten::max.unary_out", "aten::max.dim_max"], static_argnums=(1, 2))
def torch_max(x, dim=None, keepdim=False):
    if dim is None:
        return torch.max(x)
    return torch.max(x, dim, keepdim)


@register_aten(["aten::amax", "aten::amax.out"], static_argnums=(1, 2))
def torch_amax(x, dim=None, keepdim=False):
    return torch.amax(x, dim, keepdim)


@register_aten(
    ["aten::amin", "aten::amin.out"],
    static_argnums=(1, 2),
)
def torch_amin(self, dim=None, keepdim=False):
    return torch.amin(self, dim, keepdim)


@register_aten(["aten::all", "aten::all.out", "aten::all.all_out"], static_argnums=(1, 2))
def torch_all(x, dim=None, keepdim=False, out=None):
    if dim is None:
        return torch.all(x)
    return torch.all(x, dim, keepdim)


@register_aten(["aten::any", "aten::any.out", "aten::any.all_out"], static_argnums=(1, 2))
def torch_any(x, dim=None, keepdim=False, out=None):
    if dim is None:
        return torch.any(x).to(torch.bool)
    return torch.any(x, dim, keepdim).to(torch.bool)


@register_aten(
    ["aten::cumsum", "aten::cumsum.out"], static_argnums=(1,), static_argnames=("dtype",)
)
def torch_cumsum(x, dim, dtype=None):
    if x.ndim == 0:
        return x
    return torch.cumsum(x, dim, dtype=dtype)
