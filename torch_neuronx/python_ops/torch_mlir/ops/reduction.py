"""Reduction operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::sum", "aten::sum.dim_IntList", "aten::sum.IntList_out"],
    static_argnames=("dim", "keepdim", "dtype"),
)
def torch_sum(self, dim=None, keepdim=False, dtype=None):
    return torch.sum(self, dim, keepdim, dtype=dtype)


@register_aten(["aten::mean", "aten::mean.dim", "aten::mean.out", "aten::mean.dtype_out"])
def torch_mean(x, dim=None, keepdim=False, dtype=None):
    if dim is None:
        return torch.mean(x, dtype=dtype)
    return torch.mean(x, dim, keepdim, dtype=dtype)


@register_aten(["aten::max", "aten::max.unary_out", "aten::max.dim_max"])
def torch_max(x, dim=None, keepdim=False):
    if dim is None:
        return torch.max(x)
    return torch.max(x, dim, keepdim)


@register_aten(["aten::amax", "aten::amax.out"], static_argnames=("dim", "keepdim"))
def torch_amax(x, dim=None, keepdim=False):
    return torch.amax(x, dim, keepdim)


@register_aten(
    ["aten::amin", "aten::amin.out"],
    static_argnames=(
        "dim",
        "keepdim",
    ),
)
def torch_amin(self, dim=None, keepdim=False):
    return torch.amin(self, dim, keepdim)


@register_aten(
    ["aten::all", "aten::all.out", "aten::all.all_out"], static_argnames=("dim", "keepdim")
)
def torch_all(x, dim=None, keepdim=False):
    if dim is None:
        return torch.all(x)
    return torch.all(x, dim, keepdim)


@register_aten(
    ["aten::any", "aten::any.out", "aten::any.all_out"], static_argnames=("dim", "keepdim")
)
def torch_any(x, dim=None, keepdim=False):
    if dim is None:
        return torch.any(x)
    return torch.any(x, dim, keepdim)


@register_aten(["aten::cumsum", "aten::cumsum.out"], static_argnames=("dtype",))
def torch_cumsum(x, dim, dtype=None):
    return torch.cumsum(x, dim, dtype=dtype)
