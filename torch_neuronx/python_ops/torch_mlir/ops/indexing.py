"""Indexing, slicing, and selection operations."""

import torch

from ..operation_registry import register_aten


# @register_aten(["aten::index.Tensor", "aten::index.Tensor_out"])
def torch_index(x, indices):
    return x[indices]


@register_aten(["aten::index_select", "aten::index_select.out"], static_argnums=(1,))
def torch_index_select(x, dim, index):
    return torch.index_select(x, dim, index)


@register_aten(["aten::index_copy", "aten::index_copy_"], static_argnums=(1,))
def torch_index_copy(x, dim, indexes, source):
    return torch.index_copy(x, dim, indexes, source)


# @register_aten(["aten::index_add", "aten::index_add.out"])
def torch_index_add(x, dim, index, source, alpha=1):
    return torch.index_add(x, dim, index, source, alpha=alpha)


@register_aten(["aten::gather", "aten::gather.out"])
def torch_gather(x, dim, index):
    return torch.gather(x, dim, index)


@register_aten(["aten::scatter", "aten::scatter.value_out"])
def torch_scatter_value(x, dim, index, value):
    return torch.scatter(x, dim, index, value)


@register_aten(
    ["aten::scatter.src", "aten::scatter.src_out", "aten::scatter.reduce"],
    static_argnums=(1,),
    static_argnames=("reduce",),
)
def torch_scatter_src(input, dim, index, src, reduce=None):
    if reduce is None:
        return torch.scatter(input, dim, index, src)
    torch.scatter(input, dim, index, src, reduce=reduce)


# (["aten::masked_select", "aten::masked_select.out", "aten::masked_select.Tensor_out"])
def torch_masked_select(x, mask):
    return torch.masked_select(x, mask)


@register_aten(["aten::masked_fill", "aten::masked_fill_.Scalar"])
def torch_masked_fill(x, mask, value):
    return torch.masked_fill(x, mask, value)


@register_aten(
    [
        "aten::where",
        "aten::where.self",
        "aten::where.self_out",
        "aten::where.ScalarSelf",
        "aten::where.ScalarOther",
        "aten::where.Scalar",
    ],
    use_torch_compile=True,
)
def torch_where(condition, x=None, y=None, out=None):
    if x is None:
        return torch.where(condition)
    return torch.where(condition, x, y)


@register_aten(["aten::nonzero", "aten::nonzero.out"])
def torch_nonzero(x):
    return torch.nonzero(x)


@register_aten(["aten::nonzero_static", "aten::nonzero_static.out"])
def torch_nonzero_static(x, *, size, fill_value=-1):
    return torch.ops.aten.nonzero_static(x, size=size, fill_value=fill_value)


# @register_aten(["aten::topk", "aten::topk.values"])
def torch_topk(x, k, dim=-1, largest=True, sorted=True):
    return torch.topk(x, k, dim, largest, sorted)


# @register_aten(["aten::_index_put_impl_"], static_argnames=("accumulate", "unsafe"))
def torch_index_put_impl(x, indices, values, accumulate=False, unsafe=False):
    return torch.ops.aten._index_put_impl(x, indices, values, accumulate, unsafe)


# @register_aten(["aten::index_select_backward"], static_argnums=(1, 2))
def torch_index_select_backward(grad, dim, index, input_size):
    grad_input = torch.zeros(input_size, dtype=grad.dtype, device=grad.device)
    return grad_input.index_add_(dim, index, grad)
