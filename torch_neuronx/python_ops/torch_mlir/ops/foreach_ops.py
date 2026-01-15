"""Foreach operations for torch-mlir backend."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::_foreach_norm.Scalar", "aten::_foreach_norm.Scalar_out"], static_argnums=(1,)
)
def torch_foreach_norm(tensors, ord=2, dtype=None, out=None):
    """Foreach norm operation."""
    return tuple([torch.linalg.vector_norm(t.flatten(), ord=ord, dtype=dtype) for t in tensors])


@register_aten(["aten::_foreach_add.Scalar", "aten::_foreach_add.Scalar_out"])
def torch_foreach_add_scalar(tensors, scalar, out=None):
    """Foreach add with scalar."""
    return tuple([torch.add(t, scalar) for t in tensors])


@register_aten(["aten::_foreach_add.List", "aten::_foreach_add.List_out"], static_argnums=(2,))
def torch_foreach_add_list(tensors1, tensors2, alpha=1, out=None):
    """Foreach add between tensor lists."""
    return tuple(
        [torch.add(t1, t2, alpha=alpha) for t1, t2 in zip(tensors1, tensors2, strict=False)]
    )


@register_aten(["aten::_foreach_mul.Scalar", "aten::_foreach_mul.Scalar_out"])
def torch_foreach_mul_scalar(tensors, scalar, out=None):
    """Foreach multiply with scalar."""
    return tuple([torch.mul(t, scalar) for t in tensors])


@register_aten(["aten::_foreach_mul.List", "aten::_foreach_mul.List_out"])
def torch_foreach_mul_list(tensors1, tensors2, out=None):
    """Foreach multiply between tensor lists."""
    return tuple([torch.mul(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False)])


@register_aten(["aten::_foreach_sub.Scalar", "aten::_foreach_sub.Scalar_out"], static_argnums=(2,))
def torch_foreach_sub_scalar(tensors, scalar, alpha=1, out=None):
    """Foreach subtract with scalar."""
    return tuple([torch.sub(t, scalar, alpha=alpha) for t in tensors])


@register_aten(["aten::_foreach_sub.List", "aten::_foreach_sub.List_out"], static_argnums=(2,))
def torch_foreach_sub_list(tensors1, tensors2, alpha=1, out=None):
    """Foreach subtract between tensor lists."""
    return tuple(
        [torch.sub(t1, t2, alpha=alpha) for t1, t2 in zip(tensors1, tensors2, strict=False)]
    )


@register_aten(["aten::_foreach_div.Scalar", "aten::_foreach_div.Scalar_out"])
def torch_foreach_div_scalar(tensors, scalar, out=None):
    """Foreach divide with scalar."""
    return tuple([torch.div(t, scalar) for t in tensors])


@register_aten(["aten::_foreach_div.List", "aten::_foreach_div.List_out"])
def torch_foreach_div_list(tensors1, tensors2, out=None):
    """Foreach divide between tensor lists."""
    return tuple([torch.div(t1, t2) for t1, t2 in zip(tensors1, tensors2, strict=False)])
