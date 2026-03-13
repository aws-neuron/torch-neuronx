"""Tensor creation operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::arange", "aten::arange.start_out"], static_argnums=(0, 1, 2), static_argnames=("dtype",)
)
def torch_arange(start, end=None, step=1, dtype=None, out=None, **kwargs):
    if end is None:
        return torch.arange(start, dtype=dtype, **kwargs)
    return torch.arange(start, end, step, dtype=dtype, **kwargs)


@register_aten(
    ["aten::zeros", "aten::zeros.out"], static_argnums=(0,), static_argnames=("dtype", "device")
)
def torch_zeros(size, dtype=None, layout=None, device=None, pin_memory=False, out=None, **kwargs):
    return torch.zeros(size, dtype=dtype, device=device)


@register_aten(["aten::ones", "aten::ones.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_ones(size, dtype=None, out=None, **kwargs):
    return torch.ones(size, dtype=dtype, **kwargs)


@register_aten(["aten::ones_like"])
def torch_ones_like(x, dtype=None, **kwargs):
    return torch.ones_like(x, dtype=dtype, **kwargs)


@register_aten(["aten::eye", "aten::eye.out"], static_argnums=(0, 1))
def torch_eye(n, dtype=None, out=None, **kwargs):
    return torch.eye(n, dtype=dtype, **kwargs)


@register_aten(["aten::eye.m", "aten::eye.m_out"], static_argnums=(0, 1))
def torch_eye_m(n, m=None, dtype=None, out=None, **kwargs):
    return torch.eye(n, m, dtype=dtype, **kwargs)


@register_aten(["aten::fill_.Tensor"], static_argnums=(1,))
def torch_fill_tensor(x, value, out=None):
    """Fill tensor with tensor value"""
    return torch.ops.aten.fill.Tensor(x, value)


@register_aten(["aten::fill_.Scalar"], static_argnums=(1,))
def torch_fill_scalar(x, value, out=None):
    """Fill tensor with scalar value"""
    return torch.ops.aten.fill.Scalar(x, value)


@register_aten(["aten::full", "aten::full.out"], static_argnums=(0,))
def torch_full(size, value, out=None, dtype=None):
    """Scalar are converted into tensor to avoid recompilation
    so input `value` will be a tensor
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=dtype)

    return torch.broadcast_to(value, size).to(dtype)


@register_aten(["aten::zero_"])
def torch_zero(x, out=None):
    return torch.zeros_like(x)


@register_aten(["aten::scalar_tensor"], static_argnames=("dtype",))
def torch_scalar_tensor(x, dtype=None):
    """
    torch.scalar_tensor is supposed to create a scalar tensor from scalar value
    but to avoid scalar value get baked into HLO and symInt/symFloat
    we are converting scalar to tensor before passing the inputs to aten ops
    so the input x here will be a tensor instead of scalar
    """
    if dtype is None and (type(x) is int or type(x) is bool):
        dtype = torch.float32
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    else:
        return torch.tensor(x, dtype=dtype)
