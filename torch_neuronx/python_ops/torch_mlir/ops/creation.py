"""Tensor creation operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::arange", "aten::arange.start_out"], static_argnums=(0, 1, 2), static_argnames=("dtype",)
)
def torch_arange(start, end=None, step=1, dtype=None, **kwargs):
    if end is None:
        return torch.arange(start, dtype=dtype, **kwargs)
    return torch.arange(start, end, step, dtype=dtype, **kwargs)


# TODO there was a backward issue running this with gpt-oss
@register_aten(["aten::zeros", "aten::zeros.out"])
def torch_zeros(*size, dtype=None, layout=None, device=None, pin_memory=False, **kwargs):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):  # noqa: UP038
        size = size[0]
    x = torch.empty(size, dtype=dtype)
    return x.fill_(0)


@register_aten(["aten::ones", "aten::ones.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_ones(size, dtype=None, **kwargs):
    return torch.ones(size, dtype=dtype, **kwargs)


@register_aten(["aten::ones_like"])
def torch_ones_like(x, dtype=None, **kwargs):
    return torch.ones_like(x, dtype=dtype, **kwargs)


@register_aten(["aten::eye", "aten::eye.m_out"])
def torch_eye(n, m=None, dtype=None, **kwargs):
    return torch.eye(n, m, dtype=dtype, **kwargs)


@register_aten(["aten::fill_", "aten::fill_.Scalar"])
def torch_fill(x, value, out=None):
    return torch.full_like(x, value)


@register_aten(["aten::zero_"])
def torch_zero(x):
    return torch.zeros_like(x)


@register_aten(["aten::scalar_tensor"], static_argnames=("dtype",))
def torch_scalar_tensor(x, dtype=None):
    """
    torch.scalar_tensor is supposed to create a scalar tensor from scalar value
    but to avoid scalar value get baked into HLO and symInt/symFloat
    we are converting scalar to tensor before passing the inputs to aten ops
    so the input x here will be a tensor instead of scalar
    """
    return x.to(dtype)
