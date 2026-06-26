"""Tensor creation operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::arange"], static_argnums=(0,), static_argnames=("dtype",))
def torch_arange_end(end, dtype=None, **kwargs):
    return torch.arange(end, dtype=dtype, **kwargs)


@register_aten(["aten::arange.start_out"], static_argnums=(0, 1, 2), static_argnames=("dtype",))
def torch_arange(start, end, step=1, dtype=None, out=None, **kwargs):
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.arange(start, end, step, dtype=dtype, **kwargs)


@register_aten(["aten::zeros", "aten::zeros.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_zeros(size, dtype=None, out=None, **kwargs):
    return torch.zeros(size, dtype=dtype, **kwargs)


@register_aten(["aten::ones", "aten::ones.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_ones(size, dtype=None, out=None, **kwargs):
    return torch.ones(size, dtype=dtype, **kwargs)


@register_aten(
    ["aten::ones_like", "aten::ones_like.out"], static_argnames=("dtype",), output_params=("out",)
)
def torch_ones_like(x, dtype=None, out=None, **kwargs):
    return torch.ones_like(x, dtype=dtype, **kwargs)


@register_aten(["aten::eye", "aten::eye.out"], static_argnums=(0, 1), static_argnames=("dtype",))
def torch_eye(n, dtype=None, out=None, **kwargs):
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.eye(n, dtype=dtype, **kwargs)


@register_aten(
    ["aten::eye.m", "aten::eye.m_out"],
    static_argnums=(0, 1),
    static_argnames=("dtype",),
)
def torch_eye_m(n, m=None, dtype=None, out=None, **kwargs):
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.eye(n, m, dtype=dtype, **kwargs)


@register_aten(["aten::fill_.Tensor"], static_argnums=(1,))
def torch_fill_tensor(x, value, out=None):
    """Fill tensor with tensor value"""
    return torch.ops.aten.fill.Tensor(x, value)


@register_aten(["aten::fill_.Scalar"], static_argnums=(1,))
def torch_fill_scalar(x, value, out=None):
    """Fill tensor with scalar value"""
    return torch.ops.aten.fill.Scalar(x, value)


@register_aten(["aten::full", "aten::full.out"], static_argnums=(0,), static_argnames=("dtype",))
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


@register_aten(
    ["aten::scalar_tensor", "aten::scalar_tensor.out"],
    static_argnames=("dtype",),
    output_params=("out",),
)
def torch_scalar_tensor(x, dtype=None, out=None):
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


# kaiser_window functional variants (aten::kaiser_window, .periodic, .beta) work
# via CompositeImplicitAutograd decomposition into primitive ops. Only the .out
# variants need explicit registration because the .out dispatch path bypasses
# composite decomposition and hits DispatchStub directly.
#
# Note: the lowering currently falls back to CPU because torch.i0 (used
# internally by kaiser_window) is not supported in Torch-MLIR. The registration
# still resolves the DispatchStub assertion crash — moving the op from
# failed_lowering to cpu_fallback with correct results.


@register_aten(["aten::kaiser_window.out"], static_argnums=(0,), output_params=("out",))
def torch_kaiser_window_out(window_length, out=None):
    return torch.kaiser_window(window_length)


@register_aten(["aten::kaiser_window.periodic_out"], static_argnums=(0, 1), output_params=("out",))
def torch_kaiser_window_periodic_out(window_length, periodic, out=None):
    return torch.kaiser_window(window_length, periodic=periodic)


@register_aten(["aten::kaiser_window.beta_out"], static_argnums=(0, 1, 2), output_params=("out",))
def torch_kaiser_window_beta_out(window_length, periodic, beta, out=None):
    return torch.kaiser_window(window_length, periodic=periodic, beta=beta)
