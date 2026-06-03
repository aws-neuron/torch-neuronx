"""Reduction operations."""

import torch

from ..operation_registry import register_aten


def _validate_scalar_dim(dim):
    """Validate dimension for scalar tensors."""
    dim_list = dim if isinstance(dim, list | tuple) else [dim]
    for d in dim_list:
        if d < -1 or d > 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
            )


def _infer_int64_promotion(dtype, tensor):
    """Infer int64 target dtype for integer/bool inputs (PyTorch sum/cumsum semantics)."""
    if dtype is not None:
        return dtype
    if not tensor.is_floating_point() and not tensor.is_complex():
        return torch.int64
    return None


def _reduce_with_promotion(op_fn, target_dtype, caller_dtype, *args, **kwargs):
    """Execute reduction op, passing caller dtype through or applying deferred int64 promotion."""
    if caller_dtype is not None:
        return op_fn(*args, dtype=caller_dtype, **kwargs)
    result = op_fn(*args, **kwargs)
    if target_dtype is not None and result.dtype != target_dtype:
        result = result.to(target_dtype)
    return result


@register_aten(
    ["aten::sum", "aten::sum.dim_IntList", "aten::sum.IntList_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
)
def torch_sum(self, dim=None, keepdim=False, dtype=None, out=None):
    target_dtype = _infer_int64_promotion(dtype, self)
    if self.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return self.clone().to(target_dtype) if target_dtype else self.clone()
    return _reduce_with_promotion(torch.sum, target_dtype, dtype, self, dim, keepdim)


@register_aten(
    ["aten::mean", "aten::mean.dim", "aten::mean.out", "aten::mean.dtype_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
)
def torch_mean(x, dim=None, keepdim=False, dtype=None, out=None):
    # Scalar tensor with dim: short-circuit to avoid MLIR crash
    if x.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return x.clone().to(dtype) if dtype else x.clone()
    if dim is None:
        return torch.mean(x, dtype=dtype)
    return torch.mean(x, dim, keepdim, dtype=dtype)


@register_aten(
    ["aten::max", "aten::max.unary_out", "aten::max.dim", "aten::max.dim_max"],
    static_argnums=(1, 2),
    output_params=("max", "max_values"),
)
def torch_max(x, dim=None, keepdim=False, out=None, **kwargs):
    if dim is None:
        return torch.max(x)
    return torch.max(x, dim, keepdim)


@register_aten(
    ["aten::min", "aten::min.unary_out", "aten::min.dim", "aten::min.dim_min"],
    static_argnums=(1, 2),
    output_params=("min", "min_indices"),
)
def torch_min(x, dim=None, keepdim=False, out=None, **kwargs):
    if dim is None:
        return torch.min(x)
    return torch.min(x, dim, keepdim)


def _validate_amax(x, dim=None, keepdim=False, out=None):
    """Validate amax dtypes on raw args."""
    if out is not None and x.dtype != out.dtype:
        raise TypeError(
            f"Expected the dtype for input and out to match, but got "
            f"{x.dtype} for input's dtype and {out.dtype} for out's dtype."
        )


@register_aten(
    ["aten::amax", "aten::amax.out"],
    static_argnums=(1, 2),
    validate_static_inputs=_validate_amax,
)
def torch_amax(x, dim=None, keepdim=False, out=None):
    if x.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return x.clone()
    return torch.amax(x, dim, keepdim)


def _validate_amin(self, dim=None, keepdim=False, out=None):
    """Validate amin dtypes on raw args."""
    if out is not None and self.dtype != out.dtype:
        raise TypeError(
            f"Expected the dtype for input and out to match, but got "
            f"{self.dtype} for input's dtype and {out.dtype} for out's dtype."
        )


@register_aten(
    ["aten::amin", "aten::amin.out"],
    static_argnums=(1, 2),
    validate_static_inputs=_validate_amin,
)
def torch_amin(self, dim=None, keepdim=False, out=None):
    if self.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return self.clone()
    return torch.amin(self, dim, keepdim)


def _bool_reduction(op, x, dim=None, keepdim=False):
    """Shared logic for all/any reductions.

    Handles scalar tensors and non-bool input casting to work around
    Torch-MLIR StableHLO lowering limitations. Output dtype follows
    PyTorch convention: uint8 in -> uint8 out, otherwise bool.
    """
    input_dtype = x.dtype
    # Scalar tensor with dim: short-circuit to avoid MLIR crash
    if x.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        result = x.clone().to(torch.bool)
    else:
        # Cast to bool before reduction; Torch-MLIR may not lower all/any on non-bool dtypes.
        x = x.to(torch.bool)
        result = op(x) if dim is None else op(x, dim, keepdim)
    # Reference: https://docs.pytorch.org/docs/stable/generated/torch.any.html
    return result.to(torch.uint8) if input_dtype == torch.uint8 else result


@register_aten(["aten::all", "aten::all.out", "aten::all.all_out"], static_argnums=(1, 2))
def torch_all(x, dim=None, keepdim=False, out=None):
    return _bool_reduction(torch.all, x, dim, keepdim)


@register_aten(["aten::any", "aten::any.out", "aten::any.all_out"], static_argnums=(1, 2))
def torch_any(x, dim=None, keepdim=False, out=None):
    return _bool_reduction(torch.any, x, dim, keepdim)


@register_aten(
    ["aten::cumsum", "aten::cumsum.out"], static_argnums=(1,), static_argnames=("dtype",)
)
def torch_cumsum(x, dim, dtype=None, out=None):
    target_dtype = _infer_int64_promotion(dtype, x)
    # Neuron doesn't support int8/uint8; cast to int32 for computation
    if x.dtype in (torch.int8, torch.uint8):
        x = x.to(torch.int32)
    if x.ndim == 0:
        return x.clone().to(target_dtype) if target_dtype else x.clone()
    return _reduce_with_promotion(torch.cumsum, target_dtype, dtype, x, dim)


@register_aten("aten::value_selecting_reduction_backward", static_argnums=(1, 3, 4))
def torch_value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    return torch.ops.aten.value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim)
