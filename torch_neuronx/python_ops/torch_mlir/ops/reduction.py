"""Reduction operations."""

import warnings

import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type_name

from ..operation_registry import register_aten
from .unary import _validate_float_promoting_out


def _reduction_numel(tensor, dim):
    """Compute the number of elements in the reduction dimensions."""
    if dim is None:
        return tensor.numel()
    if isinstance(dim, list | tuple):
        n = 1
        for d in dim:
            n *= tensor.shape[d]
        return n
    return tensor.shape[dim]


def _reduced_shape(tensor, dim, keepdim):
    """Compute the output shape after reduction."""
    if dim is None:
        return [1] * tensor.ndim if keepdim else []
    dims = [dim] if not isinstance(dim, list | tuple) else list(dim)
    out_shape = list(tensor.shape)
    for d in sorted(dims, reverse=True):
        if keepdim:
            out_shape[d] = 1
        else:
            out_shape.pop(d)
    return out_shape


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
    ["aten::sum", "aten::sum.dim_IntList", "aten::sum.IntList_out", "aten::sum.out"],
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
    validate_static_inputs=_validate_float_promoting_out,
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
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_max(x, dim=None, keepdim=False, out=None, **kwargs):
    # neuronx-cc cannot reduce over a pred (bool) operand ("operand: pred[]").
    # Reduce in int32 and cast the values back to bool; indices stay int64.
    if x.dtype == torch.bool:
        xi = x.to(torch.int32)
        if dim is None:
            return torch.max(xi).to(torch.bool)
        values, indices = torch.max(xi, dim, keepdim)
        return values.to(torch.bool), indices
    if dim is None:
        return torch.max(x)
    return torch.max(x, dim, keepdim)


@register_aten(
    ["aten::min", "aten::min.unary_out", "aten::min.dim", "aten::min.dim_min"],
    static_argnums=(1, 2),
    output_params=("min", "min_indices"),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_min(x, dim=None, keepdim=False, out=None, **kwargs):
    # neuronx-cc cannot reduce over a pred (bool) operand ("operand: pred[]").
    # Reduce in int32 and cast the values back to bool; indices stay int64.
    if x.dtype == torch.bool:
        xi = x.to(torch.int32)
        if dim is None:
            return torch.min(xi).to(torch.bool)
        values, indices = torch.min(xi, dim, keepdim)
        return values.to(torch.bool), indices
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


def _validate_std_inputs(self, dim=None, correction=1, keepdim=False, out=None, **kwargs):
    """Validate inputs for std/std_mean operations."""
    if dim is not None and self.ndim >= 64:
        raise RuntimeError("only tensors with up to 64 dims are supported")
    if (
        dim is not None
        and not isinstance(dim, list | tuple)
        and (dim >= self.ndim or dim < -self.ndim)
    ):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-self.ndim}, {self.ndim - 1}], but got {dim})"
        )
    if not self.is_floating_point() and self.ndim > 0:
        raise RuntimeError("std and var only support floating point and complex dtypes")
    _validate_float_promoting_out(self, out=out)


@register_aten(
    ["aten::std.correction", "aten::std.correction_out"],
    static_argnums=(1, 2, 3),
    validate_static_inputs=_validate_std_inputs,
)
def torch_std_correction(self, dim=None, correction=1, keepdim=False, out=None, **kwargs):
    if correction is None:
        correction = 1
    if self.ndim == 0:
        if dim is not None:
            _validate_scalar_dim(dim)
        dtype = self.dtype if self.is_floating_point() else torch.float32
        if correction > 0:
            return torch.full((), float("nan"), dtype=dtype, device=self.device)
        return torch.zeros((), dtype=dtype, device=self.device)
    numel = _reduction_numel(self, dim)
    if numel <= correction:
        warnings.warn("std(): degrees of freedom is <= 0", stacklevel=2)
        out_shape = _reduced_shape(self, dim, keepdim)
        return torch.full(out_shape, float("nan"), dtype=self.dtype, device=self.device)
    return torch.std(self, dim=dim, correction=correction, keepdim=keepdim)


@register_aten(
    ["aten::std_mean.correction", "aten::std_mean.correction_out"],
    static_argnums=(1, 2, 3),
    output_params=("out0", "out1"),
    validate_static_inputs=_validate_std_inputs,
)
def torch_std_mean_correction(
    self, dim=None, correction=1, keepdim=False, out0=None, out1=None, **kwargs
):
    if correction is None:
        correction = 1
    if self.ndim == 0:
        if dim is not None:
            _validate_scalar_dim(dim)
        dtype = self.dtype if self.is_floating_point() else torch.float32
        if correction > 0:
            s = torch.full((), float("nan"), dtype=dtype, device=self.device)
        else:
            s = torch.zeros((), dtype=dtype, device=self.device)
        return s, self.clone().to(dtype)
    numel = _reduction_numel(self, dim)
    if numel <= correction:
        warnings.warn("std(): degrees of freedom is <= 0", stacklevel=2)
        out_shape = _reduced_shape(self, dim, keepdim)
        s = torch.full(out_shape, float("nan"), dtype=self.dtype, device=self.device)
        m = torch.mean(self, dim=dim, keepdim=keepdim) if dim is not None else torch.mean(self)
        if dim is None and keepdim:
            m = m.reshape([1] * self.ndim)
        return s, m
    if dim is None:
        s = torch.std(self, correction=correction)
        m = torch.mean(self)
        if keepdim:
            shape = [1] * self.ndim
            s = s.reshape(shape)
            m = m.reshape(shape)
        return s, m
    return (
        torch.std(self, dim, correction=correction, keepdim=keepdim),
        torch.mean(self, dim, keepdim=keepdim),
    )


@register_aten("aten::value_selecting_reduction_backward", static_argnums=(1, 3, 4))
def torch_value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    return torch.ops.aten.value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim)


def _cdist_decompose(x1, x2, p):
    """Compute cdist by an explicit pairwise reduction, shape (*batch, P, R).

    x1 is (*batch1, P, M), x2 is (*batch2, R, M); batch dims broadcast. Batch
    shapes are static at trace time, so torch broadcasting resolves differing
    ranks / size-1 batch dims here -- this deliberately avoids torch-mlir's
    _cdist_forward lowering, which marks differing-rank or broadcast batch dims
    illegal, and also fixes p=0 / p=inf, whose generic ^(1/p) path degenerates
    to wrong results.
    """
    ad = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()  # (*batch, P, R, M)
    if p == 0:
        return (ad != 0).to(x1.dtype).sum(-1)
    if p == float("inf"):
        return ad.amax(-1)
    if p == 1:
        return ad.sum(-1)
    if p == 2:
        return (ad * ad).sum(-1).sqrt()
    return ad.pow(p).sum(-1).pow(1.0 / p)


@register_aten(
    ["aten::_cdist_forward", "aten::_cdist_forward.out"],
    static_argnums=(2, 3),
)
def torch_cdist_forward(x1, x2, p, compute_mode=None, out=None):
    # torch-mlir marks differing-rank / broadcast batch dims illegal, and the
    # generic path degenerates for p=0 (^inf) and p=inf (^0 == 1) -> wrong
    # results. Decompose those cases explicitly; otherwise use the native op
    # (which has a fast matmul path for p=2 via compute_mode). Shapes/p static.
    needs_broadcast = x1.shape[:-2] != x2.shape[:-2]
    if p == 0 or p == float("inf") or needs_broadcast:
        return _cdist_decompose(x1, x2, p)
    return torch.ops.aten._cdist_forward(x1, x2, p, compute_mode)


def _validate_pdist(self, p=2.0, out=None):
    """Reject dtypes pdist can't handle correctly.

    CPU pdist (and its OpInfo) supports only float32/float64; other dtypes raise
    on CPU, so we raise here too rather than silently accepting fp16/bf16 or
    hitting an internal compiler error on integers. Under strict-fallback mode
    (TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1) this surfaces as an
    explicit error, matching CPU behavior.

    The message is spelled identically to CPU (e.g. 'Half', not 'torch.float16'),
    using the shared ScalarType-name table in ``dtype_utils``.
    """
    if self.dtype not in (torch.float32, torch.float64):
        aten_name = torch_dtype_to_scalar_type_name(self.dtype) or str(self.dtype)
        raise NotImplementedError(f"\"pdist\" not implemented for '{aten_name}'")


def _pdist_pairwise(self, reduce_fn):
    """Compute pdist for the strict upper triangle (i < j) by reducing each
    pair's (d,) difference vector, in torch.pdist's row-major order.

    The row count n is static at trace time, so self[i]/self[j] are static row
    indices and the pair loop unrolls in the trace. This deliberately avoids the
    (n, n, d) broadcast + amax-over-last-dim + matrix-slice path, which either
    fails fx import (constructed index tensors: "cannot copy out of meta") or
    trips an internal neuronx-cc compiler error on the 3-D reduction.
    """
    n = self.shape[0]
    vals = [
        reduce_fn((self[i] - self[j]).abs()).reshape(1)
        for i in range(n - 1)
        for j in range(i + 1, n)
    ]
    return torch.cat(vals) if vals else self.new_zeros(0)


@register_aten(
    ["aten::_pdist_forward", "aten::_pdist_forward.out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_pdist,
)
def torch_pdist_forward(self, p=2.0, out=None):
    # The generic sum(|diff|^p)^(1/p) lowering degenerates for p=0 (^inf) and
    # p=inf (^0 == 1), silently producing wrong results. Decompose those two
    # cases into an explicit per-pair reduction (p=0: count of differing coords;
    # p=inf: max abs coord diff) so the Neuron result is correct. p is static.
    if p == 0:
        return _pdist_pairwise(self, lambda ad: (ad != 0).to(self.dtype).sum())
    if p == float("inf"):
        return _pdist_pairwise(self, lambda ad: ad.amax())
    return torch.ops.aten._pdist_forward(self, p)
