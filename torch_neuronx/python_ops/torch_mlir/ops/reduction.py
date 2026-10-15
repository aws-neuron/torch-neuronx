"""Reduction operations."""

import warnings

import torch

from torch_neuronx.dtype_utils import torch_dtype_to_scalar_type_name

from ..operation_registry import register_aten
from ..validation import make_bool_result_out_validator
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
    """Compute a sum with PyTorch integer-promotion semantics.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve reduced dimensions when ``True``. Defaults to
            ``False``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Reduced tensor, promoted to int64 for integer and bool inputs by default.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
    """
    target_dtype = _infer_int64_promotion(dtype, self)
    if self.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return self.clone().to(target_dtype) if target_dtype else self.clone()
    return _reduce_with_promotion(torch.sum, target_dtype, dtype, self, dim, keepdim)


@register_aten(["aten::prod", "aten::prod.out"], static_argnames=("dtype",), output_params=("out",))
def torch_prod(self, dtype=None, out=None):
    """Compute the product of all input elements.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Scalar product, promoted to int64 for integer and Boolean inputs by default.

    Note:
        Float16 and bfloat16 inputs accumulate in float32 before casting back when ``dtype``
        is omitted.
    """
    if dtype is None and self.dtype in (torch.bfloat16, torch.float16):
        return torch.prod(self.to(torch.float32)).to(self.dtype)
    target_dtype = _infer_int64_promotion(dtype, self)
    return _reduce_with_promotion(torch.prod, target_dtype, dtype, self)


@register_aten(
    ["aten::prod.dim_int", "aten::prod.int_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
    output_params=("out",),
)
def torch_prod_dim(self, dim, keepdim=False, dtype=None, out=None):
    """Compute a product over one dimension.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int): Dimension to reduce.
        keepdim (bool, optional): Preserve the reduced dimension. Defaults to ``False``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Dimension product with the same promotion rules as ``torch_prod``.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.

    Note:
        This overload is registered separately from full reduction so their output shapes do
        not share shape inference. Float16 and bfloat16 accumulate in float32 by default.
    """
    target_dtype = _infer_int64_promotion(dtype, self)
    if self.ndim == 0:
        _validate_scalar_dim(dim)
        return self.clone().to(target_dtype) if target_dtype else self.clone()
    if dtype is None and self.dtype in (torch.bfloat16, torch.float16):
        return torch.prod(self.to(torch.float32), dim, keepdim).to(self.dtype)
    return _reduce_with_promotion(torch.prod, target_dtype, dtype, self, dim, keepdim)


@register_aten(
    ["aten::mean", "aten::mean.dim", "aten::mean.out", "aten::mean.dtype_out"],
    static_argnums=(1, 2),
    static_argnames=("dtype",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_mean(x, dim=None, keepdim=False, dtype=None, out=None):
    # Scalar tensor with dim: short-circuit to avoid MLIR crash
    """Compute the mean of a tensor over selected dimensions.

    Args:
        x (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve selected dimensions; ignored without ``dim``.
            Defaults to ``False``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Destination validated and written by dispatch. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Mean values in the requested dtype.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
        RuntimeError: If floating-point ``x`` uses a non-floating-point ``out`` destination.
    """
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
    """Compute maximum values and optional dimension indices.

    Args:
        x (torch.Tensor): Tensor to reduce.
        dim (int, optional): Dimension to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve a selected dimension; ignored without ``dim``.
            Defaults to ``False``.
        out (torch.Tensor or tuple[torch.Tensor, ...], optional): Output destination. Defaults to
            ``None``.
        **kwargs (Any): Named value and index destinations handled by dispatch.

    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Maximum values and optional indices.

    Raises:
        RuntimeError: If floating-point ``x`` uses a non-floating-point ``out`` destination.

    Note:
        Boolean inputs are reduced as int32 and cast back; indices remain int64.
    """
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
    """Compute minimum values and optional dimension indices.

    Args:
        x (torch.Tensor): Tensor to reduce.
        dim (int, optional): Dimension to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve a selected dimension; ignored without ``dim``.
            Defaults to ``False``.
        out (torch.Tensor or tuple[torch.Tensor, ...], optional): Output destination. Defaults to
            ``None``.
        **kwargs (Any): Named value and index destinations handled by dispatch.

    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Minimum values and optional indices.

    Raises:
        RuntimeError: If floating-point ``x`` uses a non-floating-point ``out`` destination.

    Note:
        Boolean inputs are reduced as int32 and cast back; indices remain int64.
    """
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
    """Compute maximum values over selected dimensions.

    Args:
        x (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve reduced dimensions when ``True``. Defaults to
            ``False``.
        out (torch.Tensor, optional): Output tensor whose dtype is validated against ``x``.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Maximum values.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
        TypeError: If ``x`` and ``out`` have different dtypes.
    """
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
    """Compute minimum values over selected dimensions.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve reduced dimensions when ``True``. Defaults to
            ``False``.
        out (torch.Tensor, optional): Output tensor whose dtype is validated against ``self``.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Minimum values.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
        TypeError: If ``self`` and ``out`` have different dtypes.
    """
    if self.ndim == 0 and dim is not None:
        _validate_scalar_dim(dim)
        return self.clone()
    return torch.amin(self, dim, keepdim)


def _validate_argmax(self, dim=None, keepdim=False, out=None):
    """Reject the dtypes and shapes aten::argmax does not support."""
    if self.dtype == torch.bool:
        raise RuntimeError("argmax(): does not support bool input")
    if self.is_complex():
        raise RuntimeError("argmax(): does not support complex input")
    if dim is not None and self.ndim > 64:
        raise RuntimeError("only tensors with up to 64 dims are supported")


@register_aten(
    ["aten::argmax", "aten::argmax.out"],
    static_argnums=(1, 2),
    validate_static_inputs=_validate_argmax,
)
def torch_argmax(self, dim=None, keepdim=False, out=None):
    """Return indices of maximum values over an optional dimension.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int, optional): Dimension to reduce, or all dimensions when ``None``. Defaults to
            ``None``.
        keepdim (bool, optional): Preserve reduced dimensions. Defaults to ``False``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Int64 indices of maximum values.

    Raises:
        IndexError: If ``dim`` is invalid or the selected reduction is empty.
        RuntimeError: If the input is Boolean or complex, or exceeds 64 dimensions with
            ``dim`` set.
    """
    # A scalar input always has the single index 0; torch-mlir cannot reduce a
    # 0-d operand, so return the index directly and match aten's 0-d int64 output.
    if self.ndim == 0:
        if dim is not None:
            _validate_scalar_dim(dim)
        return torch.zeros((), dtype=torch.int64, device=self.device)
    return torch.ops.aten.argmax(self, dim, keepdim)


def _validate_argmin(self, dim=None, keepdim=False, out=None):
    """Reject the dtypes and shapes aten::argmin does not support."""
    if self.dtype == torch.bool:
        raise RuntimeError("argmin(): does not support bool input")
    if self.is_complex():
        raise RuntimeError("argmin(): does not support complex input")
    if dim is None and self.numel() == 0:
        raise IndexError("argmin(): Expected reduction dim to be specified for input.numel() == 0.")
    if dim is not None and self.ndim > 64:
        raise RuntimeError("only tensors with up to 64 dims are supported")


@register_aten(
    ["aten::argmin", "aten::argmin.out"],
    static_argnums=(1, 2),
    validate_static_inputs=_validate_argmin,
)
def torch_argmin(self, dim=None, keepdim=False, out=None):
    """Return indices of minimum values over an optional dimension.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int, optional): Dimension to reduce, or all dimensions when ``None``. Defaults to
            ``None``.
        keepdim (bool, optional): Preserve reduced dimensions. Defaults to ``False``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Int64 indices of minimum values.

    Raises:
        IndexError: If no dimension is given for an empty tensor or a scalar dimension is invalid.
        RuntimeError: If the input is Boolean or complex, or exceeds 64 dimensions with
            ``dim`` set.
    """
    # A scalar input always has the single index 0; torch-mlir cannot reduce a
    # 0-d operand, so return the index directly and match aten's 0-d int64 output.
    if self.ndim == 0:
        if dim is not None:
            _validate_scalar_dim(dim)
        return torch.zeros((), dtype=torch.int64, device=self.device)
    result = torch.ops.aten.argmin(self, dim, keepdim)
    # The native argmin lowering drops keepdim for a full reduction (dim=None) and
    # returns a 0-d index. aten keeps every reduced dim as size 1, so restore that
    # shape here.
    if dim is None and keepdim:
        return result.reshape([1] * self.ndim)
    return result


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


@register_aten(
    ["aten::all", "aten::all.out", "aten::all.all_out"],
    static_argnums=(1, 2),
    validate_static_inputs=make_bool_result_out_validator("all"),
)
def torch_all(x, dim=None, keepdim=False, out=None):
    """Test whether all elements are true over selected dimensions.

    Args:
        x (torch.Tensor): Tensor whose elements are tested.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve selected dimensions; ignored without ``dim``.
            Defaults to ``False``.
        out (torch.Tensor, optional): Boolean or uint8 destination written by dispatch. Defaults
            to ``None``.

    Returns:
        torch.Tensor: uint8 results for uint8 input, otherwise Boolean results.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
        RuntimeError: If ``out`` is neither Boolean nor uint8.
    """
    return _bool_reduction(torch.all, x, dim, keepdim)


@register_aten(
    ["aten::any", "aten::any.out", "aten::any.all_out"],
    static_argnums=(1, 2),
    validate_static_inputs=make_bool_result_out_validator("any"),
)
def torch_any(x, dim=None, keepdim=False, out=None):
    """Test whether any element is true over selected dimensions.

    Args:
        x (torch.Tensor): Tensor whose elements are tested.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        keepdim (bool, optional): Preserve selected dimensions; ignored without ``dim``.
            Defaults to ``False``.
        out (torch.Tensor, optional): Boolean or uint8 destination written by dispatch. Defaults
            to ``None``.

    Returns:
        torch.Tensor: uint8 results for uint8 input, otherwise Boolean results.

    Raises:
        IndexError: If a scalar tensor is given an invalid dimension.
        RuntimeError: If ``out`` is neither Boolean nor uint8.
    """
    return _bool_reduction(torch.any, x, dim, keepdim)


@register_aten(
    ["aten::cumsum", "aten::cumsum.out"], static_argnums=(1,), static_argnames=("dtype",)
)
def torch_cumsum(x, dim, dtype=None, out=None):
    """Compute a cumulative sum along a dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to accumulate.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Accepted for out-overload compatibility and otherwise
            ignored. Defaults to ``None``.

    Returns:
        torch.Tensor: Cumulative sums, promoted to int64 for integer and bool inputs by default.

    Note:
        int8 and uint8 inputs are computed as int32 before result promotion.
    """
    target_dtype = _infer_int64_promotion(dtype, x)
    # Neuron doesn't support int8/uint8; cast to int32 for computation
    if x.dtype in (torch.int8, torch.uint8):
        x = x.to(torch.int32)
    if x.ndim == 0:
        return x.clone().to(target_dtype) if target_dtype else x.clone()
    return _reduce_with_promotion(torch.cumsum, target_dtype, dtype, x, dim)


# Narrow int and bool inputs are widened to int32 for the scan, then cast back.
_CUMMAXMIN_INT_CAST = (torch.bool, torch.int8, torch.uint8, torch.int16)


def cummaxmin_scan(self, dim, is_max):
    """Compute inclusive cumulative extrema with selected indices.

    Args:
        self (torch.Tensor): Tensor to scan.
        dim (int): Dimension along which to scan.
        is_max (bool): Select cumulative maximum when true, otherwise cumulative minimum.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Cumulative values and int64 selected indices.

    Note:
        Uses an unrolled Hillis-Steele scan because torch-MLIR has no native lowering. Ties
        select the later index and NaNs win, matching ``torch.cummax`` and ``torch.cummin``.
        Narrow integer and Boolean values are scanned in int32 and cast back.
    """
    work = self.to(torch.int32) if self.dtype in _CUMMAXMIN_INT_CAST else self
    if self.ndim == 0:
        idx = torch.zeros((), dtype=torch.int64, device=self.device)
        return self.clone(), idx
    d = dim if dim >= 0 else dim + self.ndim
    n = work.shape[d]
    ramp_shape = [1] * work.ndim
    ramp_shape[d] = n
    ramp = torch.arange(n, device=work.device).reshape(ramp_shape)
    values = work.clone()
    indices = ramp.expand(work.shape).to(torch.int64).contiguous()
    is_float = work.is_floating_point()
    offset = 1
    while offset < n:
        pad_shape = list(work.shape)
        pad_shape[d] = offset
        earlier_v = torch.cat([values.new_zeros(pad_shape), values.narrow(d, 0, n - offset)], dim=d)
        earlier_i = torch.cat(
            [indices.new_zeros(pad_shape), indices.narrow(d, 0, n - offset)], dim=d
        )
        beats = (earlier_v > values) if is_max else (earlier_v < values)
        if is_float:
            earlier_nan = torch.isnan(earlier_v)
            beats = (earlier_nan & ~torch.isnan(values)) | (~earlier_nan & beats)
        take_earlier = beats & (ramp >= offset)
        values = torch.where(take_earlier, earlier_v, values)
        indices = torch.where(take_earlier, earlier_i, indices)
        offset *= 2
    if values.dtype != self.dtype:
        values = values.to(self.dtype)
    return values, indices


def _validate_cummaxmin_out(self, dim, out=None, values=None, **kwargs):
    # cummax.out/cummin.out pass the values buffer as the `values` kwarg, not out=.
    values_out = out[0] if isinstance(out, tuple | list) else (out or values)
    _validate_float_promoting_out(self, out=values_out)


# torch.cummax/cummin are composite ops that call _cummax_helper, whose () return
# and Tensor(a!) args the C++ writeback cannot bind. A backend kernel for the
# public op takes priority over the composite, so it dispatches here and returns
# (values, indices) for the C++ writeback.
@register_aten(
    ["aten::cummax", "aten::cummax.out"],
    static_argnums=(1,),
    output_params=("values", "indices"),
    validate_static_inputs=_validate_cummaxmin_out,
)
def torch_cummax(self, dim, out=None, **kwargs):
    """Compute cumulative maximum values and indices.

    Args:
        self (torch.Tensor): Tensor to scan.
        dim (int): Dimension along which to scan.
        out (tuple[torch.Tensor, torch.Tensor] | None, optional): Output buffers handled by
            dispatch. Defaults to ``None``.
        **kwargs (Any): Additional dispatcher keywords, including named output buffers.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Cumulative maxima and int64 selected indices.

    Raises:
        IndexError: If ``dim`` is invalid for a nonscalar tensor.
        RuntimeError: If floating-point input uses a non-floating-point values output.
    """
    return cummaxmin_scan(self, dim, is_max=True)


@register_aten(
    ["aten::cummin", "aten::cummin.out"],
    static_argnums=(1,),
    output_params=("values", "indices"),
    validate_static_inputs=_validate_cummaxmin_out,
)
def torch_cummin(self, dim, out=None, **kwargs):
    """Compute cumulative minimum values and indices.

    Args:
        self (torch.Tensor): Tensor to scan.
        dim (int): Dimension along which to scan.
        out (tuple[torch.Tensor, torch.Tensor] | None, optional): Output buffers handled by
            dispatch. Defaults to ``None``.
        **kwargs (Any): Additional dispatcher keywords, including named output buffers.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Cumulative minima and int64 selected indices.

    Raises:
        IndexError: If ``dim`` is invalid for a nonscalar tensor.
        RuntimeError: If floating-point input uses a non-floating-point values output.
    """
    return cummaxmin_scan(self, dim, is_max=False)


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
    """Compute standard deviation with a degrees-of-freedom correction.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        correction (int | float | None, optional): Defaults to ``1``; ``None`` also uses ``1``.
        keepdim (bool, optional): Preserve reduced dimensions. Defaults to ``False``.
        out (torch.Tensor, optional): Validated and written destination. Defaults to ``None``.
        **kwargs: Additional overload arguments accepted and ignored.

    Returns:
        torch.Tensor: Standard deviation with implementation-specific scalar handling.

    Raises:
        IndexError: If a reduction dimension is invalid.
        RuntimeError: If rank, input dtype, or output dtype validation fails.

    Note:
        Non-scalar nonpositive degrees of freedom warn and return NaN. Scalars return NaN for
        positive correction, otherwise zero, without warning; multiple negative axes may mis-shape.
    """
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
    """Compute standard deviation and mean with a degrees-of-freedom correction.

    Args:
        self (torch.Tensor): Tensor to reduce.
        dim (int or sequence[int], optional): Dimensions to reduce. Defaults to ``None``.
        correction (int | float | None, optional): Correction; omission and ``None`` use ``1``.
        keepdim (bool, optional): Preserve reduced dimensions. Defaults to ``False``.
        out0 (torch.Tensor, optional): Standard-deviation output. Defaults to ``None``.
        out1 (torch.Tensor, optional): Mean output. Defaults to ``None``.
        **kwargs: Additional overload arguments accepted and ignored.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Standard deviation and mean tensors.

    Raises:
        IndexError: If a reduction dimension is invalid.
        RuntimeError: For rank 64 or greater with ``dim`` set or a nonscalar non-floating input.

    Note:
        Nonpositive degrees of freedom return NaN; only non-scalar cases also emit a warning.
    """
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
    """Compute the backward pass for a value-selecting reduction.

    Args:
        grad (torch.Tensor): Gradient of the reduction output.
        dim (int): Reduced dimension.
        indices (torch.Tensor): Selected input indices.
        sizes (sequence[int]): Shape of the original input.
        keepdim (bool): Whether the forward reduction retained its dimension.

    Returns:
        torch.Tensor: Gradient with respect to the original input.
    """
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
    """Compute pairwise distances between rows of two tensors.

    Args:
        x1 (torch.Tensor): First tensor with rows in its final two dimensions.
        x2 (torch.Tensor): Second tensor with rows in its final two dimensions.
        p (float): Norm degree, including ``0`` and positive infinity.
        compute_mode (int, optional): Native ATen computation mode. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Pairwise distances with broadcast batch dimensions.

    Note:
        ``p=0``, ``p=inf``, and unequal batch shapes use an explicit reduction to avoid
        invalid Torch-MLIR lowering and incorrect generic-power results.
    """
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
    """Compute pairwise distances between rows of one tensor.

    Args:
        self (torch.Tensor): Two-dimensional floating-point input tensor.
        p (float, optional): Norm degree, including ``0`` and positive infinity. Defaults to
            ``2.0``.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Distances for the strict upper triangle in row-major order.

    Raises:
        NotImplementedError: If the input dtype is not float32 or float64.

    Note:
        ``p=0`` and ``p=inf`` use explicit per-pair reductions to avoid incorrect
        generic-power results; other values use the native ATen operation.
    """
    if p == 0:

        def reduce_p0(ad):
            """Count differing coordinates while propagating NaNs.

            Args:
                ad (torch.Tensor): Absolute coordinate differences for one row pair.

            Returns:
                torch.Tensor: Scalar count, or a NaN sum when any coordinate is NaN.
            """
            count = (ad != 0).to(self.dtype).sum()
            return torch.where(ad.isnan().any(), ad.sum(), count)

        return _pdist_pairwise(self, reduce_p0)
    if p == float("inf"):

        def reduce_pinf(ad):
            """Return the largest coordinate difference while propagating NaNs.

            Args:
                ad (torch.Tensor): Absolute coordinate differences for one row pair.

            Returns:
                torch.Tensor: Scalar maximum, or a NaN sum when any coordinate is NaN.
            """
            result = ad.amax()
            return torch.where(ad.isnan().any(), ad.sum(), result)

        return _pdist_pairwise(self, reduce_pinf)
    return torch.ops.aten._pdist_forward(self, p)
