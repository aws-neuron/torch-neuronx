"""NKI kernels for foreach elementwise operations - generalized."""

import torch

try:
    from nkilib.experimental.foreach import (
        add_scalar_kernel,
        add_tensor_kernel,
        addcdiv_kernel,
        addcmul_kernel,
        div_scalar_kernel,
        div_tensor_kernel,
        lerp_kernel,
        mul_scalar_kernel,
        mul_tensor_kernel,
        sqrt_kernel,
        sub_scalar_kernel,
        sub_tensor_kernel,
    )

    from torch_neuronx.nki_hop import nki_op, wrap_nki

    _add_scalar_kernel = wrap_nki(add_scalar_kernel)
    _sub_scalar_kernel = wrap_nki(sub_scalar_kernel)
    _mul_scalar_kernel = wrap_nki(mul_scalar_kernel)
    _div_scalar_kernel = wrap_nki(div_scalar_kernel)
    _add_tensor_kernel = wrap_nki(add_tensor_kernel)
    _sub_tensor_kernel = wrap_nki(sub_tensor_kernel)
    _mul_tensor_kernel = wrap_nki(mul_tensor_kernel)
    _div_tensor_kernel = wrap_nki(div_tensor_kernel)
    _addcdiv_kernel = wrap_nki(addcdiv_kernel)
    _addcmul_kernel = wrap_nki(addcmul_kernel)
    _lerp_kernel = wrap_nki(lerp_kernel)
    _sqrt_kernel = wrap_nki(sqrt_kernel)

    P_SIZE = 128

    def _scalar_tensor(t):
        """Expand scalar tensor to (P_SIZE, 1) for kernel input."""
        return t.expand(P_SIZE, 1).contiguous()

    # NKI op wrappers
    # Note: NKI kernels require numel >= 1 and a non-scalar (ndim >= 1) input tensor.
    # Callers must handle 0-dim scalar tensors before invoking these ops.
    @nki_op("torch_neuronx::add_scalar", mutates_args={})
    def add_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        return _add_scalar_kernel[(2,)](data, _scalar_tensor(scalar_tensor), data.numel())

    @nki_op("torch_neuronx::sub_scalar", mutates_args={})
    def sub_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        return _sub_scalar_kernel[(2,)](data, _scalar_tensor(scalar_tensor), data.numel())

    @nki_op("torch_neuronx::mul_scalar", mutates_args={})
    def mul_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        return _mul_scalar_kernel[(2,)](data, _scalar_tensor(scalar_tensor), data.numel())

    @nki_op("torch_neuronx::div_scalar", mutates_args={})
    def div_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        return _div_scalar_kernel[(2,)](data, _scalar_tensor(scalar_tensor), data.numel())

    @nki_op("torch_neuronx::add_tensor", mutates_args={})
    def add_tensor_nki(
        data1: torch.Tensor, data2: torch.Tensor, alpha_tensor: torch.Tensor
    ) -> torch.Tensor:
        return _add_tensor_kernel[(2,)](data1, data2, _scalar_tensor(alpha_tensor), data1.numel())

    @nki_op("torch_neuronx::sub_tensor", mutates_args={})
    def sub_tensor_nki(
        data1: torch.Tensor, data2: torch.Tensor, alpha_tensor: torch.Tensor
    ) -> torch.Tensor:
        return _sub_tensor_kernel[(2,)](data1, data2, _scalar_tensor(alpha_tensor), data1.numel())

    @nki_op("torch_neuronx::mul_tensor", mutates_args={})
    def mul_tensor_nki(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return _mul_tensor_kernel[(2,)](data1, data2, data1.numel())

    @nki_op("torch_neuronx::div_tensor", mutates_args={})
    def div_tensor_nki(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return _div_tensor_kernel[(2,)](data1, data2, data1.numel())

    @nki_op("torch_neuronx::addcdiv", mutates_args={})
    def addcdiv_nki(
        data: torch.Tensor,
        data1: torch.Tensor,
        data2: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return _addcdiv_kernel[(2,)](data, data1, data2, _scalar_tensor(value_tensor), data.numel())

    @nki_op("torch_neuronx::addcmul", mutates_args={})
    def addcmul_nki(
        data: torch.Tensor,
        data1: torch.Tensor,
        data2: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return _addcmul_kernel[(2,)](data, data1, data2, _scalar_tensor(value_tensor), data.numel())

    @nki_op("torch_neuronx::lerp", mutates_args={})
    def lerp_nki(
        data: torch.Tensor,
        end: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return _lerp_kernel[(2,)](data, end, _scalar_tensor(weight_tensor), data.numel())

    @nki_op("torch_neuronx::sqrt", mutates_args={})
    def sqrt_nki(data: torch.Tensor) -> torch.Tensor:
        return _sqrt_kernel[(2,)](data, data.numel())

    SCALAR_NKI_OPS = {
        "add": add_scalar_nki,
        "sub": sub_scalar_nki,
        "mul": mul_scalar_nki,
        "div": div_scalar_nki,
    }

    TENSOR_NKI_OPS = {
        "add": add_tensor_nki,
        "sub": sub_tensor_nki,
        "mul": mul_tensor_nki,
        "div": div_tensor_nki,
    }

    _has_nki_elementwise_kernels = True
except (ImportError, ModuleNotFoundError):
    _has_nki_elementwise_kernels = False

# Disable NKI for ScalarList variants due to compiler lnc sync caused perf issues
DISABLE_NKI_SCALARLIST = True

# Torch fallback ops
TORCH_OPS = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
}


def _cast_to_dtype(t: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to target dtype, using truncation toward zero for float→int.

    Required because the standard upcast mechanism relies on `inferred_output_dtypes`
    populated by Meta dispatch on the op's fake impl. NKI ops register their kernel
    function as the fake (via `@nki_op` → `register_fake(fn)`), so Meta sees the
    kernel's actual post-downcast output dtype (e.g., int32 for int64 inputs), not
    the op's semantic output dtype.

    Handles two scenarios:
    1. Type promotion reversal: computation was promoted (e.g., int8→float32 for
       int8+float scalar), now cast back to original int dtype using trunc().
    2. NKI downcast reversal: NKI kernel returned int32/float32 for
       int64/float64 inputs, now cast back to the expected dtype.
    """
    if t.dtype == target_dtype:
        return t
    if not target_dtype.is_floating_point and t.dtype.is_floating_point:
        return t.trunc().to(target_dtype)
    return t.to(target_dtype)


# Generic foreach wrappers
def foreach_scalar_nki(tensors, scalar, op="mul"):
    """Foreach op with scalar."""
    orig_dtype = tensors[0].dtype
    compute_dtype = torch.result_type(tensors[0], scalar)
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar, device=tensors[0].device, dtype=compute_dtype)
    scalar_tensor = scalar.reshape(1, 1).to(compute_dtype)
    torch_op = TORCH_OPS[op]
    nki_op_fn = SCALAR_NKI_OPS[op] if _has_nki_elementwise_kernels else None

    result = []
    for t in tensors:
        if t.numel() == 0:
            result.append(t)
        elif t.numel() == 1 or not _has_nki_elementwise_kernels:
            result.append(_cast_to_dtype(torch_op(t.to(compute_dtype), scalar), orig_dtype))
        else:
            output = nki_op_fn(t.to(compute_dtype), scalar_tensor)
            result.append(_cast_to_dtype(output, orig_dtype))
    return tuple(result)


def foreach_scalarlist_nki(tensors, scalars, op="mul"):
    """Foreach op with scalar list."""
    torch_op = TORCH_OPS[op]
    nki_op_fn = SCALAR_NKI_OPS[op] if _has_nki_elementwise_kernels else None

    result = []
    for t, s in zip(tensors, scalars, strict=False):
        orig_dtype = t.dtype
        compute_dtype = torch.result_type(t, s)

        if t.numel() == 0:
            result.append(t)
        elif t.numel() == 1 or not _has_nki_elementwise_kernels or DISABLE_NKI_SCALARLIST:
            result.append(_cast_to_dtype(torch_op(t.to(compute_dtype), s), orig_dtype))
        else:
            scalar_tensor = torch.tensor(s, device=t.device, dtype=compute_dtype).reshape(1, 1)
            output = nki_op_fn(t.to(compute_dtype), scalar_tensor)
            result.append(_cast_to_dtype(output, orig_dtype))
    return tuple(result)


def foreach_list_nki(tensors1, tensors2, op="mul", alpha=1):
    """Foreach op between tensor lists."""
    needs_alpha = op in ("add", "sub")

    # Lambda ops for small tensors / MLIR fallback
    mlir_ops = {
        "add": lambda t1, t2, a: t1 + a * t2,
        "sub": lambda t1, t2, a: t1 - a * t2,
        "mul": lambda t1, t2, a: t1 * t2,
        "div": lambda t1, t2, a: t1 / t2,
    }

    nki_op_fn = TENSOR_NKI_OPS[op] if _has_nki_elementwise_kernels else None

    alpha_tensor = None
    if _has_nki_elementwise_kernels and needs_alpha and tensors1:
        if isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.reshape(1, 1)
        else:
            alpha_tensor = torch.tensor(alpha, device=tensors1[0].device).reshape(1, 1)

    result = []
    for t1, t2 in zip(tensors1, tensors2, strict=False):
        if t1.numel() == 0:
            result.append(t1)
        elif t1.numel() == 1 or not _has_nki_elementwise_kernels or t1.shape != t2.shape:
            out_dtype = torch.promote_types(t1.dtype, t2.dtype)
            result.append(mlir_ops[op](t1.to(out_dtype), t2.to(out_dtype), alpha))
        else:
            out_dtype = torch.promote_types(t1.dtype, t2.dtype)
            d1, d2 = t1.to(out_dtype), t2.to(out_dtype)
            if needs_alpha:
                output = nki_op_fn(d1, d2, alpha_tensor.to(out_dtype))
            else:
                output = nki_op_fn(d1, d2)
            # NKI kernel's fake impl may return a different dtype (e.g., int32
            # for int64 inputs since NKI doesn't support int64 natively).
            result.append(_cast_to_dtype(output, out_dtype))
    return tuple(result)


# Convenience functions for each op
def foreach_add_scalar_nki(tensors, scalar):
    return foreach_scalar_nki(tensors, scalar, "add")


def foreach_add_scalarlist_nki(tensors, scalars):
    return foreach_scalarlist_nki(tensors, scalars, "add")


def foreach_add_list_nki(tensors1, tensors2, alpha=1):
    return foreach_list_nki(tensors1, tensors2, "add", alpha)


def foreach_sub_scalar_nki(tensors, scalar):
    return foreach_scalar_nki(tensors, scalar, "sub")


def foreach_sub_scalarlist_nki(tensors, scalars):
    return foreach_scalarlist_nki(tensors, scalars, "sub")


def foreach_sub_list_nki(tensors1, tensors2, alpha=1):
    return foreach_list_nki(tensors1, tensors2, "sub", alpha)


def foreach_mul_scalar_nki(tensors, scalar):
    return foreach_scalar_nki(tensors, scalar, "mul")


def foreach_mul_scalarlist_nki(tensors, scalars):
    return foreach_scalarlist_nki(tensors, scalars, "mul")


def foreach_mul_list_nki(tensors1, tensors2):
    return foreach_list_nki(tensors1, tensors2, "mul")


def foreach_div_scalar_nki(tensors, scalar):
    return foreach_scalar_nki(tensors, scalar, "div")


def foreach_div_scalarlist_nki(tensors, scalars):
    return foreach_scalarlist_nki(tensors, scalars, "div")


def foreach_div_list_nki(tensors1, tensors2):
    return foreach_list_nki(tensors1, tensors2, "div")


# --- foreach_sqrt ---


def foreach_sqrt_nki(tensors):
    """Foreach sqrt with MLIR fallback."""
    result = []
    for t in tensors:
        if t.numel() == 0:
            result.append(t)
        elif t.numel() == 1 or not _has_nki_elementwise_kernels:
            result.append(torch.sqrt(t))
        else:
            result.append(sqrt_nki(t))
    return tuple(result)


# --- foreach_addcdiv / foreach_addcmul ---


def _foreach_addcop_scalar_nki(self, tensor1, tensor2, value, nki_fn, torch_combine_fn):
    """Foreach addcdiv/addcmul with scalar value, with MLIR fallback."""
    orig_dtype = self[0].dtype
    compute_dtype = torch.result_type(self[0], value)

    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=self[0].device, dtype=compute_dtype)
    value_tensor = value.reshape(1, 1).to(compute_dtype)

    result = []
    for s, t1, t2 in zip(self, tensor1, tensor2, strict=False):
        if s.numel() == 0:
            result.append(s)
        elif (
            s.numel() == 1
            or not _has_nki_elementwise_kernels
            or s.shape != t1.shape
            or s.shape != t2.shape
        ):
            result.append(
                _cast_to_dtype(
                    s.to(compute_dtype)
                    + value * torch_combine_fn(t1.to(compute_dtype), t2.to(compute_dtype)),
                    orig_dtype,
                )
            )
        else:
            result.append(
                _cast_to_dtype(
                    nki_fn(
                        s.to(compute_dtype),
                        t1.to(compute_dtype),
                        t2.to(compute_dtype),
                        value_tensor,
                    ),
                    orig_dtype,
                )
            )
    return tuple(result)


def _foreach_addcop_scalarlist_nki(self, tensor1, tensor2, scalars, nki_fn, torch_combine_fn):
    """Foreach addcdiv/addcmul with scalar list, with MLIR fallback."""
    result = []
    for s, t1, t2, v in zip(self, tensor1, tensor2, scalars, strict=False):
        orig_dtype = s.dtype
        compute_dtype = torch.result_type(s, v)

        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32

        if s.numel() == 0:
            result.append(s)
        elif (
            s.numel() == 1
            or not _has_nki_elementwise_kernels
            or s.shape != t1.shape
            or s.shape != t2.shape
        ):
            result.append(
                _cast_to_dtype(
                    s.to(compute_dtype)
                    + v * torch_combine_fn(t1.to(compute_dtype), t2.to(compute_dtype)),
                    orig_dtype,
                )
            )
        else:
            vt = torch.tensor(v, device=s.device, dtype=compute_dtype).reshape(1, 1)
            result.append(
                _cast_to_dtype(
                    nki_fn(s.to(compute_dtype), t1.to(compute_dtype), t2.to(compute_dtype), vt),
                    orig_dtype,
                )
            )
    return tuple(result)


def foreach_addcdiv_scalar_nki(self, tensor1, tensor2, value=1):
    _nki_fn = addcdiv_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalar_nki(self, tensor1, tensor2, value, _nki_fn, torch.div)


def foreach_addcdiv_scalarlist_nki(self, tensor1, tensor2, scalars):
    _nki_fn = addcdiv_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalarlist_nki(self, tensor1, tensor2, scalars, _nki_fn, torch.div)


def foreach_addcmul_scalar_nki(self, tensor1, tensor2, value=1):
    _nki_fn = addcmul_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalar_nki(self, tensor1, tensor2, value, _nki_fn, torch.mul)


def foreach_addcmul_scalarlist_nki(self, tensor1, tensor2, scalars):
    _nki_fn = addcmul_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalarlist_nki(self, tensor1, tensor2, scalars, _nki_fn, torch.mul)


# --- foreach_lerp ---


def foreach_lerp_scalar_nki(self, tensors1, weight):
    """Foreach lerp with scalar weight, with MLIR fallback."""
    orig_dtype = self[0].dtype
    compute_dtype = torch.result_type(self[0], weight)

    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight, device=self[0].device, dtype=compute_dtype)
    weight_tensor = weight.reshape(1, 1).to(compute_dtype)

    _nki_fn = lerp_nki if _has_nki_elementwise_kernels else None
    result = []
    for s, t in zip(self, tensors1, strict=False):
        if s.numel() == 0:
            result.append(s)
        elif s.numel() == 1 or not _has_nki_elementwise_kernels:
            result.append(
                _cast_to_dtype(
                    torch.lerp(s.to(compute_dtype), t.to(compute_dtype), weight), orig_dtype
                )
            )
        else:
            result.append(
                _cast_to_dtype(
                    _nki_fn(s.to(compute_dtype), t.to(compute_dtype), weight_tensor), orig_dtype
                )
            )
    return tuple(result)


def foreach_lerp_scalarlist_nki(self, tensors1, weights):
    """Foreach lerp with scalar list weights, with MLIR fallback."""
    _nki_fn = lerp_nki if _has_nki_elementwise_kernels else None
    result = []
    for s, t, w in zip(self, tensors1, weights, strict=False):
        orig_dtype = s.dtype
        compute_dtype = torch.result_type(s, w)

        if s.numel() == 0:
            result.append(s)
        elif s.numel() == 1 or not _has_nki_elementwise_kernels:
            result.append(
                _cast_to_dtype(torch.lerp(s.to(compute_dtype), t.to(compute_dtype), w), orig_dtype)
            )
        else:
            wt = torch.tensor(w, device=s.device, dtype=compute_dtype).reshape(1, 1)
            result.append(
                _cast_to_dtype(_nki_fn(s.to(compute_dtype), t.to(compute_dtype), wt), orig_dtype)
            )
    return tuple(result)


def foreach_lerp_list_nki(self, tensors1, weights):
    """Foreach lerp with tensor list weights, with MLIR fallback."""
    # Weight tensors are per-element tensors, use scalar NKI kernel per pair
    _nki_fn = lerp_nki if _has_nki_elementwise_kernels else None
    result = []
    for s, t, w in zip(self, tensors1, weights, strict=False):
        if s.numel() == 0:
            result.append(s)
        elif s.numel() == 1 or not _has_nki_elementwise_kernels:
            result.append(torch.lerp(s, t, w))
        else:
            # lerp_kernel expects a scalar weight; for tensor weights we fall back to MLIR
            result.append(torch.lerp(s, t, w))
    return tuple(result)
