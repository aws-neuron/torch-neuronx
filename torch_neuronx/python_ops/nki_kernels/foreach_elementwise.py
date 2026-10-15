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
    from torch_neuronx.utils import get_logical_neuron_cores

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
        """Apply scalar addition with an NKI kernel.

        Args:
            data (torch.Tensor): Input tensor.
            scalar_tensor (torch.Tensor): Scalar value tensor.

        Returns:
            torch.Tensor: Elementwise sum.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _add_scalar_kernel[(int(get_logical_neuron_cores()),)](
            data, _scalar_tensor(scalar_tensor), data.numel()
        )

    @nki_op("torch_neuronx::sub_scalar", mutates_args={})
    def sub_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        """Apply scalar subtraction with an NKI kernel.

        Args:
            data (torch.Tensor): Input tensor.
            scalar_tensor (torch.Tensor): Scalar value tensor.

        Returns:
            torch.Tensor: Elementwise difference.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _sub_scalar_kernel[(int(get_logical_neuron_cores()),)](
            data, _scalar_tensor(scalar_tensor), data.numel()
        )

    @nki_op("torch_neuronx::mul_scalar", mutates_args={})
    def mul_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        """Apply scalar multiplication with an NKI kernel.

        Args:
            data (torch.Tensor): Input tensor.
            scalar_tensor (torch.Tensor): Scalar value tensor.

        Returns:
            torch.Tensor: Elementwise product.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _mul_scalar_kernel[(int(get_logical_neuron_cores()),)](
            data, _scalar_tensor(scalar_tensor), data.numel()
        )

    @nki_op("torch_neuronx::div_scalar", mutates_args={})
    def div_scalar_nki(data: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
        """Apply scalar division with an NKI kernel.

        Args:
            data (torch.Tensor): Input tensor.
            scalar_tensor (torch.Tensor): Scalar divisor tensor.

        Returns:
            torch.Tensor: Elementwise quotient.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _div_scalar_kernel[(int(get_logical_neuron_cores()),)](
            data, _scalar_tensor(scalar_tensor), data.numel()
        )

    @nki_op("torch_neuronx::add_tensor", mutates_args={})
    def add_tensor_nki(
        data1: torch.Tensor, data2: torch.Tensor, alpha_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply scaled tensor addition with an NKI kernel.

        Args:
            data1 (torch.Tensor): Base tensor.
            data2 (torch.Tensor): Tensor to add.
            alpha_tensor (torch.Tensor): Scale factor tensor for ``data2``.

        Returns:
            torch.Tensor: Elementwise scaled sum.

        Note:
            ``data1`` must be nonempty and non-scalar.
        """
        return _add_tensor_kernel[(int(get_logical_neuron_cores()),)](
            data1, data2, _scalar_tensor(alpha_tensor), data1.numel()
        )

    @nki_op("torch_neuronx::sub_tensor", mutates_args={})
    def sub_tensor_nki(
        data1: torch.Tensor, data2: torch.Tensor, alpha_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply scaled tensor subtraction with an NKI kernel.

        Args:
            data1 (torch.Tensor): Base tensor.
            data2 (torch.Tensor): Tensor to subtract.
            alpha_tensor (torch.Tensor): Scale factor tensor for ``data2``.

        Returns:
            torch.Tensor: Elementwise scaled difference.

        Note:
            ``data1`` must be nonempty and non-scalar.
        """
        return _sub_tensor_kernel[(int(get_logical_neuron_cores()),)](
            data1, data2, _scalar_tensor(alpha_tensor), data1.numel()
        )

    @nki_op("torch_neuronx::mul_tensor", mutates_args={})
    def mul_tensor_nki(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """Apply tensor multiplication with an NKI kernel.

        Args:
            data1 (torch.Tensor): First input tensor.
            data2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Elementwise product.

        Note:
            ``data1`` must be nonempty and non-scalar.
        """
        return _mul_tensor_kernel[(int(get_logical_neuron_cores()),)](data1, data2, data1.numel())

    @nki_op("torch_neuronx::div_tensor", mutates_args={})
    def div_tensor_nki(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """Apply tensor division with an NKI kernel.

        Args:
            data1 (torch.Tensor): Dividend tensor.
            data2 (torch.Tensor): Divisor tensor.

        Returns:
            torch.Tensor: Elementwise quotient.

        Note:
            ``data1`` must be nonempty and non-scalar.
        """
        return _div_tensor_kernel[(int(get_logical_neuron_cores()),)](data1, data2, data1.numel())

    @nki_op("torch_neuronx::addcdiv", mutates_args={})
    def addcdiv_nki(
        data: torch.Tensor,
        data1: torch.Tensor,
        data2: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaled tensor division and addition with an NKI kernel.

        Args:
            data (torch.Tensor): Base tensor.
            data1 (torch.Tensor): Dividend tensor.
            data2 (torch.Tensor): Divisor tensor.
            value_tensor (torch.Tensor): Scale factor tensor for the quotient.

        Returns:
            torch.Tensor: Elementwise result of the scaled quotient and addition.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _addcdiv_kernel[(int(get_logical_neuron_cores()),)](
            data, data1, data2, _scalar_tensor(value_tensor), data.numel()
        )

    @nki_op("torch_neuronx::addcmul", mutates_args={})
    def addcmul_nki(
        data: torch.Tensor,
        data1: torch.Tensor,
        data2: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaled tensor multiplication and addition with an NKI kernel.

        Args:
            data (torch.Tensor): Base tensor.
            data1 (torch.Tensor): First multiplicand tensor.
            data2 (torch.Tensor): Second multiplicand tensor.
            value_tensor (torch.Tensor): Scale factor tensor for the product.

        Returns:
            torch.Tensor: Elementwise result of the scaled product and addition.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _addcmul_kernel[(int(get_logical_neuron_cores()),)](
            data, data1, data2, _scalar_tensor(value_tensor), data.numel()
        )

    @nki_op("torch_neuronx::lerp", mutates_args={})
    def lerp_nki(
        data: torch.Tensor,
        end: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply linear interpolation with an NKI kernel.

        Args:
            data (torch.Tensor): Starting tensor.
            end (torch.Tensor): Ending tensor.
            weight_tensor (torch.Tensor): Scalar interpolation weight tensor.

        Returns:
            torch.Tensor: Elementwise interpolated tensor.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _lerp_kernel[(int(get_logical_neuron_cores()),)](
            data, end, _scalar_tensor(weight_tensor), data.numel()
        )

    @nki_op("torch_neuronx::sqrt", mutates_args={})
    def sqrt_nki(data: torch.Tensor) -> torch.Tensor:
        """Apply square roots with an NKI kernel.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Elementwise square roots.

        Note:
            ``data`` must be nonempty and non-scalar.
        """
        return _sqrt_kernel[(int(get_logical_neuron_cores()),)](data, data.numel())

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
    """Apply a scalar operation to a tensor sequence.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalar (Scalar): Scalar operand.
        op (str, optional): Operation name. Defaults to ``"mul"``.

    Returns:
        tuple[torch.Tensor, ...]: Results cast to the first input tensor dtype.

    Note:
        Uses NKI when kernels are available for tensors with more than one element and a
        PyTorch fallback otherwise. Empty outputs alias their corresponding input tensors.
    """
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
    """Apply paired scalar operations to a tensor sequence.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalars (Sequence[Scalar]): Scalar operand for each tensor.
        op (str, optional): Operation name. Defaults to ``"mul"``.

    Returns:
        tuple[torch.Tensor, ...]: Results cast to their corresponding input dtypes.

    Note:
        Scalar-list NKI is disabled, so nonempty tensors use the PyTorch fallback.
        Empty outputs alias their inputs; pairing stops at the shorter sequence.
    """
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
    """Apply an operation between paired tensor sequences.

    Args:
        tensors1 (Sequence[torch.Tensor]): First input tensor sequence.
        tensors2 (Sequence[torch.Tensor]): Second input tensor sequence.
        op (str, optional): Operation name. Defaults to ``"mul"``.
        alpha (Scalar, optional): Scale for addition or subtraction. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, ...]: Results in the promoted input dtype.

    Note:
        Uses NKI for eligible equal-shaped tensors and an MLIR fallback otherwise.
        Empty outputs alias ``tensors1``; pairing stops at the shorter sequence.
    """
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
            # Cast back to out_dtype: `alpha * t2` can promote beyond out_dtype
            # (e.g. an int alpha on a bool tensor -> int64), which would widen a
            # bool result and wrongly pack it to shape (..., 2). Mirrors the NKI
            # branch's _cast_to_dtype below.
            result.append(
                _cast_to_dtype(mlir_ops[op](t1.to(out_dtype), t2.to(out_dtype), alpha), out_dtype)
            )
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
    """Apply foreach addition with one scalar.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalar (Scalar): Scalar to add.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise sums.

    Note:
        Empty outputs alias their corresponding input tensors.
    """
    return foreach_scalar_nki(tensors, scalar, "add")


def foreach_add_scalarlist_nki(tensors, scalars):
    """Apply foreach addition with paired scalars.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalars (Sequence[Scalar]): Scalar to add to each tensor.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise sums.

    Note:
        Empty outputs alias their inputs; pairing stops at the shorter sequence.
    """
    return foreach_scalarlist_nki(tensors, scalars, "add")


def foreach_add_list_nki(tensors1, tensors2, alpha=1):
    """Apply foreach addition between tensor sequences.

    Args:
        tensors1 (Sequence[torch.Tensor]): Base tensors.
        tensors2 (Sequence[torch.Tensor]): Tensors to add.
        alpha (Scalar, optional): Scale for ``tensors2``. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled sums.

    Note:
        Empty outputs alias ``tensors1``; pairing stops at the shorter sequence.
    """
    return foreach_list_nki(tensors1, tensors2, "add", alpha)


def foreach_sub_scalar_nki(tensors, scalar):
    """Apply foreach subtraction with one scalar.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalar (Scalar): Scalar to subtract.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise differences.

    Note:
        Empty outputs alias their corresponding input tensors.
    """
    return foreach_scalar_nki(tensors, scalar, "sub")


def foreach_sub_scalarlist_nki(tensors, scalars):
    """Apply foreach subtraction with paired scalars.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalars (Sequence[Scalar]): Scalar to subtract from each tensor.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise differences.

    Note:
        Empty outputs alias their inputs; pairing stops at the shorter sequence.
    """
    return foreach_scalarlist_nki(tensors, scalars, "sub")


def foreach_sub_list_nki(tensors1, tensors2, alpha=1):
    """Apply foreach subtraction between tensor sequences.

    Args:
        tensors1 (Sequence[torch.Tensor]): Base tensors.
        tensors2 (Sequence[torch.Tensor]): Tensors to subtract.
        alpha (Scalar, optional): Scale for ``tensors2``. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled differences.

    Note:
        Empty outputs alias ``tensors1``; pairing stops at the shorter sequence.
    """
    return foreach_list_nki(tensors1, tensors2, "sub", alpha)


def foreach_mul_scalar_nki(tensors, scalar):
    """Apply foreach multiplication with one scalar.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalar (Scalar): Scalar multiplier.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise products.

    Note:
        Empty outputs alias their corresponding input tensors.
    """
    return foreach_scalar_nki(tensors, scalar, "mul")


def foreach_mul_scalarlist_nki(tensors, scalars):
    """Apply foreach multiplication with paired scalars.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.
        scalars (Sequence[Scalar]): Scalar multiplier for each tensor.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise products.

    Note:
        Empty outputs alias their inputs; pairing stops at the shorter sequence.
    """
    return foreach_scalarlist_nki(tensors, scalars, "mul")


def foreach_mul_list_nki(tensors1, tensors2):
    """Apply foreach multiplication between tensor sequences.

    Args:
        tensors1 (Sequence[torch.Tensor]): First input tensors.
        tensors2 (Sequence[torch.Tensor]): Second input tensors.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise products.

    Note:
        Empty outputs alias ``tensors1``; pairing stops at the shorter sequence.
    """
    return foreach_list_nki(tensors1, tensors2, "mul")


def foreach_div_scalar_nki(tensors, scalar):
    """Apply foreach division by one scalar.

    Args:
        tensors (Sequence[torch.Tensor]): Dividend tensors.
        scalar (Scalar): Scalar divisor.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise quotients.

    Note:
        Empty outputs alias their corresponding input tensors.
    """
    return foreach_scalar_nki(tensors, scalar, "div")


def foreach_div_scalarlist_nki(tensors, scalars):
    """Apply foreach division by paired scalars.

    Args:
        tensors (Sequence[torch.Tensor]): Dividend tensors.
        scalars (Sequence[Scalar]): Scalar divisor for each tensor.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise quotients.

    Note:
        Empty outputs alias their inputs; pairing stops at the shorter sequence.
    """
    return foreach_scalarlist_nki(tensors, scalars, "div")


def foreach_div_list_nki(tensors1, tensors2):
    """Apply foreach division between tensor sequences.

    Args:
        tensors1 (Sequence[torch.Tensor]): Dividend tensors.
        tensors2 (Sequence[torch.Tensor]): Divisor tensors.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise quotients.

    Note:
        Empty outputs alias ``tensors1``; pairing stops at the shorter sequence.
    """
    return foreach_list_nki(tensors1, tensors2, "div")


# --- foreach_sqrt ---


def foreach_sqrt_nki(tensors):
    """Apply foreach square roots with an MLIR fallback.

    Args:
        tensors (Sequence[torch.Tensor]): Input tensors.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise square roots.

    Note:
        Uses NKI only for tensors containing more than one element. Empty outputs
        alias their corresponding input tensors.
    """
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
            vt = (
                v.to(dtype=compute_dtype, device=s.device)
                if isinstance(v, torch.Tensor)
                else torch.tensor(v, device=s.device, dtype=compute_dtype)
            ).reshape(1, 1)
            result.append(
                _cast_to_dtype(
                    nki_fn(s.to(compute_dtype), t1.to(compute_dtype), t2.to(compute_dtype), vt),
                    orig_dtype,
                )
            )
    return tuple(result)


def foreach_addcdiv_scalar_nki(self, tensor1, tensor2, value=1):
    """Apply foreach scaled division and addition with one scalar.

    Args:
        self (Sequence[torch.Tensor]): Base tensors.
        tensor1 (Sequence[torch.Tensor]): Dividend tensors.
        tensor2 (Sequence[torch.Tensor]): Divisor tensors.
        value (Scalar, optional): Quotient scale. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled quotients added to ``self``.

    Note:
        Empty outputs alias ``self``; pairing stops at the shortest tensor sequence.
    """
    _nki_fn = addcdiv_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalar_nki(self, tensor1, tensor2, value, _nki_fn, torch.div)


def foreach_addcdiv_scalarlist_nki(self, tensor1, tensor2, scalars):
    """Apply foreach scaled division and addition with paired scalars.

    Args:
        self (Sequence[torch.Tensor]): Base tensors.
        tensor1 (Sequence[torch.Tensor]): Dividend tensors.
        tensor2 (Sequence[torch.Tensor]): Divisor tensors.
        scalars (Sequence[Scalar]): Quotient scale for each tensor tuple.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled quotients added to ``self``.

    Note:
        Empty outputs alias ``self``; pairing stops at the shortest sequence.
    """
    _nki_fn = addcdiv_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalarlist_nki(self, tensor1, tensor2, scalars, _nki_fn, torch.div)


def foreach_addcmul_scalar_nki(self, tensor1, tensor2, value=1):
    """Apply foreach scaled multiplication and addition with one scalar.

    Args:
        self (Sequence[torch.Tensor]): Base tensors.
        tensor1 (Sequence[torch.Tensor]): First multiplicand tensors.
        tensor2 (Sequence[torch.Tensor]): Second multiplicand tensors.
        value (Scalar, optional): Product scale. Defaults to ``1``.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled products added to ``self``.

    Note:
        Empty outputs alias ``self``; pairing stops at the shortest tensor sequence.
    """
    _nki_fn = addcmul_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalar_nki(self, tensor1, tensor2, value, _nki_fn, torch.mul)


def foreach_addcmul_scalarlist_nki(self, tensor1, tensor2, scalars):
    """Apply foreach scaled multiplication and addition with paired scalars.

    Args:
        self (Sequence[torch.Tensor]): Base tensors.
        tensor1 (Sequence[torch.Tensor]): First multiplicand tensors.
        tensor2 (Sequence[torch.Tensor]): Second multiplicand tensors.
        scalars (Sequence[Scalar]): Product scale for each tensor tuple.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise scaled products added to ``self``.

    Note:
        Empty outputs alias ``self``; pairing stops at the shortest sequence.
    """
    _nki_fn = addcmul_nki if _has_nki_elementwise_kernels else None
    return _foreach_addcop_scalarlist_nki(self, tensor1, tensor2, scalars, _nki_fn, torch.mul)


# --- foreach_lerp ---


def foreach_lerp_scalar_nki(self, tensors1, weight):
    """Linearly interpolate tensor pairs with one scalar weight.

    Args:
        self (Sequence[torch.Tensor]): Starting tensors.
        tensors1 (Sequence[torch.Tensor]): Ending tensors.
        weight (Scalar): Shared interpolation weight.

    Returns:
        tuple[torch.Tensor, ...]: Interpolated tensors cast to the first input dtype.

    Note:
        Uses NKI for eligible tensors and an MLIR fallback otherwise. Empty outputs
        alias ``self``; pairing stops at the shorter tensor sequence.
    """
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
    """Linearly interpolate tensor pairs with paired scalar weights.

    Args:
        self (Sequence[torch.Tensor]): Starting tensors.
        tensors1 (Sequence[torch.Tensor]): Ending tensors.
        weights (Sequence[Scalar]): Interpolation weight for each tensor pair.

    Returns:
        tuple[torch.Tensor, ...]: Interpolated tensors cast to their input dtypes.

    Note:
        Uses NKI for eligible tensors and an MLIR fallback otherwise. Empty outputs
        alias ``self``; pairing stops at the shortest sequence.
    """
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
    """Linearly interpolate tensor pairs with tensor weights.

    Args:
        self (Sequence[torch.Tensor]): Starting tensors.
        tensors1 (Sequence[torch.Tensor]): Ending tensors.
        weights (Sequence[torch.Tensor]): Interpolation weights.

    Returns:
        tuple[torch.Tensor, ...]: Elementwise interpolated tensors.

    Note:
        Uses the MLIR fallback because the NKI kernel accepts only scalar weights.
        Empty outputs alias ``self``; pairing stops at the shortest sequence.
    """
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
