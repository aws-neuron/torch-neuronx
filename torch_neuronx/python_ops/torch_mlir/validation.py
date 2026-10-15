"""Static input validation helpers for Torch-MLIR operations."""

import torch


def aten_scalar_type_name(dtype):
    """Return the ATen scalar type name for a dtype.

    Args:
        dtype (torch.dtype): Data type to format.

    Returns:
        str: Scalar type name used in ATen CPU-kernel error messages.
    """
    return torch.empty((), dtype=dtype).type().rsplit(".", 1)[-1].removesuffix("Tensor")


def validate_inplace_broadcast(destination, *operands, **_kwargs):
    """Validate that broadcasting does not expand an in-place destination.

    Args:
        destination (Any): In-place destination to validate when it is a tensor.
        *operands (Any): Additional operands participating in broadcasting.
        **_kwargs (Any): Ignored operation keyword arguments.

    Raises:
        RuntimeError: If the broadcast shape differs from the destination shape.
    """
    if not isinstance(destination, torch.Tensor):
        return
    tensors = [tensor for tensor in (destination, *operands) if isinstance(tensor, torch.Tensor)]
    if len(tensors) < 2:
        return
    try:
        broadcast_shape = torch.broadcast_shapes(*[tensor.shape for tensor in tensors])
    except RuntimeError:
        return
    if tuple(broadcast_shape) != tuple(destination.shape):
        raise RuntimeError(
            f"output with shape {list(destination.shape)} doesn't match "
            f"the broadcast shape {list(broadcast_shape)}"
        )


# round() is an identity on integer inputs but is undefined for bool; the
# decimals overload and frac() are defined only for floating-point inputs.
# The passthrough would otherwise accept these dtypes that ATen rejects, so
# guard them to keep Neuron's supported-dtype set aligned with ATen/OpInfo.
_ROUND_UNSUPPORTED_DTYPES = (torch.bool,)
_FLOAT_ONLY_UNSUPPORTED_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def validate_round_dtype(self, *_operands, **_kwargs):
    """Validate that ``round()`` does not receive a Boolean tensor.

    Args:
        self (Any): Value whose data type is validated when it is a tensor.
        *_operands (Any): Ignored positional operands.
        **_kwargs (Any): Ignored keyword arguments.

    Raises:
        RuntimeError: If ``self`` is a Boolean tensor.

    Note:
        ATen implements ``round()`` only for integer and floating-point tensors.
    """
    if isinstance(self, torch.Tensor) and self.dtype in _ROUND_UNSUPPORTED_DTYPES:
        raise RuntimeError(f'"round" not implemented for {self.dtype}')


def validate_float_only_dtype(self, *_operands, **_kwargs):
    """Reject Boolean and integer tensors for round and fractional operations.

    Args:
        self (Any): Value whose data type is validated when it is a tensor.
        *_operands (Any): Ignored positional operands.
        **_kwargs (Any): Ignored keyword arguments.

    Raises:
        RuntimeError: If ``self`` is a Boolean or integer tensor.

    Note:
        This validator checks Boolean and integer dtypes; other dtype handling occurs
        during operation lowering or execution.
    """
    if isinstance(self, torch.Tensor) and self.dtype in _FLOAT_ONLY_UNSUPPORTED_DTYPES:
        raise RuntimeError(f'"round/frac" not implemented for {self.dtype}')


def check_broadcast_shapes(t1, t2):
    """Reject tensors whose shapes cannot broadcast together.

    Args:
        t1 (torch.Tensor): First tensor to compare.
        t2 (torch.Tensor): Second tensor to compare.

    Raises:
        RuntimeError: If the tensor shapes are not broadcastable.

    Note:
        The error uses TensorIterator wording instead of the lowering's message.
    """
    if t1.shape == t2.shape:
        return
    try:
        torch.broadcast_shapes(t1.shape, t2.shape)
    except RuntimeError:
        for dim in range(max(t1.ndim, t2.ndim)):
            d1 = t1.shape[-(dim + 1)] if dim < t1.ndim else 1
            d2 = t2.shape[-(dim + 1)] if dim < t2.ndim else 1
            if d1 != d2 and d1 != 1 and d2 != 1:
                actual_dim = max(t1.ndim, t2.ndim) - 1 - dim
                raise RuntimeError(
                    f"The size of tensor a ({d1}) must match the size of "
                    f"tensor b ({d2}) at non-singleton dimension {actual_dim}"
                ) from None
        raise


def is_bool_operand(operand):
    """Check whether an operand or nested sequence contains a Boolean value.

    Args:
        operand (Any): Tensor, scalar, or nested list or tuple to inspect.

    Returns:
        bool: ``True`` for a Boolean tensor, Boolean scalar, or nested Boolean value.
    """
    if isinstance(operand, torch.Tensor):
        return operand.dtype == torch.bool
    if isinstance(operand, bool):
        return True
    if isinstance(operand, int | float | complex):
        return False
    if isinstance(operand, list | tuple):
        return any(is_bool_operand(x) for x in operand)
    return False


def check_bool_sub(*operands):
    """Reject subtraction involving Boolean operands or an empty operand list.

    Args:
        *operands (Any): Tensor, scalar, or nested operands to inspect.

    Raises:
        RuntimeError: If any operand is Boolean or if no operands are provided.

    Note:
        All-Boolean and mixed-Boolean operands use the distinct ATen ``sub_check``
        error messages.
    """
    is_bool = [is_bool_operand(o) for o in operands]
    if all(is_bool):
        raise RuntimeError(
            "Subtraction, the `-` operator, with two bool tensors is not supported. "
            "Use the `^` or `logical_xor()` operator instead."
        )
    if any(is_bool):
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        )


def check_lerp_dtype(start, weight):
    """Reject Boolean interpolation inputs unsupported by ATen.

    Args:
        start (Any): Interpolation start value to validate when it is a tensor.
        weight (Any): Weight used to select the tensor or scalar kernel name.

    Raises:
        NotImplementedError: If ``start`` is a Boolean tensor.

    Note:
        The error names the tensor or scalar ATen kernel according to ``weight``.
    """
    if isinstance(start, torch.Tensor) and start.dtype == torch.bool:
        kernel = "lerp_kernel_tensor" if isinstance(weight, torch.Tensor) else "lerp_kernel_scalar"
        raise NotImplementedError(f"\"{kernel}\" not implemented for 'Bool'")


# --- out= validation ---
#
# ATen checks the out= tensor before it runs the kernel. The Neuron lowering
# writes into whatever buffer it gets, so it accepts an out= that ATen rejects
# and returns a result where CPU raises. The validators below restore that
# check. Each message repeats the ATen wording, because the upstream tests
# match on it.


def _out_tensor(out):
    """The values tensor of an out= argument, or None.

    ``out`` is a tensor, or a tuple for an op that returns values and indices.
    """
    tensor = out[0] if isinstance(out, tuple | list) and out else out
    return tensor if isinstance(tensor, torch.Tensor) else None


def make_non_boolean_out_validator(op_name):
    """Create an output validator for a Boolean predicate.

    Args:
        op_name (str): Operation name included in validation errors.

    Returns:
        Callable: Validator that rejects non-Boolean ``out`` tensors.
    """

    def _validate(_self, *_args, out=None, **_kwargs):
        out_tensor = _out_tensor(out)
        if out_tensor is not None and out_tensor.dtype != torch.bool:
            raise RuntimeError(f"{op_name} does not support non-boolean outputs.")

    return _validate


# ATen still accepts uint8 for a bool reduction result, for backward compatibility.
_BOOL_RESULT_DTYPES = (torch.bool, torch.uint8)


def make_bool_result_out_validator(op_name):
    """Create an output validator for a Boolean reduction.

    Args:
        op_name (str): Operation name included in validation errors.

    Returns:
        Callable: Validator accepting only Boolean or uint8 ``out`` tensors.
    """

    def _validate(_self, *_args, out=None, **_kwargs):
        out_tensor = _out_tensor(out)
        if out_tensor is not None and out_tensor.dtype not in _BOOL_RESULT_DTYPES:
            raise RuntimeError(
                f"{op_name} only supports bool tensor for result, got: "
                f"{aten_scalar_type_name(out_tensor.dtype)}"
            )

    return _validate


def validate_out_same_device(self, *_args, out=None, **_kwargs):
    """Reject an output tensor on a different device type.

    Args:
        self (Any): Input value whose device type is authoritative when it is a tensor.
        *_args (Any): Ignored positional operation arguments.
        out (Any, optional): Output tensor or output sequence. Defaults to ``None``.
        **_kwargs (Any): Ignored keyword operation arguments.

    Raises:
        RuntimeError: If tensor input and output device types differ.
    """
    out_tensor = _out_tensor(out)
    if out_tensor is None or not isinstance(self, torch.Tensor):
        return
    if out_tensor.device.type != self.device.type:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices, "
            f"{self.device} and {out_tensor.device}!"
        )


def validate_true_divide_out(x, y=None, rounding_mode=None, out=None, **_kwargs):
    """Reject an output tensor that cannot hold a true-division result.

    Args:
        x (Any): Dividend used for result-type inference.
        y (Any, optional): Divisor used for result-type inference. Defaults to ``None``.
        rounding_mode (str, optional): Rounding mode; any non-``None`` value bypasses
            true-division validation. Defaults to ``None``.
        out (Any, optional): Output tensor or output sequence. Defaults to ``None``.
        **_kwargs (Any): Ignored keyword operation arguments.

    Raises:
        RuntimeError: If ``out`` cannot hold the inferred true-division dtype.

    Note:
        ``torch.can_cast`` follows ATen, so a complex output may hold a real result.
    """
    out_tensor = _out_tensor(out)
    if out_tensor is None or rounding_mode is not None:
        return
    result = torch.result_type(x, y)
    if not result.is_floating_point and not result.is_complex:
        result = torch.get_default_dtype()
    if torch.can_cast(result, out_tensor.dtype):
        return
    raise RuntimeError(
        f"result type {aten_scalar_type_name(result)} can't be cast to the desired "
        f"output type {aten_scalar_type_name(out_tensor.dtype)}"
    )
