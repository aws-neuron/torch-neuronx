"""Static input validation helpers for Torch-MLIR operations."""

import torch


def aten_scalar_type_name(dtype):
    """ATen ScalarType spelling of a dtype (e.g. 'Long', not 'torch.int64'), for
    "not implemented for '<dtype>'" messages matching the CPU kernels."""
    return torch.empty((), dtype=dtype).type().rsplit(".", 1)[-1].removesuffix("Tensor")


def validate_inplace_broadcast(destination, *operands, **_kwargs):
    """Reject broadcasts that would expand an in-place destination."""
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
    """Reject bool for round() (ATen implements it for int/float only)."""
    if isinstance(self, torch.Tensor) and self.dtype in _ROUND_UNSUPPORTED_DTYPES:
        raise RuntimeError(f'"round" not implemented for {self.dtype}')


def validate_float_only_dtype(self, *_operands, **_kwargs):
    """Reject bool/integer inputs for round(decimals=) and frac() (float-only)."""
    if isinstance(self, torch.Tensor) and self.dtype in _FLOAT_ONLY_UNSUPPORTED_DTYPES:
        raise RuntimeError(f'"round/frac" not implemented for {self.dtype}')


def check_broadcast_shapes(t1, t2):
    """Reject a non-broadcastable pair with the message that TensorIterator raises.

    Neuron reports a broadcast failure from the lowering, with different wording.
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
    """True when an operand is bool. An operand is a tensor, a number, or a list."""
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
    """Reject subtraction that involves a bool operand, the same as Tensor.sub.

    Each operand is one argument slot. Two bool operands and one bool operand get
    the two different messages that ``sub_check`` in ATen ``BinaryOps.h`` raises.
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
    """Reject bool for lerp, which has no bool kernel in ATen.

    The kernel name in the message follows the weight form, the same as ATen.
    """
    if isinstance(start, torch.Tensor) and start.dtype == torch.bool:
        kernel = "lerp_kernel_tensor" if isinstance(weight, torch.Tensor) else "lerp_kernel_scalar"
        raise NotImplementedError(f"\"{kernel}\" not implemented for 'Bool'")
