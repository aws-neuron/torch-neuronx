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
