"""Static input validation helpers for Torch-MLIR operations."""

import torch


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
