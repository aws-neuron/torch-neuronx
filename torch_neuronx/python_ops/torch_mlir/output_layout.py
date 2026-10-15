"""Helpers for operation-specific output layout metadata."""

import torch
from torch._prims_common import suggest_memory_format


def channels_last_strides(shape, memory_format):
    """Compute dense channels-last strides for a supported shape.

    Args:
        shape (Sequence[int]): Tensor shape whose strides are computed.
        memory_format (torch.memory_format): Requested channels-last memory format.

    Returns:
        tuple[int, ...] | None: Dense strides for a matching 4-D or 5-D shape,
        or ``None`` when the format and rank are incompatible.
    """
    if memory_format == torch.channels_last and len(shape) == 4:
        _, c, h, w = shape
        return (h * w * c, 1, w * c, c)
    if memory_format == torch.channels_last_3d and len(shape) == 5:
        _, c, d, h, w = shape
        return (d * h * w * c, 1, h * w * c, w * c, c)
    return None


def suggest_output_memory_format(tensor):
    """Select the output memory format that preserves a tensor's dense layout.

    Args:
        tensor (torch.Tensor): Tensor whose layout is inspected.

    Returns:
        torch.memory_format: Channels-last format for a dense channels-last tensor,
        including ambiguous empty shapes, or contiguous format otherwise.
    """
    memory_format = suggest_memory_format(tensor)
    if memory_format in (torch.channels_last, torch.channels_last_3d):
        return memory_format
    if tensor.numel() != 0:
        return torch.contiguous_format

    for candidate in (torch.channels_last, torch.channels_last_3d):
        expected = channels_last_strides(tensor.shape, candidate)
        if expected is not None and tensor.stride() == expected:
            return candidate
    return torch.contiguous_format
