"""Helpers for operation-specific output layout metadata."""

import torch
from torch._prims_common import suggest_memory_format


def channels_last_strides(shape, memory_format):
    """Return dense channels-last strides for a 4-D or 5-D shape."""
    if memory_format == torch.channels_last and len(shape) == 4:
        _, c, h, w = shape
        return (h * w * c, 1, w * c, c)
    if memory_format == torch.channels_last_3d and len(shape) == 5:
        _, c, d, h, w = shape
        return (d * h * w * c, 1, h * w * c, w * c, c)
    return None


def suggest_output_memory_format(tensor):
    """Preserve exact dense channels-last strides when empty shapes are ambiguous."""
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
