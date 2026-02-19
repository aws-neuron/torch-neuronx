"""
Neuron Memory Management API

Provides memory statistics and management functions
for monitoring device memory usage on Neuron hardware.
"""

from typing import Any

import torch


def _get_device_index(device: int | torch.device | None = None) -> int:
    """Get device index, defaulting to current device."""
    # Import here to avoid circular imports
    from torch_neuronx import _C

    if device is None:
        return _C._neuron_getCurrentDevice()
    if isinstance(device, int):
        return device
    # Handle torch.device objects
    if hasattr(device, "index") and device.index is not None:
        return device.index
    return _C._neuron_getCurrentDevice()


def memory_stats(device: int | torch.device | None = None) -> dict[str, Any]:
    """
    Return a dictionary of memory allocator statistics for a device.

    Args:
        device: Device to get statistics for. Defaults to current device.

    Returns:
        Dictionary containing:
        - allocated_bytes: Total memory obtained from NRT (dict with current/peak/allocated/freed)
        - reserved_bytes: Memory held in cache for reuse (dict, currently always 0)
        - active_bytes: Memory in active use by tensors (dict, currently equals allocated_bytes)
        - num_alloc_retries: Count of OOM retries after cache flush
        - num_ooms: Count of final OOM failures
        - num_tensor_frees: Count of tensor deallocations
        - allocation_requests: Total allocation requests

        Each byte stat dict contains:
        - current: Current value in bytes
        - peak: Peak value since last reset
        - allocated: Historical total bytes increased
        - freed: Historical total bytes decreased
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    return _C._get_memory_stats(device_index)


def memory_allocated(device: int | torch.device | None = None) -> int:
    """
    Return current memory occupied by tensors in bytes.

    This is the total memory currently held by the allocator,
    including both active tensors and cached blocks.

    Args:
        device: Device to query. Defaults to current device.

    Returns:
        Current allocated bytes as an integer.
    """
    stats = memory_stats(device)
    return stats["allocated_bytes"]["current"]


def max_memory_allocated(device: int | torch.device | None = None) -> int:
    """
    Return peak memory occupied by tensors in bytes since last reset.

    Args:
        device: Device to query. Defaults to current device.

    Returns:
        Peak allocated bytes as an integer.
    """
    stats = memory_stats(device)
    return stats["allocated_bytes"]["peak"]


def memory_reserved(device: int | torch.device | None = None) -> int:
    """
    Return current memory held in the cache in bytes.

    Note: Currently always returns 0 as caching is not implemented.

    Args:
        device: Device to query. Defaults to current device.

    Returns:
        Current reserved (cached) bytes as an integer.
    """
    stats = memory_stats(device)
    return stats["reserved_bytes"]["current"]


def max_memory_reserved(device: int | torch.device | None = None) -> int:
    """
    Return peak memory held in the cache in bytes since last reset.

    Note: Currently always returns 0 as caching is not implemented.

    Args:
        device: Device to query. Defaults to current device.

    Returns:
        Peak reserved (cached) bytes as an integer.
    """
    stats = memory_stats(device)
    return stats["reserved_bytes"]["peak"]


def reset_peak_memory_stats(device: int | torch.device | None = None) -> None:
    """
    Reset the peak memory statistics for a device.

    After calling this, max_memory_allocated() and max_memory_reserved()
    will return values relative to this point in time.

    Args:
        device: Device to reset stats for. Defaults to current device.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    _C._reset_peak_memory_stats(device_index)


def reset_accumulated_memory_stats(device: int | torch.device | None = None) -> None:
    """
    Reset the accumulated memory statistics for a device.

    This resets counters like num_alloc_retries and num_ooms.

    Args:
        device: Device to reset stats for. Defaults to current device.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    _C._reset_accumulated_memory_stats(device_index)


def memory_summary(device: int | torch.device | None = None, abbreviated: bool = False) -> str:
    """
    Return a human-readable summary of memory usage for a device.

    Args:
        device: Device to summarize. Defaults to current device.
        abbreviated: If True, return a shorter summary (not yet implemented).

    Returns:
        Formatted string showing memory statistics.
    """
    device_index = _get_device_index(device)
    stats = memory_stats(device_index)

    def format_bytes(b: int) -> str:
        """Format bytes with appropriate unit."""
        for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
            if abs(b) < 1024:
                return f"{b:,.0f} {unit}"
            b /= 1024
        return f"{b:,.0f} PiB"

    lines = [
        "=" * 75,
        f"  Neuron Memory Summary, Device {device_index}",
        "=" * 75,
        f"  Allocations: {stats['allocation_requests']:,}  |  "
        f"Frees: {stats['num_tensor_frees']:,}",
        f"  OOMs: {stats['num_ooms']:,}  |  Alloc Retries: {stats['num_alloc_retries']:,}",
        "=" * 75,
        "  Metric              | Current    | Peak       | Allocated  | Freed",
        "-" * 75,
    ]

    for name, key in [
        ("Allocated Memory", "allocated_bytes"),
        ("Reserved Memory", "reserved_bytes"),
        ("Active Memory", "active_bytes"),
    ]:
        s = stats[key]
        lines.append(
            f"  {name:<18} | {format_bytes(s['current']):>10} | {format_bytes(s['peak']):>10} "
            f"| {format_bytes(s['allocated']):>10} | {format_bytes(s['freed']):>10}"
        )

    lines.append("=" * 75)
    return "\n".join(lines)
