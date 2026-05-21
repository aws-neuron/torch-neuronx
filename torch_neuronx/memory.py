"""
Neuron Memory Management API

Provides memory statistics and management functions
for monitoring device memory usage on Neuron hardware.
"""

import sys
from typing import Any

import torch


def _get_device_index(device: int | torch.device | None = None) -> int:
    """Get device index, defaulting to current device."""
    # Import here to avoid circular imports
    import torch_neuronx

    return torch_neuronx._get_device_index(device, optional=False)


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


def peak_memory_stats(device: int | torch.device | None = None) -> dict[str, Any]:
    """
    Return the memory stats snapshot captured at the moment allocated_bytes
    last exceeded its previous peak. Automatically dumped on OOM.

    Returns:
        Dictionary containing:
        - valid: Whether a peak snapshot has been captured
        - timestamp_ns: When the peak was recorded
        - allocated_bytes: Block sizes at peak (dict with current/peak/allocated/freed)
        - reserved_bytes: Segment sizes at peak
        - active_bytes: Requested sizes at peak
        - internal_fragmentation: allocated_bytes.current - active_bytes.current at peak
        - num_alloc_retries, num_ooms, num_tensor_frees, allocation_requests
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    return _C._get_peak_memory_stats(device_index)


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


# =============================================================================
# Memory Snapshot API
# =============================================================================


def _record_memory_history(
    enabled: bool | str | None = "all",
    *,
    context: str | None = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: int | torch.device | None = None,
) -> None:
    """
    Enable or disable recording of memory allocation/deallocation history.

    When enabled, the allocator will record a trace of memory events that can
    be retrieved with _snapshot(). This is useful for debugging memory issues.

    Args:
        enabled: Controls recording. Can be:
            - True or "all": Enable recording
            - False, None, or "none": Disable recording
        context: When to capture stack traces. One of:
            - "all": Capture for all events (alloc and free)
            - "alloc": Capture only for allocation events
            - "state": Capture for current state only
            - "never" or None: Don't capture stack traces
        stacks: Stack trace mode. One of:
            - "all": Include C++ frames (not yet implemented)
            - "python": Python frames only (default behavior)
        max_entries: Maximum number of trace entries to keep in the ring buffer.
            Older entries are overwritten when the buffer is full.
        device: Device to configure (currently applies globally).

    Example:
        >>> import torch_neuronx
        >>> torch_neuronx.memory._record_memory_history(True, context="all")
        >>> # ... perform allocations ...
        >>> snapshot = torch_neuronx.memory._snapshot()
        >>> torch_neuronx.memory._record_memory_history(None)  # Disable
    """
    _record_memory_history_impl(
        enabled=enabled,
        context=context,
        stacks=stacks,
        max_entries=max_entries,
    )


def _record_memory_history_impl(
    enabled: bool | str | None = "all",
    context: str | None = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
) -> None:
    """
    Internal implementation of _record_memory_history.
    """
    from torch_neuronx import _C

    # Convert enabled to appropriate format
    if enabled is None or enabled is False or enabled == "none":
        _C._neuron_record_memory_history(
            enabled=False,
            context=None,
            stacks=None,
            max_entries=max_entries,
            clear_history=False,
        )
        return

    # Validate context parameter
    valid_contexts = {"all", "alloc", "state", "never", None}
    if context not in valid_contexts:
        raise ValueError(f"Invalid context '{context}'. Must be one of: {valid_contexts}")

    # Validate stacks parameter
    valid_stacks = {"all", "python"}
    if stacks not in valid_stacks:
        raise ValueError(f"Invalid stacks '{stacks}'. Must be one of: {valid_stacks}")

    # Map context to C++ enum string
    context_str = context if context else "never"

    _C._neuron_record_memory_history(
        enabled=True,
        context=context_str,
        stacks=stacks,
        max_entries=max_entries,
        clear_history=True,
    )


def _snapshot(device: int | torch.device | None = None) -> dict[str, Any]:
    """
    Capture a snapshot of the current memory state.

    Returns a dictionary containing:
    - segments: List of memory segments with their blocks
    - device_traces: Per-device list of trace entries (if recording enabled)

    Each segment contains:
    - device: Device index
    - address: Memory address
    - total_size: Total segment size in bytes
    - allocated_size: Allocated portion size
    - active_size: Active (in-use) portion size
    - stream: Stream ID where segment was allocated
    - blocks: List of blocks within the segment

    Each block contains:
    - size: Block size in bytes
    - requested_size: Originally requested size
    - address: Block address
    - state: "active_allocated" or "inactive"
    - frames: Stack trace frames (if context recording enabled)

    Each trace entry contains:
    - action: Event type (alloc, free_requested, free_completed, etc.)
    - device: Device index
    - addr: Memory address
    - size: Allocation size
    - stream: Stream ID
    - time_us: Timestamp in microseconds
    - frames: Stack trace frames (if context recording enabled)

    Args:
        device: Device to snapshot (currently returns all devices).

    Returns:
        Dictionary with segments and device_traces.

    Example:
        >>> import torch_neuronx
        >>> torch_neuronx.memory._record_memory_history(True)
        >>> x = torch.zeros(1000, device='neuron')
        >>> snapshot = torch_neuronx.memory._snapshot()
        >>> print(f"Segments: {len(snapshot['segments'])}")
    """
    from torch_neuronx import _C

    return _C._neuron_memory_snapshot()


def segment_pool_analysis(device: int | torch.device | None = None) -> dict[str, Any]:
    """Compute segment pool fragmentation and efficiency stats from the memory snapshot."""
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    stats = _C._get_memory_stats(device_index)
    snapshot = _C._neuron_memory_snapshot()
    total_hbm = _C._neuron_getDeviceProperties(device_index).total_memory

    # Collect segments for this device, sorted by address
    segments = []
    for seg in snapshot.get("segments", []):
        if seg.get("device", -1) != device_index:
            continue
        segments.append(seg)
    segments.sort(key=lambda s: s.get("address", 0))

    # Build free chunk lists separately:
    # - nrt_free_chunks: inter-segment gaps + edge unreserved (real NRT free)
    # - cached_free_chunks: inactive blocks within segments (reusable without NRT call)
    nrt_free_chunks = []
    cached_free_chunks = []
    internal_frag = 0
    num_segments = len(segments)

    for seg in segments:
        internal_frag += seg.get("allocated_size", 0) - seg.get("active_size", 0)
        for block in seg.get("blocks", []):
            if block.get("state") != "active_allocated":
                cached_free_chunks.append(block.get("size", 0))

    # Inter-segment gaps (NRT-level free)
    inter_seg_gap_total = 0
    for i in range(len(segments) - 1):
        gap_start = segments[i].get("address", 0) + segments[i].get("total_size", 0)
        gap_end = segments[i + 1].get("address", 0)
        if gap_end > gap_start:
            gap = gap_end - gap_start
            nrt_free_chunks.append(gap)
            inter_seg_gap_total += gap

    # Address span
    if segments:
        min_addr = segments[0].get("address", 0)
        last = segments[-1]
        max_addr = last.get("address", 0) + last.get("total_size", 0)
        address_span = max_addr - min_addr
    else:
        address_span = 0

    # Edge unreserved HBM (outside segment span)
    reserved = stats["reserved_bytes"]["current"]
    nrt_free = total_hbm - reserved
    edge_unreserved = max(0, nrt_free - inter_seg_gap_total)
    if edge_unreserved > 0:
        nrt_free_chunks.append(edge_unreserved)

    active = stats["active_bytes"]["current"]
    allocated = stats["allocated_bytes"]["current"]
    total_free = max(0, total_hbm - allocated)
    cached_free = reserved - allocated
    all_free_chunks = nrt_free_chunks + cached_free_chunks
    free_chunks = len(all_free_chunks)
    # largest_free is the largest NRT free chunk (inter-segment gap or edge),
    # not cached blocks — cached blocks are already in cached_free
    largest_free = max(nrt_free_chunks) if nrt_free_chunks else 0

    external_frag = max(total_free - largest_free - cached_free, 0)
    # No free memory = fully fragmented (can't allocate anything)
    frag_index = (external_frag + internal_frag) / total_free if total_free > 0 else 1.0
    efficiency = reserved / address_span if address_span > 0 else 0.0

    return {
        "free_chunks": free_chunks,
        "largest_free": largest_free,
        "total_free": total_free,
        "nrt_free": nrt_free,
        "total_hbm": total_hbm,
        "internal_fragmentation": internal_frag,
        "external_fragmentation": external_frag,
        "cached_free": cached_free,
        "fragmentation_index": frag_index,
        "memory_efficiency": efficiency,
        "address_span": address_span,
        "num_segments": num_segments,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "active_bytes": active,
    }


def segment_pool_analysis_summary(device: int | torch.device | None = None) -> str:
    """Return a formatted string of the segment pool analysis."""
    s = segment_pool_analysis(device)

    def _fmt(b: int | float) -> str:
        for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
            if abs(b) < 1024:
                return f"{b:,.2f} {u}".rjust(14)
            b /= 1024
        return f"{b:,.2f} PiB".rjust(14)

    return (
        f"{'=' * 75}\n"
        f"  Segment Pool Analysis (current state at exit)\n"
        f"{'=' * 75}\n"
        f"  Allocated:        {_fmt(s['allocated_bytes'])}  [{s['allocated_bytes']:,} B]\n"
        f"  Reserved:         {_fmt(s['reserved_bytes'])}  [{s['reserved_bytes']:,} B]\n"
        f"  Active:           {_fmt(s['active_bytes'])}  [{s['active_bytes']:,} B]\n"
        f"  Total HBM:        {_fmt(s['total_hbm'])}  [{s['total_hbm']:,} B]\n"
        f"  ---\n"
        f"  Free chunks:      {s['free_chunks']:>14}\n"
        f"  Largest free:     {_fmt(s['largest_free'])}  [{s['largest_free']:,} B]\n"
        f"  Total free:       {_fmt(s['total_free'])}  [{s['total_free']:,} B]\n"
        f"  NRT free:         {_fmt(s['nrt_free'])}  [{s['nrt_free']:,} B]\n"
        f"  Cached free:      {_fmt(s['cached_free'])}  [{s['cached_free']:,} B]\n"
        f"  Internal frag:    {_fmt(s['internal_fragmentation'])}"
        f"  [{s['internal_fragmentation']:,} B]\n"
        f"  External frag:    {_fmt(s['external_fragmentation'])}"
        f"  [{s['external_fragmentation']:,} B]\n"
        f"  Frag index:       {s['fragmentation_index']:>14.4f}\n"
        f"  Mem efficiency:   {s['memory_efficiency']:>14.4f}\n"
        f"  Address span:     {_fmt(s['address_span'])}  [{s['address_span']:,} B]\n"
        f"  Segments:         {s['num_segments']:>14}\n"
        f"{'=' * 75}"
    )


def _dump_snapshot(filename: str = "memory_snapshot.pickle") -> None:
    """
    Dump a memory snapshot to a pickle file.

    The snapshot can be loaded and analyzed later, or visualized using
    PyTorch's memory visualization tools.

    Args:
        filename: Output filename. Defaults to "memory_snapshot.pickle".

    Example:
        >>> import torch_neuronx
        >>> torch_neuronx.memory._record_memory_history(True)
        >>> # ... perform allocations ...
        >>> torch_neuronx.memory._dump_snapshot("debug_snapshot.pickle")
    """
    import pickle

    snapshot = _snapshot()
    with open(filename, "wb") as f:
        pickle.dump(snapshot, f)
