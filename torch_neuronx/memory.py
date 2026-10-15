"""
Neuron Memory Management API

Provides memory statistics and management functions
for monitoring device memory usage on Neuron hardware.
"""

import json
import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _get_device_index(device: int | torch.device | None = None) -> int:
    """Get device index, defaulting to current device."""
    # Import here to avoid circular imports
    import torch_neuronx

    return torch_neuronx._get_device_index(device, optional=False)


def memory_stats(device: int | torch.device | None = None) -> dict[str, Any]:
    """Return flat memory allocator statistics for a Neuron device.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the current device.

    Returns:
        dict: Byte values at ``{allocated_bytes,reserved_bytes,active_bytes}.``
            ``{all,large_pool,small_pool}.{current,peak,allocated,freed}``; allocator,
            deferred-allocation, and scalar-cache counters; scalar cache size; and optional
            per-category byte values at ``usage.<category>``.


    Raises:
        TypeError: If the runtime returns a non-dictionary byte category.

    Note:
        The binding provides only ``all`` totals; current pool values are derived
        from snapshots. Other ``large_pool`` values mirror ``all`` and ``small_pool``
        values are zero. ``num_allocator_cache_hits``, ``num_allocator_cache_misses``,
        ``num_splits``, and ``num_merges`` from the binding are omitted.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    raw = _C._get_memory_stats(device_index)

    flat: dict[str, Any] = {}
    _byte_categories = ("allocated_bytes", "reserved_bytes", "active_bytes")
    _sub_keys = ("current", "peak", "allocated", "freed")

    pool_current = _compute_per_pool_current(device_index)

    for cat in _byte_categories:
        sub = raw.get(cat, {})
        if not isinstance(sub, dict):
            raise TypeError(
                f"Expected dict for '{cat}' from _get_memory_stats, got {type(sub).__name__}"
            )
        for metric in _sub_keys:
            val = sub.get(metric, 0)
            flat[f"{cat}.all.{metric}"] = val
            if metric == "current":
                flat[f"{cat}.large_pool.{metric}"] = pool_current.get(f"{cat}.large_pool", 0)
                flat[f"{cat}.small_pool.{metric}"] = pool_current.get(f"{cat}.small_pool", 0)
            else:
                flat[f"{cat}.large_pool.{metric}"] = val
                flat[f"{cat}.small_pool.{metric}"] = 0

    for key in (
        "num_alloc_retries",
        "num_ooms",
        "num_tensor_frees",
        "allocation_requests",
        "num_alloc_deferred",
        "num_alloc_drain_retry_ok",
        "num_alloc_drain_retry_fail",
        "num_scalar_cache_hits",
        "num_scalar_cache_misses",
        "scalar_cache_size",
    ):
        flat[key] = raw.get(key, 0)

    # Per-category HBM usage breakdown from the runtime
    try:
        breakdown = _C._nrt_get_memory_usage_breakdown(device_index)
        for category, bytes_used in breakdown:
            flat[f"usage.{category}"] = bytes_used
    except Exception as e:
        # The older runtime does not have this API. Skip the breakdown.
        logger.debug(f"memory usage breakdown not available for device {device_index}: {e}")

    return flat


def memory_stats_as_nested_dict(device: int | torch.device | None = None) -> dict[str, Any]:
    """Return memory allocator statistics grouped into nested dictionaries.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        dict: Byte values nested by category, pool, and state; allocator,
            deferred-allocation, and scalar-cache counters; scalar cache size; and an optional
            ``usage`` mapping of runtime categories to byte values.
    """
    flat = memory_stats(device)
    nested: dict[str, Any] = {}
    _byte_categories = ("allocated_bytes", "reserved_bytes", "active_bytes")
    _pools = ("all", "large_pool", "small_pool")
    _sub_keys = ("current", "peak", "allocated", "freed")

    for cat in _byte_categories:
        nested[cat] = {}
        for pool in _pools:
            nested[cat][pool] = {}
            for metric in _sub_keys:
                nested[cat][pool][metric] = flat.get(f"{cat}.{pool}.{metric}", 0)

    # Scalar counters
    for key in (
        "num_alloc_retries",
        "num_ooms",
        "num_tensor_frees",
        "allocation_requests",
        "num_alloc_deferred",
        "num_alloc_drain_retry_ok",
        "num_alloc_drain_retry_fail",
        "num_scalar_cache_hits",
        "num_scalar_cache_misses",
        "scalar_cache_size",
    ):
        nested[key] = flat.get(key, 0)

    # Per-category HBM usage breakdown. The runtime supplies the categories.
    # A key is present only on a runtime that supports the breakdown API.
    for key, value in flat.items():
        if key.startswith("usage."):
            nested.setdefault("usage", {})[key[len("usage.") :]] = value

    return nested


def _compute_per_pool_current(device_index: int) -> dict[str, int]:
    """Derive per-pool current byte stats from the memory snapshot.

    Walks snapshot segments and sums sizes by pool category (is_small).
    Returns a dict with keys like "allocated_bytes.large_pool", etc.
    """
    from torch_neuronx import _C

    try:
        snapshot = _C._neuron_memory_snapshot()
    except Exception:
        return {}

    small_reserved = 0
    small_allocated = 0
    small_active = 0
    large_reserved = 0
    large_allocated = 0
    large_active = 0

    for seg in snapshot.get("segments", []):
        if seg.get("device", -1) != device_index:
            continue
        total = seg.get("total_size", 0)
        alloc = seg.get("allocated_size", 0)
        active = seg.get("active_size", 0)
        if seg.get("is_small", False):
            small_reserved += total
            small_allocated += alloc
            small_active += active
        else:
            large_reserved += total
            large_allocated += alloc
            large_active += active

    return {
        "reserved_bytes.small_pool": small_reserved,
        "allocated_bytes.small_pool": small_allocated,
        "active_bytes.small_pool": small_active,
        "reserved_bytes.large_pool": large_reserved,
        "allocated_bytes.large_pool": large_allocated,
        "active_bytes.large_pool": large_active,
    }


def memory_allocated(device: int | torch.device | None = None) -> int:
    """Return bytes currently occupied by allocated memory blocks.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        int: Current ``allocated_bytes.all`` value.
    """
    stats = memory_stats(device)
    return stats["allocated_bytes.all.current"]


def max_memory_allocated(device: int | torch.device | None = None) -> int:
    """Return peak allocated memory in bytes since the last reset.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        int: Peak ``allocated_bytes.all`` value.
    """
    stats = memory_stats(device)
    return stats["allocated_bytes.all.peak"]


def memory_reserved(device: int | torch.device | None = None) -> int:
    """Return bytes currently reserved by the caching allocator.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        int: Current ``reserved_bytes.all`` value.
    """
    stats = memory_stats(device)
    return stats["reserved_bytes.all.current"]


def max_memory_reserved(device: int | torch.device | None = None) -> int:
    """Return peak reserved memory in bytes since the last reset.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        int: Peak ``reserved_bytes.all`` value.
    """
    stats = memory_stats(device)
    return stats["reserved_bytes.all.peak"]


def reset_peak_memory_stats(device: int | torch.device | None = None) -> None:
    """Reset peak memory statistics to their current values.

    Args:
        device (int or torch.device, optional): Device whose peak statistics to
            reset. Defaults to the current device.

    Note:
        Subsequent peak allocated and reserved values are measured from this point.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    _C._reset_peak_memory_stats(device_index)


def reset_accumulated_memory_stats(device: int | torch.device | None = None) -> None:
    """Reset accumulated memory allocator counters.

    Args:
        device (int or torch.device, optional): Device whose counters to reset.
            Defaults to the current device.

    Note:
        This resets all event counters, the scalar cache size, and the historical
        ``allocated`` and ``freed`` totals, but leaves ``current`` and ``peak`` byte
        values unchanged.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    _C._reset_accumulated_memory_stats(device_index)


def peak_memory_stats(device: int | torch.device | None = None) -> dict[str, Any]:
    """Return the allocator snapshot captured at the allocation peak.

    Args:
        device (int or torch.device, optional): Device to query. Defaults to the
            current device.

    Returns:
        dict: ``valid``; ``timestamp_ns`` in steady-clock nanoseconds; byte-valued
            ``allocated_bytes``, ``reserved_bytes``, and ``active_bytes`` dictionaries
            with ``current``, ``peak``, ``allocated``, and ``freed``; byte-valued
            ``internal_fragmentation``; and counters ``num_alloc_retries``, ``num_ooms``,
            ``num_tensor_frees``, ``allocation_requests``, ``num_allocator_cache_hits``,
            ``num_allocator_cache_misses``, ``num_splits``, ``num_merges``,
            ``num_alloc_deferred``, ``num_alloc_drain_retry_ok``, and
            ``num_alloc_drain_retry_fail``.

    Note:
        The snapshot updates when allocated bytes exceed their previous peak, survives
        ``reset_peak_memory_stats``, and is logged via ``TORCH_WARN`` on OOM.
    """
    from torch_neuronx import _C

    device_index = _get_device_index(device)
    return _C._get_peak_memory_stats(device_index)


def memory_summary(device: int | torch.device | None = None, abbreviated: bool = False) -> str:
    """Format a human-readable memory summary for a Neuron device.

    Args:
        device (int or torch.device, optional): Device to summarize. Defaults to the
            current device.
        abbreviated (bool, optional): Request abbreviated output. Currently has no
            effect. Defaults to ``False``.

    Returns:
        str: Table of allocated, reserved, and active memory in binary units for
            current, peak, allocated, and freed totals, plus allocation requests,
            tensor frees, OOMs, allocation retries, and deferred-retry outcomes.
    """
    device_index = _get_device_index(device)
    stats = memory_stats(device_index)

    def format_bytes(b: int) -> str:
        """Format a byte count using a binary unit.

        Args:
            b (int): Byte count to format.

        Returns:
            str: Byte count scaled from bytes through pebibytes.
        """
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
        f"  Defer-retry: deferred={stats.get('num_alloc_deferred', 0):,}  "
        f"drain_retry_ok={stats.get('num_alloc_drain_retry_ok', 0):,}  "
        f"drain_retry_fail={stats.get('num_alloc_drain_retry_fail', 0):,}",
        "=" * 75,
        "  Metric              | Current    | Peak       | Allocated  | Freed",
        "-" * 75,
    ]

    for name, key in [
        ("Allocated Memory", "allocated_bytes"),
        ("Reserved Memory", "reserved_bytes"),
        ("Active Memory", "active_bytes"),
    ]:
        lines.append(
            f"  {name:<18} "
            f"| {format_bytes(stats[f'{key}.all.current']):>10} "
            f"| {format_bytes(stats[f'{key}.all.peak']):>10} "
            f"| {format_bytes(stats[f'{key}.all.allocated']):>10} "
            f"| {format_bytes(stats[f'{key}.all.freed']):>10}"
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
    """Capture current allocator segments and memory-event traces.

    Args:
        device (int or torch.device, optional): Device selector. Currently ignored;
            the snapshot includes all devices.

    Returns:
        dict: Contains ``segments`` and per-device ``device_traces``. Each segment has
            ``device``, ``address``, ``total_size``, ``allocated_size``, ``active_size``,
            ``stream``, ``is_small``, and ``blocks``. Each block has ``size``,
            ``requested_size``, ``address``, ``state``, and ``frames``; state is
            ``"active_allocated"`` or ``"inactive"``. Each trace has ``action``,
            ``device``, ``addr``, ``size``, ``block_size``, ``stream``, ``time_us``,
            and ``frames``. Segment and block size fields, ``size``, and ``block_size``
            are bytes; ``time_us`` is microseconds. Frames are present when context
            recording is enabled.

    Note:
        Trace actions include ``alloc``, ``free_requested``, and ``free_completed``.
        An ``oom`` trace adds ``device_free`` in bytes and uses ``addr`` for the failed
        requested byte size rather than a memory address.
    """
    from torch_neuronx import _C

    return _C._neuron_memory_snapshot()


def segment_pool_analysis(device: int | torch.device | None = None) -> dict[str, Any]:
    """Compute segment-pool fragmentation and efficiency statistics.

    Args:
        device (int or torch.device, optional): Device to analyze. Defaults to the
            current device.

    Returns:
        dict: Counts ``free_chunks`` and ``num_segments``; byte values
            ``largest_free``, ``total_free``, ``nrt_free``, ``total_hbm``,
            ``internal_fragmentation``, ``external_fragmentation``, ``cached_free``,
            ``address_span``, ``allocated_bytes``, ``reserved_bytes``, and
            ``active_bytes``; ratios ``fragmentation_index`` and ``memory_density``.

    Note:
        Device HBM is queried only when the runtime is initialized; otherwise
        ``total_hbm`` is ``0`` and ``fragmentation_index`` is ``None``. Memory
        density uses reserved bytes and the segment address span and remains
        available without querying device HBM.
    """
    from torch_neuronx import _C, is_neuron_runtime_initialized

    device_index = _get_device_index(device)
    stats = memory_stats(device_index)
    snapshot = _C._neuron_memory_snapshot()
    # Only call into NRT when the runtime is alive in this process.
    if is_neuron_runtime_initialized():
        total_hbm = _C._neuron_getDeviceProperties(device_index).total_memory
    else:
        total_hbm = 0

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
    reserved = stats["reserved_bytes.all.current"]
    nrt_free = total_hbm - reserved
    edge_unreserved = max(0, nrt_free - inter_seg_gap_total)
    if edge_unreserved > 0:
        nrt_free_chunks.append(edge_unreserved)

    active = stats["active_bytes.all.current"]
    allocated = stats["allocated_bytes.all.current"]
    total_free = max(0, total_hbm - allocated)
    cached_free = reserved - allocated
    all_free_chunks = nrt_free_chunks + cached_free_chunks
    free_chunks = len(all_free_chunks)
    # largest_free is the largest NRT free chunk (inter-segment gap or edge),
    # not cached blocks — cached blocks are already in cached_free
    largest_free = max(nrt_free_chunks) if nrt_free_chunks else 0

    external_frag = max(total_free - largest_free - cached_free, 0)
    # No free memory = fully fragmented (can't allocate anything).
    if total_hbm <= 0:
        frag_index = None
    elif total_free > 0:
        frag_index = (external_frag + internal_frag) / total_free
    else:
        frag_index = 1.0
    density = reserved / address_span if address_span > 0 else 0.0

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
        "memory_density": density,
        "address_span": address_span,
        "num_segments": num_segments,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "active_bytes": active,
    }


def segment_pool_analysis_summary(device: int | torch.device | None = None) -> str:
    """Format segment-pool analysis for a Neuron device.

    Args:
        device (int or torch.device, optional): Device to analyze. Defaults to the
            current device.

    Returns:
        str: Table of allocator bytes, free chunks, fragmentation, density, address
            span, and segment count.
    """
    s = segment_pool_analysis(device)
    frag = s["fragmentation_index"]
    frag_str = (f"{frag:.4f}" if frag is not None else "n/a").rjust(14)

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
        f"  Internal frag:    {_fmt(s['internal_fragmentation'])}  "
        f"[{s['internal_fragmentation']:,} B]\n"
        f"  External frag:    {_fmt(s['external_fragmentation'])}  "
        f"[{s['external_fragmentation']:,} B]\n"
        f"  Frag index:       {frag_str}\n"
        f"  Mem density:      {s['memory_density']:>14.4f}\n"
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


# =============================================================================
# On-demand dump of allocator-quality metrics
# =============================================================================

# When the Neuron metrics subsystem is enabled
# (``torch_neuronx.metrics.set_enabled(True)`` or ``TORCH_NEURONX_METRICS_ENABLED=1``)
# callers can invoke :func:`dump_allocator_stats` to write per-rank
# ``allocator_stats_rank{rank}.json`` files. This is intended for torchrun-style
# worker processes: the caller (e.g. the model training entrypoint) invokes
# this before the runtime tears down so the orchestrator can read live
# allocator state after the worker has exited.
#
# Output directory follows the existing torch_neuronx convention:
#   ``TORCH_NEURONX_METRICS_DIR`` if set, else ``/tmp``.
ALLOCATOR_STATS_METRICS_DIR_ENV = "TORCH_NEURONX_METRICS_DIR"
DEFAULT_ALLOCATOR_STATS_DIR = "/tmp"


def _allocator_stats_dump_rank() -> int:
    """Best-effort rank lookup that doesn't require ``torch.distributed`` init."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception as e:
        logger.debug(f"torch.distributed rank lookup failed, falling back: {e}")
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env)
        except ValueError:
            pass
    return 0


def _collect_allocator_stats_per_device() -> list[dict[str, Any]]:
    """Snapshot ``memory_stats`` and ``segment_pool_analysis`` for this rank's device.

    Returns a single-element list keyed by the device index that the current
    process owns. In SPMD torchrun-style runs each worker only sees one
    Neuron core (other cores belong to sibling processes).
    Returns an empty list if no current device is bound.
    """
    from torch_neuronx import current_device

    try:
        dev = current_device()
    except Exception as e:
        logger.debug(f"current_device() lookup failed: {e}")
        return []

    entry: dict[str, Any] = {"device": dev}
    try:
        entry["memory_stats"] = memory_stats(dev)
    except Exception as e:
        logger.debug(f"memory_stats failed for device {dev}: {e}")
        entry["memory_stats"] = {}
    try:
        entry["segment_pool_analysis"] = segment_pool_analysis(dev)
    except Exception as e:
        logger.debug(f"segment_pool_analysis failed for device {dev}: {e}")
        entry["segment_pool_analysis"] = {}
    return [entry]


def dump_allocator_stats(out_dir: str | None = None) -> str | None:
    """Write per-rank allocator statistics to disk.

    Args:
        out_dir (str, optional): Destination directory. Defaults to
            ``$TORCH_NEURONX_METRICS_DIR`` and then ``/tmp``.

    Returns:
        str or None: Written JSON path, or ``None`` when metrics are disabled,
            directory creation fails, collection setup raises, or writing fails.

    Note:
        Call this near the end of training while the runtime remains alive. It
        attempts to synchronize the current device and records ``memory_stats`` and
        ``segment_pool_analysis`` for the current worker; per-device collection
        failures are represented by empty device data or dictionaries.
    """
    try:
        from torch_neuronx import metrics

        if not metrics.is_enabled():
            return None
    except Exception:
        return None

    target_dir = out_dir or os.environ.get(
        ALLOCATOR_STATS_METRICS_DIR_ENV, DEFAULT_ALLOCATOR_STATS_DIR
    )
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        logger.debug(f"Failed to create allocator stats dir {target_dir}: {e}")
        return None

    try:
        # Drain in-flight kernels on this rank's device so accounting
        # reflects the steady state.
        from torch_neuronx import current_device, synchronize

        try:
            synchronize(current_device())
        except Exception as e:
            logger.debug(f"torch_neuronx.synchronize() failed during dump: {e}")

        per_device = _collect_allocator_stats_per_device()
    except Exception as e:
        logger.debug(f"Failed to collect allocator stats: {e}")
        return None

    rank = _allocator_stats_dump_rank()
    payload = {
        "rank": rank,
        "pid": os.getpid(),
        "devices": per_device,
    }

    path = os.path.join(target_dir, f"allocator_stats_rank{rank}.json")
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug(f"Wrote allocator stats to {path}")
        return path
    except OSError as e:
        logger.debug(f"Failed to write allocator stats to {path}: {e}")
        return None
