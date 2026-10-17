# Neuron device module for PyTorch
# This module is accessed as torch.neuron and provides device-specific functionality

# Import memory module to enable torch.neuron.memory.* access
import sys

import torch_neuronx.memory as memory  # noqa: F401


def is_initialized() -> bool:
    """Return whether the Neuron runtime is initialized.

    Returns:
        bool: ``True`` if the runtime is initialized; otherwise ``False``.
    """
    import torch_neuronx

    return torch_neuronx.is_neuron_runtime_initialized()


def current_device() -> int:
    """Return the current Neuron device index.

    Returns:
        int: Thread-local selection, process default, or ``0`` when neither is set.

    Note:
        This host-side query does not synchronize device work.
    """
    import torch_neuronx

    return torch_neuronx._C._neuron_getCurrentDevice()


def set_device(device) -> None:
    """Set the current Neuron device from an index-bearing value.

    Args:
        device (int, torch.device, or None): Value whose index selects the Neuron
            device. ``None`` or a device with no index keeps the current selection.

    Raises:
        ValueError: If the resolved index is outside the available device range.
    """
    import torch_neuronx

    device_idx = getattr(device, "index", device)
    if device_idx is None:
        device_idx = current_device()
    torch_neuronx.set_device(device_idx)


def get_device_name(device=0):
    """Return a device name from an integer or device object.

    Args:
        device (int or torch.device, optional): Device specification to name.
            Defaults to ``0``.

    Returns:
        str: String form of ``device``, or ``"neuron:<index>"`` for an integer.

    Raises:
        ValueError: If ``device`` is neither an integer nor a ``torch.device``.
    """
    import torch

    if isinstance(device, torch.device):
        return str(device)
    elif isinstance(device, int):
        return f"neuron:{device}"
    raise ValueError(
        f"Incorrect device argument found, expected torch.device or int type, found {type(device)}"
    )


def device_count() -> int:
    """Return the number of Neuron device indices exposed to PyTorch.

    Returns:
        int: Configured local world size when distributed state is initialized;
            otherwise, the visible VNC count.
    """
    import torch_neuronx

    return torch_neuronx.device_count()


def is_available() -> bool:
    """Return whether at least one Neuron device is available.

    Returns:
        bool: ``True`` when the device count is positive; ``False`` when it is
        zero or the count cannot be queried.
    """
    try:
        import torch_neuronx

        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def init():
    """Initialize the Neuron runtime and apply the pending device seed.

    Raises:
        RuntimeError: If initialization fails or the process was forked after
            initialization.
    """
    import torch_neuronx

    torch_neuronx._lazy_init()


def get_amp_supported_dtype():
    """Return dtypes supported for automatic mixed precision.

    Returns:
        list[torch.dtype]: ``torch.float32``, ``torch.float16``, and
        ``torch.bfloat16``.
    """
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]


def _is_in_bad_fork() -> bool:
    """Check if we're in a bad fork state."""
    import torch_neuronx

    return torch_neuronx._is_in_bad_fork


def manual_seed_all(seed: int) -> None:
    """Set every accessible generator seed and reset its Philox offset.

    Args:
        seed (int): Seed value, converted to an integer before use.

    Note:
        The accessible generator count is frozen on its first native query and
        may differ from a later ``device_count()``. When initialized and NKI
        RNG is available, this also queues a GPSIMD RNG-state update on the
        current Neuron device.
    """
    import torch_neuronx

    seed = int(seed)
    for i in range(torch_neuronx._C._get_default_generator_count()):
        default_generator = torch_neuronx._C._get_default_generator(i)
        default_generator.manual_seed(seed)
    if is_initialized():
        _set_device_rng_seed(seed)


def manual_seed(seed: int) -> None:
    """Set the current generator seed and reset its Philox offset.

    Args:
        seed (int): Seed value, converted to an integer before use.

    Raises:
        RuntimeError: If the current device is outside the frozen generator set.

    Note:
        When initialized and NKI RNG is available, this also queues a GPSIMD
        RNG-state update on the current Neuron device.
    """
    import torch_neuronx

    seed = int(seed)
    idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    default_generator.manual_seed(seed)
    if is_initialized():
        _set_device_rng_seed(seed)


def _set_device_rng_seed(seed: int) -> None:
    """Set the on-device GPSIMD RNG state from a scalar seed."""
    import torch
    from torch.utils._python_dispatch import _disable_current_modes

    from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

    if _has_nki_rng:
        from torch_neuronx.python_ops.nki_kernels.rng_state import NUM_RNG_SEEDS, set_rng_state

        # Seeding the device is host-side state management, not part of the
        # computation, so it must stay invisible to an active dispatch or
        # __torch_function__ mode. CUDA gets this for free because it seeds in
        # C++; Neuron builds a seed tensor in Python, so a tracing or
        # cross-reference mode captures it and .to("neuron") then raises
        # "Cannot copy out of meta tensor; no data!". torch.manual_seed() reaches
        # here through the PrivateUse1 hook, so every test that seeds before a
        # random op hit that error.
        # Both stacks must be off: _disable_current_modes() pops TorchDispatchMode
        # only, and Tensor.to is intercepted through __torch_function__.
        with _disable_current_modes(), torch._C.DisableTorchFunction():
            # Cast to int32 until torch-mlir uint32 is there
            state = torch.zeros(1, NUM_RNG_SEEDS, dtype=torch.uint32, device="cpu")
            state[0][0] = seed & 0xFFFFFFFF
            state = state.view(torch.int32).to("neuron")
            set_rng_state(state)
    # If nki rng is not available, no-op


def seed() -> None:
    """Assign a nondeterministic seed and reset the current generator's Philox offset.

    Raises:
        RuntimeError: If the current device is outside the frozen generator set.
    """
    import torch_neuronx

    idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    default_generator.seed()


def seed_all() -> None:
    """Assign one nondeterministic seed to all accessible device generators.

    Note:
        The first generator produces the seed, which is copied to every other
        accessible generator; every generator's Philox offset is reset. The
        accessible generator count is frozen on its first native query.
    """
    import torch_neuronx

    random_seed = 0
    for i in range(torch_neuronx._C._get_default_generator_count()):
        default_generator = torch_neuronx._C._get_default_generator(i)
        if i == 0:
            default_generator.seed()
            random_seed = default_generator.initial_seed()
        else:
            default_generator.manual_seed(random_seed)


def initial_seed() -> int:
    """Return the initial seed of the current Neuron device generator.

    Returns:
        int: Initial seed for the current generator.

    Raises:
        RuntimeError: If the current device is outside the frozen generator set.
    """
    import torch_neuronx

    idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    return default_generator.initial_seed()


def get_rng_state(device=None):
    """Return a synchronous CPU copy of a Neuron generator's RNG state.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        torch.Tensor: A 16-element CPU ``uint8`` tensor containing the 8-byte
        seed followed by the 8-byte Philox offset.

    Raises:
        RuntimeError: If the device index is outside the frozen generator set.
    """
    import torch

    import torch_neuronx

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("neuron", device)
    idx = device.index if device is not None else None
    if idx is None:
        idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    return default_generator.get_state()


_last_seed_state = None


def set_rng_state(state, device=None):
    """Replace a Neuron generator's RNG state synchronously.

    Args:
        state (torch.Tensor): CPU ``uint8`` tensor containing an 8-byte seed or
            a 16-byte seed-and-offset state.
        device (int, str, or torch.device, optional): Device to update. Defaults
            to ``None`` (the current Neuron device).

    Raises:
        RuntimeError: If the device index is invalid or ``state`` contains
            neither 8 nor 16 elements.

    Note:
        An 8-byte state resets the Philox offset to zero. Outside
        ``FakeTensorMode``, a changed pair also reseeds the CPU default generator
        with the seed XOR offset.
    """
    import torch

    import torch_neuronx

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("neuron", device)
    idx = device.index if device is not None else None
    if idx is None:
        idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    default_generator.set_state(state)

    # Sync Philox state to CPU so fallback RNG ops respect DTensor's offset.
    # Also sync to GPSIMD hardware RNG so on-device NKI random ops produce
    # rank-differentiated values when DTensor adjusts per-rank offsets.
    # Guard: skip when inside FakeTensorMode (e.g., DTensor sharding propagation
    # with LocalTensorMode), as torch.manual_seed triggers clone() on the CPU
    # RNG state tensor which FakeTensorMode rejects.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None:
        current_state = (default_generator.initial_seed(), default_generator.get_offset())
        global _last_seed_state
        if current_state != _last_seed_state:
            _last_seed_state = current_state
            combined_seed = current_state[0] ^ current_state[1]
            torch._C.default_generator.manual_seed(combined_seed)
            if is_initialized():
                _set_device_rng_seed(combined_seed)


# Import device class from main module to avoid duplication
def device(device_spec):
    """Create a context manager that changes the selected Neuron device.

    Args:
        device_spec (int, str, torch.device, or None): Device to select, or
            ``None`` to use the current device.

    Returns:
        torch_neuronx.device: Context manager that restores the prior device on
            exit.

    Raises:
        ValueError: If ``device_spec`` is not a valid Neuron device specification.
    """
    import torch_neuronx

    return torch_neuronx.device(device_spec)


# Stream management functions


def Stream(**kwargs):  # noqa: N802
    """Create a Neuron stream.

    Args:
        **kwargs (Any): Stream options forwarded to ``torch_neuronx.Stream``,
            including ``device``, ``priority``, or reconstruction fields.

    Returns:
        torch_neuronx.Stream: Newly created or reconstructed Neuron stream.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
    """
    import torch_neuronx

    return torch_neuronx.Stream(**kwargs)


def stream(stream):
    """Create a context manager that selects a Neuron stream.

    Args:
        stream (torch_neuronx.Stream or None): Stream to select. ``None``
            creates a no-op context.

    Returns:
        torch_neuronx.StreamContext: Context manager that restores prior streams
        on exit.
    """
    import torch_neuronx

    return torch_neuronx.stream(stream)


def current_stream(device=None):
    """Return the currently selected stream for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        torch_neuronx.Stream: Currently selected stream.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is invalid.
    """
    import torch_neuronx

    return torch_neuronx.current_stream(device)


def default_stream(device=None):
    """Return the default stream for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        torch_neuronx.Stream: Default stream for the selected device.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is invalid.
    """
    import torch_neuronx

    return torch_neuronx.default_stream(device)


def set_stream(stream):
    """Set the current Neuron stream.

    Args:
        stream (torch_neuronx.Stream or None): Stream to select. ``None`` leaves
            the current stream unchanged.

    Note:
        Selecting a stream on another device also makes that device current.
    """
    import torch_neuronx

    torch_neuronx.set_stream(stream)


def synchronize(device=None) -> None:
    """Wait for all streams on a Neuron device to complete.

    Args:
        device (int, str, or torch.device, optional): Device to synchronize.
            Defaults to ``None`` (the current Neuron device).

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        RuntimeError: If the resolved device index is invalid.
    """
    import torch_neuronx

    torch_neuronx.synchronize(device)


# Stream operations on current stream
def wait_stream(stream) -> None:
    """Make future work on the current stream wait for another stream.

    Args:
        stream (torch_neuronx.Stream): Stream whose previously submitted work
            must complete.

    Raises:
        RuntimeError: If the current Neuron device is invalid.
    """
    current_stream().wait_stream(stream)


def wait_event(event) -> None:
    """Make future work on the current stream wait for an event.

    Args:
        event (torch_neuronx.Event): Event whose captured work must complete.

    Raises:
        RuntimeError: If the current Neuron device is invalid or ``event`` is not
            a Neuron event.
    """
    current_stream().wait_event(event)


def record_event(event=None):
    """Record an event after work already submitted to the current stream.

    Args:
        event (torch_neuronx.Event, optional): Event to record. Defaults to
            ``None``, which creates a new event.

    Returns:
        torch_neuronx.Event: Recorded event, including the supplied object when
        one is provided.

    Raises:
        RuntimeError: If the current Neuron device is invalid.
    """
    return current_stream().record_event(event)


def query() -> bool:
    """Return whether all work submitted to the current stream has completed.

    Returns:
        bool: ``True`` when all submitted work is complete; otherwise ``False``.

    Raises:
        RuntimeError: If the current Neuron device is invalid.
    """
    return current_stream().query()


def Event(enable_timing=False, blocking=False):  # noqa: N802
    """Create a Neuron event.

    Args:
        enable_timing (bool, optional): Whether to capture host timestamps when
            the event is recorded. Defaults to ``False``.
        blocking (bool, optional): Store the blocking synchronization preference,
            which is currently unused. Defaults to ``False``.

    Returns:
        torch_neuronx.Event: Newly created Neuron event.

    Note:
        Elapsed timing uses host steady-clock timestamps, stamped immediately for
        an idle stream or when queued work reaches each event; it is not a device
        hardware timer.
    """
    import torch_neuronx

    return torch_neuronx.Event(enable_timing=enable_timing, blocking=blocking)


# Memory metrics functions


def get_device_properties(device=None):
    """Return properties for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        object: Properties whose ``name`` is the instance type and whose
            ``total_memory`` is the VNC HBM limit in bytes, or ``0`` when the
            runtime memory query fails.

    Raises:
        ValueError: If the device type or index is invalid.
        RuntimeError: If runtime initialization or instance-type lookup fails.
    """
    import torch_neuronx

    return torch_neuronx.get_device_properties(
        str(device) if device is not None else current_device()
    )


def reset_peak_memory_stats(device=None):
    """Reset peak byte statistics to their current values.

    Args:
        device (int, str, or torch.device, optional): Device to reset. Defaults
            to ``None`` (the current Neuron device).

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.

    Note:
        This does not clear the snapshot returned by ``peak_memory_stats``.
    """
    import torch_neuronx

    torch_neuronx.reset_peak_memory_stats(device)


def memory_stats(device=None):
    """Return allocator memory statistics for a Neuron device.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        dict[str, Any]: Byte values by category, pool, and state; allocator,
            deferred-allocation, and scalar-cache counters; scalar cache size; and
            optional per-category runtime usage bytes.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns a non-dictionary byte category.

    Note:
        The binding exposes only ``all`` totals. Current pool values derive from
        snapshots; for non-current fields, ``large_pool`` mirrors ``all`` and
        ``small_pool`` is zero. Allocator-cache and split/merge counters are omitted.
    """
    import torch_neuronx

    return torch_neuronx.memory_stats(device)


def memory_allocated(device=None):
    """Return memory occupied by allocated blocks in bytes.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        int: Current ``allocated_bytes.all.current`` value.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.memory_allocated(device)


def max_memory_allocated(device=None):
    """Return peak allocated-block memory in bytes since the last reset.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        int: Peak ``allocated_bytes.all.peak`` value.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.max_memory_allocated(device)


def memory_reserved(device=None):
    """Return memory currently reserved by the caching allocator in bytes.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        int: Current ``reserved_bytes.all.current`` value.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.memory_reserved(device)


def max_memory_reserved(device=None):
    """Return peak reserved memory in bytes since the last reset.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        int: Peak ``reserved_bytes.all.peak`` value.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.max_memory_reserved(device)


def reset_accumulated_memory_stats(device=None):
    """Reset accumulated memory allocator counters.

    Args:
        device (int, str, or torch.device, optional): Device to reset. Defaults
            to ``None`` (the current Neuron device).

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.

    Note:
        This resets event counters and historical allocated and freed totals,
        but leaves current and peak byte values unchanged.
    """
    import torch_neuronx

    torch_neuronx.reset_accumulated_memory_stats(device)


def memory_summary(device=None, abbreviated=False):
    """Return a human-readable summary of memory usage for a device.

    Args:
        device (int, str, or torch.device, optional): Device to summarize.
            Defaults to ``None`` (the current Neuron device).
        abbreviated (bool, optional): Request abbreviated output. Currently has
            no effect. Defaults to ``False``.

    Returns:
        str: Table of allocated, reserved, and active bytes in binary units,
            plus allocation, free, OOM, retry, and deferred-allocation counters.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.memory_summary(device, abbreviated)


def empty_cache():
    """Release cached allocator blocks to the Neuron runtime.

    Note:
        With lazy allocation, this synchronizes every visible Neuron device
        before releasing cache segments. Allocated tensor memory is not released.
    """
    import torch_neuronx

    torch_neuronx.empty_cache()


def peak_memory_stats(device=None):
    """Return the allocator snapshot captured at the allocation peak.

    Args:
        device (int, str, or torch.device, optional): Device to query. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        dict[str, Any]: ``valid``; steady-clock ``timestamp_ns``; byte-valued
            allocated, reserved, and active dictionaries with current, peak,
            allocated, and freed fields; internal fragmentation; and allocator
            counters.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.

    Note:
        The snapshot updates at a new allocated-byte peak, survives
        ``reset_peak_memory_stats``, and is logged via ``TORCH_WARN`` on OOM.
    """
    import torch_neuronx

    return torch_neuronx.peak_memory_stats(device)


def segment_pool_analysis(device=None):
    """Compute segment-pool fragmentation and efficiency metrics.

    Args:
        device (int, str, or torch.device, optional): Device to analyze. Defaults to ``None``.

    Returns:
        dict[str, Any]: Counts ``free_chunks`` and ``num_segments``; byte fields
            for free space, HBM, fragmentation, address span, and allocator use;
            and ratios ``fragmentation_index`` and ``memory_density``.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.

    Note:
        Before runtime initialization, ``total_hbm`` is ``0`` and
        ``fragmentation_index`` is ``None``; memory density remains available.
    """
    import torch_neuronx

    return torch_neuronx.segment_pool_analysis(device)


def segment_pool_analysis_summary(device=None):
    """Return a formatted segment-pool analysis.

    Args:
        device (int, str, or torch.device, optional): Device to analyze. Defaults
            to ``None`` (the current Neuron device).

    Returns:
        str: Table of allocator bytes, free chunks, fragmentation, density,
            address span, and segment count.

    Raises:
        ValueError: If ``device`` is not a valid Neuron device specification.
        TypeError: If the runtime returns malformed byte statistics.
    """
    import torch_neuronx

    return torch_neuronx.segment_pool_analysis_summary(device)


def _sleep(cycles: int) -> None:
    """Sleep for the specified number of cycles.

    Args:
        cycles: Number of cycles to sleep
    """
    # TODO: Implement _sleep for Neuron
    return


def get_dynamo_metrics():
    """Return all Dynamo backend compilation metrics with resolved timing.

    Returns:
        list[NeuronCompilationMetrics]: Shallow list copy of recorded per-graph
            metrics whose lazy NEFF compilation times are populated in place.

    Note:
        Synchronizes the current Neuron device before resolving asynchronous
        compilation timing.
    """
    import torch_neuronx

    return torch_neuronx.get_dynamo_metrics()


def reset_dynamo_metrics():
    """Reset all Dynamo backend compilation metrics and counters.

    Note:
        This clears process-wide stored records and ``counters["neuron"]``.
    """
    import torch_neuronx

    torch_neuronx.reset_dynamo_metrics()


def is_bf16_supported() -> bool:
    """Report bfloat16 support for the ``torch.neuron`` compatibility module.

    Returns:
        bool: Always ``True``.

    Note:
        When test infrastructure aliases ``torch.cuda`` to ``torch.neuron``,
        this satisfies CUDA compatibility checks.
    """
    return True


# =============================================================================
# Memory Snapshot Functions
# =============================================================================


def _record_memory_history(
    enabled="all",
    *,
    context="all",
    stacks="all",
    max_entries=sys.maxsize,
    device=None,
):
    """Enable or disable recording of memory allocation/deallocation history.

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
        device: Device to configure (currently applies globally).

    Example:
        >>> torch.neuron._record_memory_history(True, context="all")
        >>> # ... perform allocations ...
        >>> snapshot = torch.neuron._snapshot()
        >>> torch.neuron._record_memory_history(None)  # Disable
    """
    import torch_neuronx

    torch_neuronx.memory._record_memory_history(
        enabled=enabled,
        context=context,
        stacks=stacks,
        max_entries=max_entries,
        device=device,
    )


def _snapshot(device=None):
    """Capture a snapshot of the current memory state.

    Returns a dictionary containing:
    - segments: List of memory segments with their blocks
    - device_traces: Per-device list of trace entries (if recording enabled)

    Args:
        device: Device to snapshot (currently returns all devices).

    Returns:
        Dictionary with segments and device_traces.

    Example:
        >>> torch.neuron._record_memory_history(True)
        >>> x = torch.zeros(1000, device='neuron')
        >>> snapshot = torch.neuron._snapshot()
        >>> print(f"Segments: {len(snapshot['segments'])}")
    """
    import torch_neuronx

    return torch_neuronx.memory._snapshot(device)


def _dump_snapshot(filename="memory_snapshot.pickle"):
    """Dump a memory snapshot to a pickle file.

    The snapshot can be loaded and analyzed later, or visualized using
    PyTorch's memory visualization tools.

    Args:
        filename: Output filename. Defaults to "memory_snapshot.pickle".

    Example:
        >>> torch.neuron._record_memory_history(True)
        >>> # ... perform allocations ...
        >>> torch.neuron._dump_snapshot("debug_snapshot.pickle")
    """
    import torch_neuronx

    torch_neuronx.memory._dump_snapshot(filename)
