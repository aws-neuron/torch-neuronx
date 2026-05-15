# Neuron device module for PyTorch
# This module is accessed as torch.neuron and provides device-specific functionality

# Import memory module to enable torch.neuron.memory.* access
import sys

import torch_neuronx.memory as memory  # noqa: F401


def is_initialized() -> bool:
    """Check if the Neuron runtime has been initialized."""
    import torch_neuronx

    return torch_neuronx.is_neuron_runtime_initialized()


def current_device() -> int:
    """Return the current Neuron device index."""
    import torch_neuronx

    return torch_neuronx._C._neuron_getCurrentDevice()


def set_device(device) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index (int) or torch.device to set as current
    """
    import torch_neuronx

    device_idx = getattr(device, "index", device)
    if device_idx is None:
        device_idx = current_device()
    torch_neuronx.set_device(device_idx)


def get_device_name(device=0):
    """Get the name of the Neuron device.

    Args:
        device: Device index or torch.device object

    Returns:
        str: Device name
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
    """Return the number of Neuron devices available."""
    import torch_neuronx

    return torch_neuronx.device_count()


def is_available() -> bool:
    """Check if Neuron devices are available."""
    try:
        import torch_neuronx

        return torch_neuronx.device_count() > 0
    except Exception:
        return False


def init():
    """Initialize the Neuron runtime."""
    import torch_neuronx

    torch_neuronx._lazy_init()


def get_amp_supported_dtype():
    """Return list of dtypes supported by Neuron for automatic mixed precision."""
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]


def _is_in_bad_fork() -> bool:
    """Check if we're in a bad fork state."""
    import torch_neuronx

    return torch_neuronx._is_in_bad_fork


def manual_seed_all(seed: int) -> None:
    """Set seed for all neuron devices."""
    import torch_neuronx

    seed = int(seed)
    for i in range(device_count()):
        default_generator = torch_neuronx._C._get_default_generator(i)
        default_generator.manual_seed(seed)
    if is_initialized():
        _set_device_rng_seed(seed)


def manual_seed(seed: int) -> None:
    """Set seed for the current neuron device."""
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

    from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

    if _has_nki_rng:
        from torch_neuronx.python_ops.nki_kernels.rng_state import NUM_RNG_SEEDS, set_rng_state

        # Cast to int32 until torch-mlir uint32 is there
        state = torch.zeros(1, NUM_RNG_SEEDS, dtype=torch.uint32, device="cpu")
        state[0][0] = seed & 0xFFFFFFFF
        state = state.view(torch.int32).to("neuron")
        set_rng_state(state)
    # If nki rng is not available, no-op


def seed() -> None:
    """Set the seed for generating random numbers to a random number for the current device."""
    import torch_neuronx

    idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    default_generator.seed()


def seed_all() -> None:
    """Set the seed for generating random numbers to a random number on all devices."""
    import torch_neuronx

    random_seed = 0
    for i in range(device_count()):
        default_generator = torch_neuronx._C._get_default_generator(i)
        if i == 0:
            default_generator.seed()
            random_seed = default_generator.initial_seed()
        else:
            default_generator.manual_seed(random_seed)


def initial_seed() -> int:
    """Return the current random seed of the current device."""
    import torch_neuronx

    idx = current_device()
    default_generator = torch_neuronx._C._get_default_generator(idx)
    return default_generator.initial_seed()


def get_rng_state(device=None):
    """Get the RNG state for a Neuron device.

    Args:
        device: Device index or torch.device (default: current device)

    Returns:
        ByteTensor containing the RNG state
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
    """Set the RNG state for a Neuron device.

    Args:
        state: ByteTensor containing the desired RNG state
        device: Device index or torch.device (default: current device)
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
    # Guard: skip when inside FakeTensorMode (e.g., DTensor sharding propagation
    # with LocalTensorMode), as torch.manual_seed triggers clone() on the CPU
    # RNG state tensor which FakeTensorMode rejects.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None:
        current_state = (default_generator.initial_seed(), default_generator.get_offset())
        global _last_seed_state
        if current_state != _last_seed_state:
            _last_seed_state = current_state
            torch._C.default_generator.manual_seed(current_state[0] ^ current_state[1])


# Import device class from main module to avoid duplication
def device(device_spec):
    """Context-manager that changes the selected device.

    Args:
        device_spec: Device index (int) or torch.device object

    Returns:
        Context manager that switches to the specified device
    """
    import torch_neuronx

    return torch_neuronx.device(device_spec)


# Stream management functions


def Stream(**kwargs):  # noqa: N802
    """Wrapper around a Neuron stream."""
    import torch_neuronx

    return torch_neuronx.Stream(**kwargs)


def stream(stream):
    """Context-manager that selects a given stream.

    Args:
        stream (Stream): A Neuron stream to select
    """
    import torch_neuronx

    return torch_neuronx.stream(stream)


def current_stream(device=None):
    """Return the currently selected Stream for a given device."""
    import torch_neuronx

    return torch_neuronx.current_stream(device)


def default_stream(device=None):
    """Return the default Stream for a given device."""
    import torch_neuronx

    return torch_neuronx.default_stream(device)


def set_stream(stream):
    """Set the current stream."""
    import torch_neuronx

    torch_neuronx.set_stream(stream)


def synchronize(device=None) -> None:
    """Wait for all kernels in all streams on a Neuron device to complete."""
    import torch_neuronx

    torch_neuronx.synchronize(device)


# Stream operations on current stream
def wait_stream(stream) -> None:
    """Make the current stream wait for another stream."""
    current_stream().wait_stream(stream)


def wait_event(event) -> None:
    """Make the current stream wait for an event."""
    current_stream().wait_event(event)


def record_event(event=None):
    """Record an event on the current stream."""
    return current_stream().record_event(event)


def query() -> bool:
    """Check if all work on the current stream has completed."""
    return current_stream().query()


def Event(enable_timing=False, blocking=False):  # noqa: N802
    """Create a Neuron event."""
    import torch_neuronx

    return torch_neuronx.Event(enable_timing=enable_timing, blocking=blocking)


# Memory metrics functions


def get_device_properties(device=None):
    """Get device properties for Neuron device.

    Args:
        device: Device module (optional)

    Returns:
        Device properties object with attributes:
        - name: Device name (str)
        - total_memory: Total device memory in bytes (int)
    """
    import torch_neuronx

    return torch_neuronx.get_device_properties(
        str(device) if device is not None else current_device()
    )


def reset_peak_memory_stats(device=None):
    """Reset peak memory statistics for Neuron device.

    Args:
        device: Device index (optional)
    """
    import torch_neuronx

    torch_neuronx.reset_peak_memory_stats(device)


def memory_stats(device=None):
    """Get comprehensive memory statistics for Neuron device.

    Args:
        device: Device index (optional)

    Returns:
        dict: Memory statistics including allocated_bytes, reserved_bytes, active_bytes, etc.
    """
    import torch_neuronx

    return torch_neuronx.memory_stats(device)


def memory_allocated(device=None):
    """Return current memory occupied by tensors in bytes.

    Args:
        device: Device index (optional)

    Returns:
        int: Current allocated bytes
    """
    import torch_neuronx

    return torch_neuronx.memory_allocated(device)


def max_memory_allocated(device=None):
    """Return peak memory occupied by tensors in bytes since last reset.

    Args:
        device: Device index (optional)

    Returns:
        int: Peak allocated bytes
    """
    import torch_neuronx

    return torch_neuronx.max_memory_allocated(device)


def memory_reserved(device=None):
    """Return current memory held in the caching allocator's pool in bytes.

    Args:
        device: Device index (optional)

    Returns:
        int: Current reserved bytes
    """
    import torch_neuronx

    return torch_neuronx.memory_reserved(device)


def max_memory_reserved(device=None):
    """Return peak memory held in the caching allocator's pool in bytes.

    Args:
        device: Device index (optional)

    Returns:
        int: Peak reserved bytes
    """
    import torch_neuronx

    return torch_neuronx.max_memory_reserved(device)


def reset_accumulated_memory_stats(device=None):
    """Reset accumulated memory statistics for Neuron device.

    Args:
        device: Device index (optional)
    """
    import torch_neuronx

    torch_neuronx.reset_accumulated_memory_stats(device)


def memory_summary(device=None, abbreviated=False):
    """Return a human-readable summary of memory usage for a device.

    Args:
        device: Device index (optional)
        abbreviated: If True, return a shorter summary

    Returns:
        str: Formatted memory summary
    """
    import torch_neuronx

    return torch_neuronx.memory_summary(device, abbreviated)


def empty_cache():
    """Clear Neuron device memory cache."""
    import torch_neuronx

    torch_neuronx.empty_cache()


def peak_memory_stats(device=None):
    """Return peak memory stats snapshot captured at the all-time allocation peak.

    Args:
        device: Device index (optional)

    Returns:
        dict: Peak stats snapshot with allocated/reserved/active bytes at peak,
              internal fragmentation, and OOM counters.
    """
    import torch_neuronx

    return torch_neuronx.peak_memory_stats(device)


def segment_pool_analysis(device=None):
    """Compute segment pool fragmentation and efficiency metrics.

    Args:
        device: Device index (optional)

    Returns:
        dict: Analysis with fragmentation_index, memory_efficiency,
              largest_free, free_chunks, address_span, etc.
    """
    import torch_neuronx

    return torch_neuronx.segment_pool_analysis(device)


def segment_pool_analysis_summary(device=None):
    """Return a formatted string of segment pool analysis.

    Args:
        device: Device index (optional)

    Returns:
        str: Formatted analysis summary
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
    """Get all torch.compile compilation metrics with timing."""
    import torch_neuronx

    return torch_neuronx.get_dynamo_metrics()


def reset_dynamo_metrics():
    """Reset all torch.compile compilation metrics."""
    import torch_neuronx

    torch_neuronx.reset_dynamo_metrics()


def is_bf16_supported() -> bool:
    """
    torch.cuda is patched to torch.neuron so bf16 is supported
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
