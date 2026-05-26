try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("torch-neuronx")
except Exception:
    from torch_neuronx._version import __version__


import os
import sys
import warnings

# Ensure the embedded torch_mlir is used instead of any standalone
# neuron-torch-mlir wheel. For source builds, prepend the embedded path.
# For wheel installs, torch_mlir is already bundled in the wheel.
_embedded_torch_mlir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "torch-mlir",
    "python",
)
if os.path.isdir(_embedded_torch_mlir):
    sys.path.insert(0, _embedded_torch_mlir)

import threading

# Global variable to track if module has been loaded
_is_loaded = False

# Thread-local context for transmitting the C++ dispatch cache key to Python
# lowering code without injecting it into kwargs (which breaks composite ops).
_dispatch_context = threading.local()


def _autoload():
    """Entry point for PyTorch autoload mechanism."""
    global _is_loaded
    if _is_loaded:
        return  # Already loaded

    # actual module import
    import torch_neuronx


import torch

# Store reference to real torch.device before any test framework patches it
_real_torch_device = torch.device

# Import build-time configuration (generated during wheel build)
try:
    from ._build_config import (
        DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED,
        DEFAULT_MLIR_ATEN_OPS,
        DEFAULT_RETAIN_DEVICE_MODE,
        DEFAULT_SYNC_MODE,
    )
except ImportError:
    # Fallback if _build_config.py doesn't exist (e.g., editable install)
    DEFAULT_SYNC_MODE = "0"  # Dev default
    DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED = "1"
    DEFAULT_MLIR_ATEN_OPS = "1"
    DEFAULT_RETAIN_DEVICE_MODE = "0"

# Set defaults from build-time configuration
# Users can still override these at runtime
os.environ.setdefault("NEURON_LAUNCH_BLOCKING", DEFAULT_SYNC_MODE)
os.environ.setdefault(
    "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED
)

# Set default for aten ops lowering
os.environ.setdefault("TORCH_NEURONX_MLIR_ATEN_OPS", DEFAULT_MLIR_ATEN_OPS)

# Set default for executor retain device
os.environ.setdefault("TORCH_NEURONX_RETAIN_DEVICE_MODE", DEFAULT_RETAIN_DEVICE_MODE)

# Set default for legacy neff
os.environ.setdefault("NEURON_RT_ALLOW_LEGACY_NEFF", "1")

# Set default for async nrt execution
os.environ.setdefault("NEURON_RT_DBG_ZEROCOPY", "1")

# Set default for HBM mapping
os.environ.setdefault("NEURON_RT_MAP_HBM", "1")

# Set default for host-cc
os.environ.setdefault("TORCH_NEURONX_ENABLE_HOST_CC", "0")
os.environ.setdefault("NEURON_RT_DISABLE_EXECUTION_BARRIER", "0")

# Default to C++ NeuronBackend process group
os.environ.setdefault("TORCH_NEURONX_CPP_PROCESS_GROUP", "1")

# Disable stack trace by default
os.environ.setdefault("TORCH_NEURONX_ENABLE_STACK_TRACE", "0")

# Set default debug directory
os.environ.setdefault("TORCH_NEURONX_DEBUG_DIR", "/tmp/neuron_backend")

# Disable artifact preservation by default
os.environ.setdefault("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "0")

# Lazy allocations by default.
os.environ.setdefault("TORCH_NEURONX_ENABLE_LAZY_ALLOC", "1")

# Import and register neuron device
try:
    from . import _C

    _C._register_device()
    _C._register_profiler()

    # Wrap neuron C extensions with torch._dynamo.disable since they can't be traced.
    import inspect

    for name in dir(_C):
        obj = getattr(_C, name)
        if callable(obj) and not inspect.isclass(obj):
            try:
                wrapped = torch._dynamo.disable(obj)
                setattr(_C, name, wrapped)
            except (TypeError, AttributeError):
                pass
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import torch_neuronx C++ extensions: {e}", stacklevel=2)
    raise

# Import the neuron submodule
# Import and register Python operations
# Import profiling functionality
from . import distributed, neuron, profiling, python_ops

# Import memory apis
from .memory import (
    _dump_snapshot,
    _record_memory_history,
    _snapshot,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    memory_stats_as_nested_dict,
    memory_summary,
    peak_memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
    segment_pool_analysis,
    segment_pool_analysis_summary,
)

# Import torch.compile metrics APIs
from .neuron_dynamo_backend.metrics import get_dynamo_metrics, reset_dynamo_metrics

# Import NKI kernel support
from .nki_hop import nki_op, wrap_nki
from .profiling import NeuronProfiler

# Import stream and event classes
from .streams import Event, Stream

# Register neuron as a custom backend
# This replaces PrivateUse1 with 'neuron'
torch.utils.rename_privateuse1_backend("neuron")

# Register the device module - this makes torch.neuron available
torch._register_device_module("neuron", neuron)

# Generate standard methods for the neuron backend
# This creates methods like tensor.neuron(), module.neuron(), etc.
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True
)

# Global state for lazy initialization
_initialized = False
_original_pid = 0
_initialization_lock = threading.RLock()
_is_in_bad_fork = False


def _lazy_init():
    """Initialize Neuron runtime if not already initialized."""
    global _initialized, _original_pid, _is_in_bad_fork

    if _initialized:
        return

    with _initialization_lock:
        # Double-checked locking
        if _initialized:
            return

        # Check for bad fork
        if _is_in_bad_fork:
            raise RuntimeError(
                "Cannot re-initialize Neuron runtime in forked subprocess. "
                "To use Neuron with multiprocessing, use the 'spawn' start method"
            )

        # Initialize the runtime
        _C._neuron_init()

        _original_pid = os.getpid()
        _initialized = True

        # Set any seed set via manual_seed before runtime was ready
        from torch_neuronx import neuron as _neuron_module

        idx = _neuron_module.current_device()
        gen = _C._get_default_generator(idx)
        _neuron_module._set_device_rng_seed(gen.initial_seed())


def _after_fork():
    """Mark that we're in a forked process where reinit is not allowed."""
    global _initialized, _is_in_bad_fork
    current_pid = os.getpid()
    if _initialized and _original_pid != current_pid:
        _initialized = False
        _is_in_bad_fork = True


# Register fork handler
os.register_at_fork(after_in_child=_after_fork)

# HLO persistent cache configuration (enabled by default, disable with TORCH_NEURONX_HLO_CACHE=0)
os.environ.setdefault("TORCH_NEURONX_HLO_CACHE", "1")

# Register Python operations first
python_ops.register_python_operations()

# Tell C++ that Python ops are now registered
_C._set_python_ops_registered(True)

# Apply mesh registry patch for SPMD collective legalization
# TODO: Remove this patch once PyTorch's compile_on_one_rank lands upstream
# (see https://github.com/pytorch/pytorch/pull/178116)
from .distributed.mesh_registry import _apply_mesh_registry_patch

_apply_mesh_registry_patch()


def is_neuron_runtime_initialized():
    return _C._is_neuron_runtime_initialized()


def device_count() -> int:
    """Return the number of Neuron devices (Virtual Neuron Cores) available."""
    return _C._neuron_getDeviceCount()


def current_device() -> int:
    """Return the current Neuron device index."""
    return _C._neuron_getCurrentDevice()


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index to set as current
    """
    # Validate device ID
    if device < 0 or device >= device_count():
        raise ValueError(f"Invalid device id: {device}. Valid range is 0 to {device_count() - 1}")

    _C._neuron_setDevice(device)


def get_device_properties(device):
    """
    Get the properties of a Neuron device.

    Args:
        device: Device index (int) or torch device string like 'neuron:0'
    Returns:
        Device properties object with attributes:
        - name: Device name (str)
        - total_memory: Total device memory in bytes (int)
    """
    if isinstance(device, str) and "neuron" in device:
        # Parse device string like 'neuron:0'
        import torch

        device_obj = torch.device(device)
        if device_obj.type != "neuron":
            raise ValueError(f"Expected neuron device, got {device_obj.type}")
        device_id = device_obj.index if device_obj.index is not None else current_device()
    else:
        device_id = int(device)

    # Validate device ID
    if device_id < 0 or device_id >= device_count():
        raise ValueError(f"Invalid device id: {device_id}")

    # Initialize runtime if needed (required for HBM memory queries)
    _lazy_init()

    return _C._neuron_getDeviceProperties(device_id)


def empty_cache() -> None:
    """
    Releases all cached memory blocks in the caching allocator.

    This will free all cached blocks back to the Neuron runtime, but
    will not affect memory of allocated tensors.
    """
    _C.NeuronCachingAllocator.emptyCache()


def get_fallback_ops() -> list:
    """
    Get the list of operations that fell back to CPU.

    Returns:
        List of operation names that were executed on CPU instead of Neuron.
    """
    return _C._get_fallback_ops()


def get_executed_ops() -> list:
    """
    Get the list of operations that were executed on Neuron device.

    Returns:
        List of operation names that were successfully executed on Neuron.
    """
    return _C._get_executed_ops()


def clear_op_tracking() -> None:
    """
    Clear fallback, executed, and concatenated operation tracking.

    This resets the internal tracking of which operations have been
    executed on Neuron vs CPU, and concatenated op mappings, useful for testing.
    """
    _C._clear_op_tracking()


def get_concatenated_ops() -> dict:
    """
    Get mapping of concatenated op names to their original constituent op names.

    When TORCH_NEURONX_ENABLE_CONCATENATION=1, operations may be merged into
    concatenated groups. This function returns which original ops were merged
    into each concatenated group.

    Returns:
        Dictionary mapping concatenated op name (str) to list of original op names (list[str]).
        Empty dict if no concatenation has occurred.
    """
    return _C._get_concatenated_ops()


def get_concatenated_op_execution_counts() -> dict:
    """
    Get per-op execution counts for concatenated operations.

    Each concatenated op's first execution requires compilation (uncached).
    Subsequent executions of the same concatenated group are cache hits.
    A count of 1 means the op was compiled but never re-executed from cache.
    A count > 1 means (count - 1) cache hits occurred.

    Returns:
        Dictionary mapping concatenated op name (str) to execution count (int).
        Empty dict if no concatenation has occurred.
    """
    return _C._get_concatenated_op_execution_counts()


def get_concatenation_profiles() -> dict:
    """
    Get per-concatenation-type profiling data.

    Returns detailed merge time and group size data for each unique
    concatenation type (identified by the pipe-separated op name).

    Returns:
        Dictionary mapping concat op name to profile dict with:
        - merge_times_ns: list[float] — merge time per invocation in nanoseconds
        - group_sizes: list[int] — number of ops per group per invocation
        - invocation_count: int — total number of times this concat type was created
        - avg_merge_time_ms: float — average merge time in milliseconds
        - avg_group_size: float — average group size
        Empty dict if no profiling data.
    """
    raw = _C._get_concatenation_profiles()
    result = {}
    for op_name, profile in raw.items():
        merge_times = profile["merge_times_ns"]
        group_sizes = profile["group_sizes"]
        count = profile["invocation_count"]
        avg_merge_ms = (sum(merge_times) / len(merge_times) / 1e6) if merge_times else 0.0
        avg_group = (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0
        result[op_name] = {
            "merge_times_ns": merge_times,
            "group_sizes": group_sizes,
            "invocation_count": count,
            "avg_merge_time_ms": avg_merge_ms,
            "avg_group_size": avg_group,
        }
    return result


def get_concatenation_status() -> dict:
    """
    Diagnose whether op concatenation is actually working.

    When TORCH_NEURONX_ENABLE_CONCATENATION=1, ops should be merged into
    concatenated groups whose names contain "|" (e.g.,
    "aten::add.Tensor|aten::linear_backward|...").

    This function checks both the executed ops (for "|" marker) and the
    ConcatenatedOpTracker (for explicit group recordings) to determine
    if concatenation is actually producing merged ops at runtime.

    Returns:
        Dictionary with diagnostic fields:
        - enabled: bool — whether the env var is set to "1"
        - concatenated_ops_in_executed: list[str] — executed op names containing "|"
        - individual_ops_in_executed: list[str] — executed op names without "|"
        - concatenated_groups_tracked: int — number of groups in ConcatenatedOpTracker
        - is_active: bool — True if concatenation produced at least one merged op
        - diagnosis: str — human-readable explanation of the status
    """
    import re

    enabled = os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"

    executed_ops = get_executed_ops()
    concat_pattern = re.compile(r".*\|.*")
    concatenated_ops_in_executed = [op for op in executed_ops if concat_pattern.match(op)]
    individual_ops_in_executed = [op for op in executed_ops if not concat_pattern.match(op)]

    concatenated_groups = get_concatenated_ops()
    num_groups_tracked = len(concatenated_groups)

    is_active = len(concatenated_ops_in_executed) > 0 or num_groups_tracked > 0

    # Build diagnosis
    if not enabled:
        diagnosis = (
            "Concatenation is DISABLED (TORCH_NEURONX_ENABLE_CONCATENATION != 1). "
            "Set TORCH_NEURONX_ENABLE_CONCATENATION=1 to enable."
        )
    elif is_active:
        diagnosis = (
            f"Concatenation is ACTIVE. "
            f"Found {len(concatenated_ops_in_executed)} concatenated op(s) in executed ops "
            f"and {num_groups_tracked} group(s) in ConcatenatedOpTracker."
        )
    else:
        diagnosis = (
            "Concatenation is ENABLED but NOT ACTIVE — no merged ops detected in executed ops "
            "(no '|' separator found) and ConcatenatedOpTracker has 0 groups. "
            "This may indicate dispatch overhead is preventing concatenation from triggering, "
            "or the workload does not have fusible op sequences."
        )

    return {
        "enabled": enabled,
        "concatenated_ops_in_executed": concatenated_ops_in_executed,
        "individual_ops_in_executed": individual_ops_in_executed,
        "concatenated_groups_tracked": num_groups_tracked,
        "is_active": is_active,
        "diagnosis": diagnosis,
    }


def _get_device_index(device, optional=False, allow_cpu=False):
    """Get device index from device specification."""
    if device is None:
        if optional:
            return None
        else:
            return current_device()

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(device, _real_torch_device):
        if device.type != "neuron":
            raise ValueError(f"Expected neuron device, got {device.type}")
        return device.index if device.index is not None else current_device()

    if isinstance(device, int):
        return device

    raise ValueError(f"Invalid device specification: {device}")


def synchronize(device=None) -> None:
    """Wait for all kernels in all streams on a Neuron device to complete."""
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    _C._neuron_synchronize(device_index)


def current_stream(device=None) -> Stream:
    """Return the currently selected Stream for a given device."""
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    streamdata = _C._neuron_getCurrentStream(device_index)
    return Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def default_stream(device=None) -> Stream:
    """Return the default Stream for a given device."""
    device_index = _get_device_index(device, optional=True)
    if device_index is None:
        device_index = -1
    streamdata = _C._neuron_getDefaultStream(device_index)
    return Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def set_stream(stream: Stream) -> None:
    """Set the current stream."""
    if stream is None:
        return
    _C._neuron_setStream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


class device:  # noqa: N801
    """Context-manager that changes the selected device.

    Args:
        device: Device index (int) or torch.device object

    Returns:
        Context manager that switches to the specified device
    """

    def __init__(self, device):
        self.idx = _get_device_index(device)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = current_device()
        if self.prev_idx != self.idx:
            set_device(self.idx)

    def __exit__(self, type, value, traceback):
        if self.prev_idx != self.idx and self.prev_idx != -1:
            set_device(self.prev_idx)


class StreamContext:
    """Context-manager that selects a given stream."""

    def __init__(self, stream):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting() and self.idx is None:
            # If no device index, use the stream's device index
            if stream is not None:
                self.idx = stream.device_index
            else:
                self.idx = -1

        self.src_prev_stream = None
        self.dst_prev_stream = None

    def __enter__(self):
        if self.stream is None:
            return

        # Always save the previous stream and set the new one
        self.src_prev_stream = current_stream(None)

        if self.src_prev_stream.device != self.stream.device:
            with device(self.stream.device):
                self.dst_prev_stream = current_stream(self.stream.device)
        set_stream(self.stream)

    def __exit__(self, type, value, traceback):
        if self.stream is None:
            return

        if self.src_prev_stream is not None:
            if (
                self.dst_prev_stream is not None
                and self.src_prev_stream.device != self.stream.device
            ):
                set_stream(self.dst_prev_stream)
            set_stream(self.src_prev_stream)


def stream(stream) -> StreamContext:
    """Wrap around the Context-manager StreamContext that selects a given stream."""
    return StreamContext(stream)
