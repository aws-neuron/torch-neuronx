__version__ = "0.1.0"

import os
import threading

# Global variable to track if module has been loaded
_is_loaded = False


def _autoload():
    """Entry point for PyTorch autoload mechanism."""
    global _is_loaded
    if _is_loaded:
        return  # Already loaded

    # actual module import
    import torch_neuronx


import torch

# Import build-time configuration (generated during wheel build)
try:
    from ._build_config import DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED, DEFAULT_SYNC_MODE
except ImportError:
    # Fallback if _build_config.py doesn't exist (e.g., editable install)
    DEFAULT_SYNC_MODE = "0"  # Dev default
    DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED = "1"

# For some reason log level is being set to TRACE without this
if "JAX_LOGGING_LEVEL" not in os.environ:
    os.environ["JAX_LOGGING_LEVEL"] = "WARNING"

# Set defaults from build-time configuration
# Users can still override these at runtime
os.environ.setdefault("TORCH_NEURONX_SYNC_MODE", DEFAULT_SYNC_MODE)
os.environ.setdefault(
    "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", DEFAULT_FALLBACK_ONLY_FOR_UNIMPLEMENTED
)

# Disable stack trace by default
os.environ.setdefault("TORCH_NEURONX_ENABLE_STACK_TRACE", "0")
# Import and register neuron device
try:
    from . import _C

    _C._register_device()
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import torch_neuronx C++ extensions: {e}", stacklevel=2)
    raise

# Import the neuron submodule
# Import and register Python operations
from . import distributed, neuron, python_ops

# Import stream and event classes
from .streams import Event, Stream

# Expose NeuronCachingAllocator for TorchNeuronNKIKernel
NeuronCachingAllocator = _C.NeuronCachingAllocator

# Import NKI kernel support
from .nki_kernel import TorchNeuronNKIKernel, nki, nki_op

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


def _after_fork(arg):
    """Mark that we're in a forked process where reinit is not allowed."""
    global _initialized, _is_in_bad_fork
    current_pid = os.getpid()
    if _initialized and _original_pid != current_pid:
        _initialized = False
        _is_in_bad_fork = True


# Register fork handler
os.register_at_fork(after_in_child=_after_fork)

# Register Python operations first
python_ops.register_python_operations()

# Tell C++ that Python ops are now registered
_C._set_python_ops_registered(True)


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
        device_id = device_obj.index if device_obj.index is not None else 0
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
    Clear both fallback and executed operation tracking.

    This resets the internal tracking of which operations have been
    executed on Neuron vs CPU, useful for testing.
    """
    _C._clear_op_tracking()


def _get_device_index(device, optional=False, allow_cpu=False):
    """Get device index from device specification."""
    if device is None:
        if optional:
            return None
        else:
            return current_device()

    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(device, torch.device):
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
