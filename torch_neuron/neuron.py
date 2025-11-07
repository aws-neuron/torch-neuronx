# Neuron device module for PyTorch
# This module is accessed as torch.neuron and provides device-specific functionality


def is_initialized() -> bool:
    """Check if the Neuron runtime has been initialized."""
    import torch_neuron

    return torch_neuron.is_neuron_runtime_initialized()


def current_device() -> int:
    """Return the current Neuron device index."""
    import torch_neuron

    return torch_neuron._C._neuron_getCurrentDevice()


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index to set as current
    """
    import torch_neuron

    torch_neuron.set_device(device)


def device_count() -> int:
    """Return the number of Neuron devices available."""
    import torch_neuron

    return torch_neuron.device_count()


def is_available() -> bool:
    """Check if Neuron devices are available."""
    try:
        import torch_neuron

        return torch_neuron.device_count() > 0
    except Exception:
        return False


def init():
    """Initialize the Neuron runtime."""
    import torch_neuron

    torch_neuron._lazy_init()


def get_amp_supported_dtype():
    """Return list of dtypes supported by Neuron for automatic mixed precision."""
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]


def _is_in_bad_fork() -> bool:
    """Check if we're in a bad fork state."""
    import torch_neuron

    return torch_neuron._is_in_bad_fork


def manual_seed_all(seed: int) -> None:
    """Set seed for all neuron devices."""
    # Neuron devices use CPU's RNG (see NeuronHooksInterface::getDefaultGenerator)
    # CPU RNG is already seeded by torch.manual_seed() before calling this
    # No device-specific seeding needed
    _ = int(seed)  # Validate seed format


# Import device class from main module to avoid duplication
def device(device_spec):
    """Context-manager that changes the selected device.

    Args:
        device_spec: Device index (int) or torch.device object

    Returns:
        Context manager that switches to the specified device
    """
    import torch_neuron

    return torch_neuron.device(device_spec)


# Stream management functions


def Stream(**kwargs):  # noqa: N802
    """Wrapper around a Neuron stream."""
    import torch_neuron

    return torch_neuron.Stream(**kwargs)


def stream(stream):
    """Context-manager that selects a given stream.

    Args:
        stream (Stream): A Neuron stream to select
    """
    import torch_neuron

    return torch_neuron.stream(stream)


def current_stream(device=None):
    """Return the currently selected Stream for a given device."""
    import torch_neuron

    return torch_neuron.current_stream(device)


def default_stream(device=None):
    """Return the default Stream for a given device."""
    import torch_neuron

    return torch_neuron.default_stream(device)


def set_stream(stream):
    """Set the current stream."""
    import torch_neuron

    torch_neuron.set_stream(stream)


def synchronize(device=None) -> None:
    """Wait for all kernels in all streams on a Neuron device to complete."""
    import torch_neuron

    torch_neuron.synchronize(device)


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
    import torch_neuron

    return torch_neuron.Event(enable_timing=enable_timing, blocking=blocking)
