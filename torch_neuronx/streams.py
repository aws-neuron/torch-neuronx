from typing import Optional

import torch

from . import _C

_NeuronStreamBase = _C._NeuronStreamBase
_NeuronEventBase = _C._NeuronEventBase


class Stream(_NeuronStreamBase):
    """Wrapper around a Neuron stream.

    A Neuron stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If device is None (default) or a negative integer,
            this will use the current device.
        priority(int, optional): priority of the stream. Lower numbers
            represent higher priorities.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # Check if we're creating from existing stream parameters
        if "stream_id" in kwargs and "device_index" in kwargs and "device_type" in kwargs:
            return super().__new__(
                cls,
                stream_id=kwargs["stream_id"],
                device_index=kwargs["device_index"],
                device_type=kwargs["device_type"],
            )

        # Setting device manager is expensive, so we avoid it unless necessary
        if device is None:
            return super().__new__(cls, priority=priority)
        else:
            # TODO: Fix properly, right now we import here to avoid circular imports
            import torch_neuronx

            with torch_neuronx.device(device):
                return super().__new__(cls, priority=priority)

    def wait_event(self, event: "Event") -> None:
        """Make all future work submitted to the stream wait for an event.

        Args:
            event: The event to wait for
        """
        super().wait_event(event)

    def wait_stream(self, stream: "Stream") -> None:
        """Synchronize with another stream.

        Args:
            stream: The stream to wait for
        """
        self.wait_event(stream.record_event())

    def record_event(self, event: Optional["Event"] = None) -> "Event":
        """Record an event.

        Args:
            event: Event to record (None to create new event)

        Returns:
            The recorded event
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        """Check if all the work submitted has been completed."""
        return super().query()

    def synchronize(self) -> None:
        """Wait for all the kernels in this stream to complete."""
        return super().synchronize()

    @property
    def device(self):
        """Return the device associated with this stream."""
        return torch.device("neuron", self.device_index)


class Event(_NeuronEventBase):
    """Wrapper around a Neuron event.

    Neuron events are synchronization markers that can be used to monitor
    device execution, accurately measure timing, and synchronize streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
    """

    def __new__(cls, enable_timing=False, blocking=False):
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
        )

    def record(self, stream: Optional["Stream"] = None) -> None:
        """Record the event in a given stream.

        Args:
            stream: Stream to record on (None for current stream)
        """
        if stream is None:
            # Import here to avoid circular import
            import torch_neuronx

            stream = torch_neuronx.current_stream()
        super().record(stream)

    def wait(self, stream: Optional["Stream"] = None) -> None:
        """Make all future work submitted to the given stream wait for this event.

        Args:
            stream: Stream to wait on (None for current stream)
        """
        if stream is None:
            # Import here to avoid circular import
            import torch_neuronx

            stream = torch_neuronx.current_stream()
        super().wait(stream)

    def query(self) -> bool:
        """Check if all work currently captured by event has completed."""
        return super().query()

    def elapsed_time(self, end_event: "Event") -> float:
        """Return the time elapsed in milliseconds between the event and end_event.

        Args:
            end_event: The end event to measure elapsed time to

        Returns:
            Elapsed time in milliseconds
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        """Wait for the event to complete."""
        super().synchronize()
