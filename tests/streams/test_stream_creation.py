"""
Test stream creation and basic properties for multi-stream support.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises, requires_nrt_streams


class TestStreamCreation:
    """Test basic stream creation functionality."""

    def test_default_stream_creation(self):
        """Test creating stream with default parameters."""
        stream = torch.neuron.Stream()
        assert stream.device.type == "neuron"
        assert stream.device.index is not None
        assert stream.device.index >= 0

    def test_stream_with_device_index(self):
        """Test creating stream with specific device."""
        stream = torch.neuron.Stream(device=0)
        assert stream.device.index == 0
        assert stream.device.type == "neuron"

    def test_stream_with_device_object(self):
        """Test creating stream with torch.device object."""
        device = torch.device("neuron:0")
        stream = torch.neuron.Stream(device=device)
        assert stream.device.index == 0
        assert stream.device.type == "neuron"

    def test_stream_with_priority(self):
        """Test creating stream with priority."""
        stream = torch.neuron.Stream(priority=1)
        # Priority is internal, just ensure no errors during creation
        assert stream is not None
        assert stream.device.type == "neuron"

    def test_stream_with_negative_priority(self):
        """Test creating stream with negative priority (higher priority)."""
        stream = torch.neuron.Stream(priority=-1)
        assert stream is not None
        assert stream.device.type == "neuron"

    @requires_nrt_streams
    def test_stream_equality(self):
        """Test stream equality comparison."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        # Different streams should not be equal
        assert stream1 != stream2

        # Same stream should be equal to itself
        assert stream1 == stream1
        assert stream2 == stream2

    @requires_nrt_streams
    def test_stream_hash(self):
        """Test stream hashing for use in sets/dicts."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        # Should be able to hash streams
        stream_set = {stream1, stream2}
        assert len(stream_set) == 2

        # Same stream should hash consistently
        stream_dict = {stream1: "test"}
        assert stream_dict[stream1] == "test"

    def test_stream_device_property(self):
        """Test stream device property returns correct device."""
        stream = torch.neuron.Stream()
        device = stream.device

        assert isinstance(device, torch.device)
        assert device.type == "neuron"
        assert device.index is not None

    @assert_raises((ValueError, RuntimeError, IndexError))
    def test_invalid_device_index(self):
        """Test creating stream with invalid device index raises error."""
        device_count = torch_neuronx.device_count()
        torch.neuron.Stream(device=device_count + 10)

    @assert_raises((ValueError, RuntimeError))
    def test_invalid_device_type(self):
        """Test creating stream with invalid device type raises error."""
        torch.neuron.Stream(device=torch.device("cpu"))

    def test_stream_repr(self):
        """Test stream string representation."""
        stream = torch.neuron.Stream()
        repr_str = repr(stream)

        # Should contain useful information
        assert "Stream" in repr_str or "neuron" in repr_str.lower()

    @requires_nrt_streams
    def test_multiple_stream_creation(self):
        """Test creating multiple streams doesn't interfere."""
        streams = []

        # Create multiple streams
        for _ in range(5):
            stream = torch.neuron.Stream()
            streams.append(stream)
            assert stream.device.type == "neuron"

        # All streams should be different
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                assert streams[i] != streams[j]

    @requires_nrt_streams
    def test_stream_creation_sequential_safety(self):
        """Test creating multiple streams sequentially is safe."""
        streams = []

        # Create streams sequentially
        for _ in range(10):
            stream = torch.neuron.Stream()
            streams.append(stream)
            assert stream.device.type == "neuron"

        # Should have correct number of streams
        assert len(streams) == 10

        # All streams should be valid and different
        for stream in streams:
            assert stream.device.type == "neuron"

        # All streams should be unique
        assert len(set(streams)) == len(streams)

    def test_single_stream_mode_default(self):
        """Test that single-stream mode routes all streams to default stream."""
        default_stream = torch_neuronx.default_stream()
        streams = [torch.neuron.Stream() for _ in range(5)]

        # All created streams should be the default stream in single-stream mode
        for stream in streams:
            assert (
                stream == default_stream
            ), "Single-stream mode should route all streams to default"
