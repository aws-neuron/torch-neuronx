"""
Test suite for record_stream functionality in TorchNeuronx.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


class TestRecordStream:
    """Test class for record_stream functionality."""

    def test_basic_record_stream(self):
        """Test basic record_stream functionality."""
        # Create a tensor on neuron device
        tensor = torch.randn(10, 10, device="neuron")

        # Create a new stream
        stream = torch.neuron.Stream()
        assert stream.device_index == 0

        # Test with a new stream
        tensor.record_stream(stream)

        # Test with current stream
        current_stream = torch.neuron.current_stream()
        tensor.record_stream(current_stream)

    def test_multi_stream_recording(self):
        """Test record_stream with multiple streams."""
        tensor = torch.randn(5, 5, device="neuron")

        # Create multiple streams
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        # Record tensor usage on both streams
        tensor.record_stream(stream1)
        tensor.record_stream(stream2)

        # Synchronize streams
        stream1.synchronize()
        stream2.synchronize()

    def test_dispatcher_integration(self):
        """Test that record_stream works through PyTorch's dispatcher."""
        tensor = torch.randn(3, 3, device="neuron")

        # Test different stream types
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.current_stream()
        stream3 = torch.neuron.default_stream()

        # All should work through dispatcher
        tensor.record_stream(stream1)
        tensor.record_stream(stream2)
        tensor.record_stream(stream3)

    @assert_raises(
        NotImplementedError,
        match="Could not run 'aten::record_stream' with arguments from the 'CPU' backend",
    )
    def test_invalid_tensor_device(self):
        """Test record_stream with non-neuron tensor."""

        cpu_tensor = torch.randn(3, 3)
        stream = torch.neuron.Stream()

        # Should raise error for non-neuron tensor (CPU backend doesn't support record_stream)
        cpu_tensor.record_stream(stream)

    @assert_raises(RuntimeError, match="unknown parameter type")
    def test_invalid_stream_device(self):
        """Test record_stream with invalid stream."""
        tensor = torch.randn(3, 3, device="neuron")

        # Test with None stream - should raise RuntimeError
        tensor.record_stream(None)

    def test_stream_conversion(self):
        """Test c10::Stream conversion in record_stream."""
        tensor = torch.randn(2, 2, device="neuron")

        # Create stream and verify it works
        stream = torch.neuron.Stream()
        assert hasattr(stream, "stream_id")
        assert hasattr(stream, "device_index")

        # Should work without error
        tensor.record_stream(stream)


class TestRecordStreamEdgeCases:
    """Test edge cases for record_stream functionality."""

    def test_zero_size_tensor(self):
        """Test record_stream with zero-size tensor."""
        tensor = torch.empty(0, device="neuron")
        stream = torch.neuron.Stream()

        # Should work even with zero-size tensor
        tensor.record_stream(stream)

    def test_same_stream_multiple_times(self):
        """Test recording the same stream multiple times."""
        tensor = torch.randn(3, 3, device="neuron")
        stream = torch.neuron.Stream()

        # Recording same stream multiple times should be safe
        tensor.record_stream(stream)
        tensor.record_stream(stream)
        tensor.record_stream(stream)

    def test_default_stream_recording(self):
        """Test recording with default stream."""
        tensor = torch.randn(2, 2, device="neuron")
        default_stream = torch.neuron.default_stream()

        # Should work with default stream
        tensor.record_stream(default_stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
