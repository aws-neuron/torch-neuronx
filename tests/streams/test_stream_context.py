"""
Test stream context management for multi-stream support.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import requires_nrt_streams


class TestStreamContext:
    """Test stream context manager functionality."""

    @requires_nrt_streams
    def test_stream_context_manager(self):
        """Test stream context manager switches current stream."""
        default_stream = torch_neuronx.current_stream()
        new_stream = torch.neuron.Stream()

        # Streams should be different
        assert new_stream != default_stream

        with torch.neuron.stream(new_stream):
            current = torch_neuronx.current_stream()
            assert current == new_stream

        # Should restore original stream
        restored_stream = torch_neuronx.current_stream()
        assert restored_stream == default_stream

    def test_nested_stream_contexts(self):
        """Test nested stream context managers."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()
        default_stream = torch_neuronx.current_stream()

        with torch.neuron.stream(stream1):
            assert torch_neuronx.current_stream() == stream1
            with torch.neuron.stream(stream2):
                assert torch_neuronx.current_stream() == stream2
            # Should restore stream1
            assert torch_neuronx.current_stream() == stream1
        # Should restore default stream
        assert torch_neuronx.current_stream() == default_stream

    def test_stream_context_with_none(self):
        """Test stream context manager with None stream."""
        default_stream = torch_neuronx.current_stream()

        # Using None should not change the current stream
        with torch.neuron.stream(None):
            current = torch_neuronx.current_stream()
            assert current == default_stream

        # Should still be the same
        assert torch_neuronx.current_stream() == default_stream

    def test_stream_context_exception_handling(self):
        """Test stream context manager properly restores on exception."""
        default_stream = torch_neuronx.current_stream()
        new_stream = torch.neuron.Stream()

        try:
            with torch.neuron.stream(new_stream):
                assert torch_neuronx.current_stream() == new_stream
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore original stream even after exception
        assert torch_neuronx.current_stream() == default_stream

    def test_stream_context_same_stream(self):
        """Test stream context manager with same stream as current."""
        current_stream = torch_neuronx.current_stream()

        with torch.neuron.stream(current_stream):
            # Should still be the same stream
            assert torch_neuronx.current_stream() == current_stream

        # Should still be the same stream
        assert torch_neuronx.current_stream() == current_stream

    def test_multiple_context_managers_same_stream(self):
        """Test multiple context managers using the same stream."""
        new_stream = torch.neuron.Stream()
        default_stream = torch_neuronx.current_stream()

        # First context
        with torch.neuron.stream(new_stream):
            assert torch_neuronx.current_stream() == new_stream

        assert torch_neuronx.current_stream() == default_stream

        # Second context with same stream
        with torch.neuron.stream(new_stream):
            assert torch_neuronx.current_stream() == new_stream

        assert torch_neuronx.current_stream() == default_stream

    def test_stream_context_cross_device(self):
        """Test stream context manager with streams from different devices."""
        if torch_neuronx.device_count() > 1:
            stream0 = torch.neuron.Stream(device=0)
            stream1 = torch.neuron.Stream(device=1)
            default_stream = torch_neuronx.current_stream()

            with torch.neuron.stream(stream0):
                assert torch_neuronx.current_stream() == stream0
                with torch.neuron.stream(stream1):
                    assert torch_neuronx.current_stream() == stream1
                assert torch_neuronx.current_stream() == stream0
            assert torch_neuronx.current_stream() == default_stream
        else:
            pytest.skip("Need multiple Neuron devices for cross-device test")

    def test_stream_context_multi_stream_safety(self):
        """Test stream context manager with multiple streams sequentially."""
        streams = [torch.neuron.Stream() for _ in range(3)]
        results = {}

        # Test contexts sequentially with different streams
        for i, stream in enumerate(streams):
            with torch.neuron.stream(stream):
                # Verify we're using the correct stream
                current = torch_neuronx.current_stream()
                results[i] = current == stream
                # Verify stream is still correct after some operations
                tensor = torch.ones(5, 5, device="neuron")
                result = tensor * (i + 1)
                assert result is not None
                current_after = torch_neuronx.current_stream()
                results[f"{i}_after"] = current_after == stream

        # Check results
        for i in range(len(streams)):
            assert results[i], f"Stream {i} didn't use correct stream"
            assert results[f"{i}_after"], f"Stream {i} context changed"

    def test_stream_context_with_operations(self):
        """Test stream context manager with actual tensor operations."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        # Create tensors on different streams
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(10, 10, device="neuron")
            result1 = tensor1 * 2

        with torch.neuron.stream(stream2):
            tensor2 = torch.ones(10, 10, device="neuron")
            result2 = tensor2 * 3

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()

        # Verify results
        expected1 = torch.ones(10, 10) * 2
        expected2 = torch.ones(10, 10) * 3

        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)

    def test_default_stream_context(self):
        """Test using default stream in context manager."""
        default_stream = torch_neuronx.default_stream()
        current_stream = torch_neuronx.current_stream()

        # Default and current might be the same initially
        with torch.neuron.stream(default_stream):
            context_stream = torch_neuronx.current_stream()
            assert context_stream == default_stream

        # Should restore to whatever was current before
        assert torch_neuronx.current_stream() == current_stream

    def test_stream_context_manager_properties(self):
        """Test StreamContext properties and behavior."""
        stream = torch.neuron.Stream()
        context = torch.neuron.stream(stream)

        # Context should have the stream
        assert context.stream == stream

        # Should be usable as context manager
        with context:
            assert torch_neuronx.current_stream() == stream
