"""
Test stream synchronization functionality for multi-stream support.
"""

import time

import pytest
import torch

import torch_neuronx


class TestStreamSynchronization:
    """Test stream synchronization methods."""

    def test_stream_query_empty_stream(self):
        """Test stream query for completion status on empty stream."""
        stream = torch.neuron.Stream()
        # Empty stream should be complete
        assert stream.query()

    def test_stream_query_return_type(self):
        """Test stream query returns boolean."""
        stream = torch.neuron.Stream()
        result = stream.query()
        assert isinstance(result, bool)

    def test_stream_synchronize_empty_stream(self):
        """Test stream synchronization on empty stream."""
        stream = torch.neuron.Stream()
        # Should not raise exception and should complete quickly
        start_time = time.time()
        stream.synchronize()
        end_time = time.time()

        # Should complete very quickly for empty stream
        assert (end_time - start_time) < 1.0

    def test_stream_synchronize_with_operations(self):
        """Test stream synchronization with actual operations."""
        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            # Submit some operations
            tensor1 = torch.ones(50, 50, device="neuron")
            result1 = tensor1 * 2
            tensor2 = torch.ones(50, 50, device="neuron")
            result2 = tensor2 + tensor1

        # Stream should not be complete immediately (operations might be async)
        # But synchronize should wait for completion
        stream.synchronize()

        # After synchronization, stream should be complete
        assert stream.query()

        # Results should be available
        expected = torch.ones(50, 50) * 2
        torch.testing.assert_close(result1.cpu(), expected)
        torch.testing.assert_close(result2.cpu(), expected)

    def test_wait_stream_basic(self):
        """Test waiting for another stream."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        # Should not raise exception
        stream1.wait_stream(stream2)

    def test_wait_stream_with_operations(self):
        """Test stream waiting with actual operations."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()
        results = {}

        # Submit operation to stream1
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(30, 30, device="neuron")
            results["stream1"] = tensor1 * 2

        # Make stream2 wait for stream1, then do its own work
        with torch.neuron.stream(stream2):
            stream2.wait_stream(stream1)
            tensor2 = torch.ones(30, 30, device="neuron")
            results["stream2"] = tensor2 * 3

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()

        # Both results should be correct
        expected1 = torch.ones(30, 30) * 2
        expected2 = torch.ones(30, 30) * 3
        torch.testing.assert_close(results["stream1"].cpu(), expected1)
        torch.testing.assert_close(results["stream2"].cpu(), expected2)

    def test_wait_stream_same_stream(self):
        """Test stream waiting for itself (should be no-op)."""
        stream = torch.neuron.Stream()

        # Waiting for itself should not cause deadlock
        stream.wait_stream(stream)

        # Should still be able to use the stream
        with torch.neuron.stream(stream):
            tensor = torch.ones(10, 10, device="neuron")
            result = tensor * 2

        stream.synchronize()
        expected = torch.ones(10, 10) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_multiple_stream_synchronization(self):
        """Test synchronizing multiple streams."""
        streams = [torch.neuron.Stream() for _ in range(3)]
        results = []

        # Submit operations to each stream
        for i, stream in enumerate(streams):
            with torch.neuron.stream(stream):
                tensor = torch.ones(20, 20, device="neuron")
                result = tensor * (i + 1)
                results.append(result)

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

        # All results should be correct
        for i, result in enumerate(results):
            expected = torch.ones(20, 20) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_global_device_synchronize(self):
        """Test global device synchronization."""
        # Create multiple streams and submit operations
        streams = [torch.neuron.Stream() for _ in range(2)]
        results = []

        for i, stream in enumerate(streams):
            with torch.neuron.stream(stream):
                tensor = torch.ones(15, 15, device="neuron")
                result = tensor * (i + 2)
                results.append(result)

        # Global synchronize should wait for all streams
        torch_neuronx.synchronize()

        # All streams should be complete
        for stream in streams:
            assert stream.query()

        # All results should be correct
        for i, result in enumerate(results):
            expected = torch.ones(15, 15) * (i + 2)
            torch.testing.assert_close(result.cpu(), expected)

    def test_synchronization_multi_stream_safety(self):
        """Test synchronization across multiple streams."""
        streams = [torch.neuron.Stream() for _ in range(3)]
        results = {}

        # Run operations on different streams
        for i, stream in enumerate(streams, 1):
            with torch.neuron.stream(stream):
                tensor = torch.ones(10, 10, device="neuron")
                result = tensor * i
                results[i] = result

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

        # Verify all results are correct after sync
        for i in range(1, 4):
            expected = torch.ones(10, 10) * i
            torch.testing.assert_close(results[i].cpu(), expected)

        assert len(results) == 3

    def test_query_during_operations(self):
        """Test querying stream status during operations."""
        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            # Submit a larger operation that might take some time
            tensor = torch.randn(100, 100, device="neuron")
            torch.mm(tensor, tensor.t())

        # Query should return a boolean (might be True or False depending on timing)
        query_result = stream.query()
        assert isinstance(query_result, bool)

        # After synchronization, should definitely be True
        stream.synchronize()
        assert stream.query()

    def test_synchronize_timeout_behavior(self):
        """Test synchronization doesn't hang indefinitely."""
        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            # Submit some operations
            tensor = torch.ones(25, 25, device="neuron")
            result = tensor * 2

        # Synchronization should complete within reasonable time
        start_time = time.time()
        stream.synchronize()
        end_time = time.time()

        # Should not take too long (adjust threshold as needed)
        assert (end_time - start_time) < 30.0, "Synchronization took too long"

        # Result should be correct
        expected = torch.ones(25, 25) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_nested_synchronization(self):
        """Test nested synchronization calls."""
        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            tensor = torch.ones(10, 10, device="neuron")
            result = tensor * 2

        # Multiple synchronization calls should be safe
        stream.synchronize()
        stream.synchronize()  # Second call should be no-op

        # Result should still be correct
        expected = torch.ones(10, 10) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_cross_stream_dependencies(self):
        """Test complex cross-stream dependencies."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()
        stream3 = torch.neuron.Stream()

        # Create a dependency chain: stream1 -> stream2 -> stream3
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(20, 20, device="neuron")
            result1 = tensor1 * 2

        with torch.neuron.stream(stream2):
            stream2.wait_stream(stream1)
            tensor2 = torch.ones(20, 20, device="neuron")
            result2 = tensor2 * 3

        with torch.neuron.stream(stream3):
            stream3.wait_stream(stream2)
            tensor3 = torch.ones(20, 20, device="neuron")
            result3 = tensor3 * 4

        # Synchronize final stream should wait for all dependencies
        stream3.synchronize()

        # All results should be correct
        expected1 = torch.ones(20, 20) * 2
        expected2 = torch.ones(20, 20) * 3
        expected3 = torch.ones(20, 20) * 4

        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)
        torch.testing.assert_close(result3.cpu(), expected3)

    def test_synchronization_with_default_stream(self):
        """Test synchronization behavior with default stream."""
        default_stream = torch_neuronx.default_stream()
        custom_stream = torch.neuron.Stream()

        # Submit work to default stream
        with torch.neuron.stream(default_stream):
            tensor1 = torch.ones(15, 15, device="neuron")
            result1 = tensor1 * 2

        # Submit work to custom stream that waits for default
        with torch.neuron.stream(custom_stream):
            custom_stream.wait_stream(default_stream)
            tensor2 = torch.ones(15, 15, device="neuron")
            result2 = tensor2 * 3

        # Synchronize custom stream
        custom_stream.synchronize()

        # Both results should be correct
        expected1 = torch.ones(15, 15) * 2
        expected2 = torch.ones(15, 15) * 3

        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)
