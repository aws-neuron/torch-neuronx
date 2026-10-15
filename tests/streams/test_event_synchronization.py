"""
Test event synchronization and timing functionality for multi-stream support.
"""

import time

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


class TestEventSynchronization:
    """Test event synchronization and timing functionality."""

    def test_event_elapsed_time_basic(self):
        """Test basic elapsed time measurement between events."""
        start_event = torch.neuron.Event(enable_timing=True)
        end_event = torch.neuron.Event(enable_timing=True)

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            start_event.record()

            # Perform some computation
            tensor = torch.ones(50, 50, device="neuron")
            tensor = torch.mm(tensor, tensor.t())

            end_event.record()

        stream.synchronize()

        # Measure elapsed time
        elapsed = start_event.elapsed_time(end_event)

        # Should return non-negative time
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)

    @assert_raises(RuntimeError, match="Event was not created with timing enabled")
    def test_event_elapsed_time_no_timing(self):
        """Test elapsed time with events that don't have timing enabled."""
        start_event = torch.neuron.Event(enable_timing=False)
        end_event = torch.neuron.Event(enable_timing=False)

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            start_event.record()
            tensor = torch.ones(10, 10, device="neuron")
            tensor = tensor * 2
            end_event.record()

        stream.synchronize()

        # Should raise RuntimeError when timing is not enabled
        start_event.elapsed_time(end_event)

    def test_event_elapsed_time_same_event(self):
        """Test elapsed time between same event (should be 0)."""
        event = torch.neuron.Event(enable_timing=True)

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            event.record()

        stream.synchronize()

        # Elapsed time from event to itself should be 0
        elapsed = event.elapsed_time(event)
        assert elapsed == 0.0

    def test_event_elapsed_time_multiple_operations(self):
        """Test elapsed time across multiple operations."""
        events = [torch.neuron.Event(enable_timing=True) for _ in range(3)]

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            events[0].record()

            # First operation
            tensor1 = torch.ones(30, 30, device="neuron")
            tensor1 = tensor1 * 2
            events[1].record()

            # Second operation
            tensor2 = torch.ones(30, 30, device="neuron")
            tensor2 = tensor2 * 3
            events[2].record()

        stream.synchronize()

        # Measure different intervals
        total_time = events[0].elapsed_time(events[2])
        first_op_time = events[0].elapsed_time(events[1])
        second_op_time = events[1].elapsed_time(events[2])

        # All should be non-negative
        assert total_time >= 0.0
        assert first_op_time >= 0.0
        assert second_op_time >= 0.0

        # Total time should be at least the sum of individual times
        # (allowing for some measurement precision issues)
        assert total_time >= (first_op_time + second_op_time - 1.0)

    def test_event_timing_accuracy(self):
        """Test timing accuracy with known delays."""
        start_event = torch.neuron.Event(enable_timing=True)
        end_event = torch.neuron.Event(enable_timing=True)

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            start_event.record()

            # Perform operations that should take some measurable time
            for _ in range(5):
                tensor = torch.randn(100, 100, device="neuron")
                tensor = torch.mm(tensor, tensor.t())

            end_event.record()

        stream.synchronize()

        elapsed = start_event.elapsed_time(end_event)

        # Should have taken some measurable time (more than 0)
        assert elapsed > 0.0

    def test_event_timing_cross_stream(self):
        """Test timing measurements across different streams."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        start_event = torch.neuron.Event(enable_timing=True)
        mid_event = torch.neuron.Event(enable_timing=True)
        end_event = torch.neuron.Event(enable_timing=True)

        # Record events on different streams
        with torch.neuron.stream(stream1):
            start_event.record()
            tensor1 = torch.ones(40, 40, device="neuron")
            tensor2 = tensor1 * 2
            mid_event.record()

        with torch.neuron.stream(stream2):
            stream2.wait_event(mid_event)  # Wait for stream1
            tensor2 = torch.ones(40, 40, device="neuron")
            tensor2 = tensor2 * 3
            end_event.record()

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()

        # Measure total time across streams
        total_time = start_event.elapsed_time(end_event)
        stream1_time = start_event.elapsed_time(mid_event)

        assert total_time >= 0.0
        assert stream1_time >= 0.0
        assert total_time >= stream1_time

    def test_event_synchronization_ordering(self):
        """Test event synchronization maintains proper ordering."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        event1 = torch.neuron.Event()
        event2 = torch.neuron.Event()

        results = {}

        # Submit work to stream1
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(20, 20, device="neuron")
            results["stream1"] = tensor1 * 2
            event1.record()

        # Submit work to stream2 that depends on stream1
        with torch.neuron.stream(stream2):
            event1.wait(stream2)  # Wait for stream1 to complete
            tensor2 = torch.ones(20, 20, device="neuron")
            results["stream2"] = tensor2 * 3
            event2.record()

        # Synchronize in order
        event1.synchronize()
        event2.synchronize()

        # Both results should be correct
        expected1 = torch.ones(20, 20) * 2
        expected2 = torch.ones(20, 20) * 3

        torch.testing.assert_close(results["stream1"].cpu(), expected1)
        torch.testing.assert_close(results["stream2"].cpu(), expected2)

    def test_event_synchronization_reverse_order(self):
        """Test synchronizing events in reverse order."""
        stream = torch.neuron.Stream()
        events = [torch.neuron.Event() for _ in range(3)]
        results = []

        with torch.neuron.stream(stream):
            for i, event in enumerate(events):
                tensor = torch.ones(15, 15, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

        # Synchronize events in reverse order
        for event in reversed(events):
            event.synchronize()

        # All results should still be correct
        for i, result in enumerate(results):
            expected = torch.ones(15, 15) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_wait_chain(self):
        """Test chain of event dependencies."""
        streams = [torch.neuron.Stream() for _ in range(4)]
        events = [torch.neuron.Event() for _ in range(4)]
        results = []

        # Create a chain: stream0 -> stream1 -> stream2 -> stream3
        for i, (stream, event) in enumerate(zip(streams, events, strict=False)):
            with torch.neuron.stream(stream):
                if i > 0:
                    events[i - 1].wait(stream)  # Wait for previous stream

                tensor = torch.ones(12, 12, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

        # Synchronize final event (should wait for entire chain)
        events[-1].synchronize()

        # All results should be correct
        for i, result in enumerate(results):
            expected = torch.ones(12, 12) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_timing_multi_stream_safety(self):
        """Test event timing operations across multiple streams."""
        streams = [torch.neuron.Stream() for _ in range(3)]
        timing_results = {}

        # Run timing operations on different streams sequentially
        for i in range(3):
            start_event = torch.neuron.Event(enable_timing=True)
            end_event = torch.neuron.Event(enable_timing=True)
            stream = streams[i]

            with torch.neuron.stream(stream):
                start_event.record()

                # Do some work
                tensor = torch.ones(25, 25, device="neuron")
                tensor = torch.mm(tensor, tensor.t())

                end_event.record()

            stream.synchronize()

            # Measure timing
            elapsed = start_event.elapsed_time(end_event)
            timing_results[i] = elapsed

        # Check results
        assert len(timing_results) == 3

        # All timing results should be reasonable
        for _, elapsed in timing_results.items():
            assert elapsed >= 0.0

    def test_event_synchronization_with_exceptions(self):
        """Test event synchronization behavior with exceptions."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        try:
            with torch.neuron.stream(stream):
                tensor = torch.ones(10, 10, device="neuron")
                tensor = tensor * 2
                event.record()

                # Simulate an exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Event should still be synchronizable
        event.synchronize()

    def test_event_timing_precision(self):
        """Test event timing precision with very short operations."""
        start_event = torch.neuron.Event(enable_timing=True)
        end_event = torch.neuron.Event(enable_timing=True)

        stream = torch.neuron.Stream()

        with torch.neuron.stream(stream):
            start_event.record()

            # Very simple operation
            tensor = torch.ones(5, 5, device="neuron")
            tensor = tensor * 1.0

            end_event.record()

        stream.synchronize()

        elapsed = start_event.elapsed_time(end_event)

        # Should be non-negative and finite
        assert elapsed >= 0.0
        assert elapsed != float("inf")
        assert not torch.isnan(torch.tensor(elapsed))

    def test_event_multiple_wait_same_event(self):
        """Test multiple streams waiting for same event."""
        source_stream = torch.neuron.Stream()
        wait_streams = [torch.neuron.Stream() for _ in range(3)]

        event = torch.neuron.Event()
        results = []

        # Record event on source stream
        with torch.neuron.stream(source_stream):
            tensor = torch.ones(18, 18, device="neuron")
            source_result = tensor * 2
            event.record()

        # Multiple streams wait for same event
        for i, stream in enumerate(wait_streams):
            with torch.neuron.stream(stream):
                event.wait(stream)
                tensor = torch.ones(18, 18, device="neuron")
                result = tensor * (i + 3)  # 3, 4, 5
                results.append(result)

        # Synchronize all streams
        source_stream.synchronize()
        for stream in wait_streams:
            stream.synchronize()

        # All results should be correct
        expected_source = torch.ones(18, 18) * 2
        torch.testing.assert_close(source_result.cpu(), expected_source)

        for i, result in enumerate(results):
            expected = torch.ones(18, 18) * (i + 3)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_timing_with_synchronization(self):
        """Test timing measurements with explicit synchronization."""
        start_event = torch.neuron.Event(enable_timing=True)
        sync_event = torch.neuron.Event(enable_timing=True)
        end_event = torch.neuron.Event(enable_timing=True)

        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()

        with torch.neuron.stream(stream1):
            start_event.record()
            tensor1 = torch.ones(35, 35, device="neuron")
            tensor1 = tensor1 * 2
            sync_event.record()

        with torch.neuron.stream(stream2):
            sync_event.wait(stream2)
            tensor2 = torch.ones(35, 35, device="neuron")
            tensor2 = tensor2 * 3
            end_event.record()

        # Synchronize everything
        stream1.synchronize()
        stream2.synchronize()

        # Measure different time intervals
        total_time = start_event.elapsed_time(end_event)
        first_part = start_event.elapsed_time(sync_event)

        assert total_time >= 0.0
        assert first_part >= 0.0
        assert total_time >= first_part

    def test_event_synchronization_cleanup(self):
        """Test proper cleanup of event synchronization resources."""
        # Create many events and streams to test resource management
        events = [torch.neuron.Event() for _ in range(10)]
        streams = [torch.neuron.Stream() for _ in range(5)]

        # Use events across multiple streams
        for i, event in enumerate(events):
            stream = streams[i % len(streams)]

            with torch.neuron.stream(stream):
                tensor = torch.ones(8, 8, device="neuron")
                _ = tensor * (i + 1)
                event.record()

        # Synchronize all events
        for event in events:
            event.synchronize()
