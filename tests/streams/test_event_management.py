"""
Test event management functionality for multi-stream support.
"""

import time

import pytest
import torch

import torch_neuronx


class TestEventManagement:
    """Test event creation and basic functionality."""

    def test_event_creation_basic(self):
        """Test basic event creation with default parameters."""
        event = torch.neuron.Event()
        assert event is not None
        event = torch.neuron.Event(enable_timing=True)
        assert event is not None
        event = torch.neuron.Event(blocking=True)
        assert event is not None
        event = torch.neuron.Event(enable_timing=True, blocking=True)
        assert event is not None

    def test_event_record_on_stream(self):
        """Test recording event on specific stream."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        # Should not raise exception
        event.record(stream)

    def test_event_record_current_stream(self):
        """Test recording event on current stream."""
        event = torch.neuron.Event()
        # Should not raise exception
        event.record()

    def test_event_record_with_operations(self):
        """Test recording event after operations."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        with torch.neuron.stream(stream):
            tensor = torch.ones(10, 10, device="neuron")
            result = tensor * 2
            event.record()

        # Event should be recorded
        stream.synchronize()

        # Result should be correct
        expected = torch.ones(10, 10) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_multiple_events_same_stream(self):
        """Test recording multiple events on same stream."""
        stream = torch.neuron.Stream()
        events = [torch.neuron.Event() for _ in range(3)]
        results = []

        with torch.neuron.stream(stream):
            for i, event in enumerate(events):
                tensor = torch.ones(5, 5, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

        stream.synchronize()

        # Validate that all events are recorded and completed
        for i, event in enumerate(events):
            # Event should be completed after stream synchronization
            assert event.query(), f"Event {i} should be completed"

            # Event synchronization should not block (already completed)
            event.synchronize()  # Should return immediately

        # Validate that the operations completed correctly
        for i, result in enumerate(results):
            expected = torch.ones(5, 5) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_record_different_streams(self):
        """Test recording events on different streams."""
        streams = [torch.neuron.Stream() for _ in range(2)]
        events = [torch.neuron.Event() for _ in range(2)]

        for i, (stream, event) in enumerate(zip(streams, events, strict=False)):
            with torch.neuron.stream(stream):
                tensor = torch.ones(8, 8, device="neuron")
                tensor = tensor * (i + 1)
                event.record()

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()

    def test_event_query_unrecorded(self):
        """Test querying unrecorded event."""
        event = torch.neuron.Event()
        # Unrecorded event behavior - should return boolean
        result = event.query()
        assert isinstance(result, bool)

    def test_event_query_recorded(self):
        """Test querying recorded event."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        with torch.neuron.stream(stream):
            tensor = torch.ones(5, 5, device="neuron")
            tensor = tensor * 2
            event.record()

        # Query should return boolean
        query_result = event.query()
        assert isinstance(query_result, bool)

        # After synchronization, should be complete
        stream.synchronize()
        assert event.query()

    def test_event_synchronize(self):
        """Test event synchronization."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        with torch.neuron.stream(stream):
            tensor = torch.ones(10, 10, device="neuron")
            result = tensor * 2
            event.record()

        # Event synchronize should wait for completion
        event.synchronize()

        # After event sync, event should be complete
        assert event.query()

        # Result should be available
        expected = torch.ones(10, 10) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_event_wait_on_stream(self):
        """Test making stream wait for event."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()
        event = torch.neuron.Event()

        # Record event on stream1
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(12, 12, device="neuron")
            result1 = tensor1 * 2
            event.record()

        # Make stream2 wait for the event
        with torch.neuron.stream(stream2):
            event.wait(stream2)
            tensor2 = torch.ones(12, 12, device="neuron")
            result2 = tensor2 * 3

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()

        # Both results should be correct
        expected1 = torch.ones(12, 12) * 2
        expected2 = torch.ones(12, 12) * 3
        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)

    def test_event_wait_current_stream(self):
        """Test event wait on current stream."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()

        with torch.neuron.stream(stream):
            tensor1 = torch.ones(8, 8, device="neuron")
            result1 = tensor1 * 2
            event.record()

            # Wait for event on current stream (should be no-op)
            event.wait()

            tensor2 = torch.ones(8, 8, device="neuron")
            result2 = tensor2 * 3

        stream.synchronize()

        # Both results should be correct
        expected1 = torch.ones(8, 8) * 2
        expected2 = torch.ones(8, 8) * 3
        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)

    def test_event_reuse(self):
        """Test reusing same event multiple times."""
        stream = torch.neuron.Stream()
        event = torch.neuron.Event()
        results = []

        # Use event multiple times
        for i in range(3):
            with torch.neuron.stream(stream):
                tensor = torch.ones(6, 6, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

            # Wait for this iteration to complete
            event.synchronize()

        # All results should be correct
        for i, result in enumerate(results):
            expected = torch.ones(6, 6) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_multi_stream_safety(self):
        """Test event operations across multiple streams."""
        streams = [torch.neuron.Stream() for _ in range(3)]
        events = [torch.neuron.Event() for _ in range(3)]
        results = {}

        # Launch operations on different streams with different events
        for i, (stream, event) in enumerate(zip(streams, events, strict=False), 1):
            with torch.neuron.stream(stream):
                tensor = torch.ones(5, 5, device="neuron")
                result = tensor * i
                results[i] = result
                event.record()

        # Synchronize all events
        for event in events:
            event.synchronize()

        # Verify all results
        for i in range(1, 4):
            expected = torch.ones(5, 5) * i
            torch.testing.assert_close(results[i].cpu(), expected)

        assert len(results) == 3

    def test_multiple_events_same_stream_with_timing(self):
        """Test recording multiple events with timing validation."""
        stream = torch.neuron.Stream()
        events = [torch.neuron.Event(enable_timing=True) for _ in range(3)]
        results = []

        with torch.neuron.stream(stream):
            for i, event in enumerate(events):
                tensor = torch.ones(6, 6, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

        stream.synchronize()

        # Validate events are completed
        for i, event in enumerate(events):
            assert event.query(), f"Event {i} should be completed"

        # Validate timing measurements between events
        for i in range(len(events) - 1):
            elapsed = events[i].elapsed_time(events[i + 1])
            assert (
                elapsed >= 0.0
            ), f"Elapsed time between events {i} and {i + 1} should be non-negative"

        # Validate computation results
        for i, result in enumerate(results):
            expected = torch.ones(6, 6) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_cross_stream_synchronization(self):
        """Test event-based cross-stream synchronization."""
        stream1 = torch.neuron.Stream()
        stream2 = torch.neuron.Stream()
        stream3 = torch.neuron.Stream()
        event1 = torch.neuron.Event()
        event2 = torch.neuron.Event()

        # Chain of dependencies: stream1 -> stream2 -> stream3
        with torch.neuron.stream(stream1):
            tensor1 = torch.ones(10, 10, device="neuron")
            result1 = tensor1 * 2
            event1.record()

        with torch.neuron.stream(stream2):
            event1.wait(stream2)  # Wait for stream1
            tensor2 = torch.ones(10, 10, device="neuron")
            result2 = tensor2 * 3
            event2.record()

        with torch.neuron.stream(stream3):
            event2.wait(stream3)  # Wait for stream2
            tensor3 = torch.ones(10, 10, device="neuron")
            result3 = tensor3 * 4

        # Synchronize final stream
        stream3.synchronize()

        # All results should be correct
        expected1 = torch.ones(10, 10) * 2
        expected2 = torch.ones(10, 10) * 3
        expected3 = torch.ones(10, 10) * 4

        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)
        torch.testing.assert_close(result3.cpu(), expected3)

    def test_event_with_default_stream(self):
        """Test event operations with default stream."""
        default_stream = torch_neuronx.default_stream()
        event = torch.neuron.Event()

        with torch.neuron.stream(default_stream):
            tensor = torch.ones(7, 7, device="neuron")
            result = tensor * 5
            event.record()

        event.synchronize()

        expected = torch.ones(7, 7) * 5
        torch.testing.assert_close(result.cpu(), expected)

    def test_multiple_events_synchronization_order(self):
        """Test synchronization order with multiple events."""
        stream = torch.neuron.Stream()
        events = [torch.neuron.Event() for _ in range(3)]
        results = []

        with torch.neuron.stream(stream):
            for i, event in enumerate(events):
                tensor = torch.ones(4, 4, device="neuron")
                result = tensor * (i + 1)
                results.append(result)
                event.record()

        # Synchronize events in reverse order
        for event in reversed(events):
            event.synchronize()

        # All results should be correct regardless of sync order
        for i, result in enumerate(results):
            expected = torch.ones(4, 4) * (i + 1)
            torch.testing.assert_close(result.cpu(), expected)

    def test_event_error_handling(self):
        """Test event behavior with error conditions."""
        event = torch.neuron.Event()

        # Synchronizing unrecorded event should not crash
        event.synchronize()

        # Querying unrecorded event should return boolean
        result = event.query()
        assert isinstance(result, bool)
