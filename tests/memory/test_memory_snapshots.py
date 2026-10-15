"""
Tests for Neuron memory snapshot API.

Tests the memory history recording and snapshot functionality including:
- API structure and basic functionality
- Allocation/deallocation tracking
- Recycling behavior and stack traces
- Snapshot serialization
"""

import pickle
import tempfile

import pytest
import torch

import torch_neuronx
from torch_neuronx import memory


def sync():
    """Synchronize Neuron device."""
    torch.neuron.synchronize()


@pytest.fixture(autouse=True)
def reset_memory_state():
    """Reset memory state before and after each test."""
    memory._record_memory_history(None)
    torch_neuronx.empty_cache()
    yield
    memory._record_memory_history(None)
    torch_neuronx.empty_cache()


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMemorySnapshotAPI:
    """Test cases for memory snapshot API structure and basic functionality."""

    def test_snapshot_returns_dict_with_expected_keys(self):
        """Test that _snapshot() returns a dict with segments and device_traces."""
        snapshot = memory._snapshot()

        assert isinstance(snapshot, dict)
        assert "segments" in snapshot
        assert "device_traces" in snapshot
        assert isinstance(snapshot["segments"], list)
        assert isinstance(snapshot["device_traces"], list)

    def test_record_memory_history_enables_recording(self):
        """Test that _record_memory_history(True) enables recording."""
        memory._record_memory_history(True, context="all", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        assert len(snapshot["device_traces"]) > 0

        del x

    def test_record_memory_history_disables_recording(self):
        """Test that _record_memory_history(None) disables recording."""
        memory._record_memory_history(True, context="all", max_entries=100)
        memory._record_memory_history(None)

        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        alloc_traces = sum(
            1 for traces in snapshot["device_traces"] for t in traces if t.get("action") == "alloc"
        )
        assert alloc_traces == 0

        del x

    def test_snapshot_segments_have_required_fields(self):
        """Test that snapshot segments contain required fields."""
        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        assert len(snapshot["segments"]) > 0

        for seg in snapshot["segments"]:
            assert "device" in seg
            assert "address" in seg
            assert "total_size" in seg
            assert "blocks" in seg
            assert isinstance(seg["blocks"], list)

            for block in seg["blocks"]:
                assert "size" in block
                assert "address" in block
                assert "state" in block

        del x

    def test_trace_entries_have_required_fields(self):
        """Test that trace entries contain required fields."""
        memory._record_memory_history(True, context="all", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        all_traces = [t for traces in snapshot["device_traces"] for t in traces]
        assert len(all_traces) > 0

        for entry in all_traces:
            assert "action" in entry
            assert "device" in entry
            assert "addr" in entry
            assert "size" in entry
            assert "stream" in entry
            assert "time_us" in entry

        del x

    def test_snapshot_includes_clock_domain_offset(self):
        """Test that the snapshot reports the clock domain offset."""
        snapshot = memory._snapshot()

        assert "clock_domain_offset_us" in snapshot
        assert isinstance(snapshot["clock_domain_offset_us"], int)

    def test_alloc_trace_entries_have_trace_id(self):
        """Test that ALLOC trace entries carry the runtime trace id for correlation."""
        memory._record_memory_history(True, context="alloc", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        alloc_entries = [
            t for traces in snapshot["device_traces"] for t in traces if t.get("action") == "alloc"
        ]
        assert len(alloc_entries) > 0

        # The trace id is only stamped when the runtime returns a nonzero id.
        # When present it must be a positive integer that tooling can correlate.
        for entry in alloc_entries:
            if "trace_id" in entry:
                assert isinstance(entry["trace_id"], int)
                assert entry["trace_id"] > 0

        del x

    def test_context_parameter_validation(self):
        """Test that invalid context parameter raises ValueError."""
        with pytest.raises(ValueError):
            memory._record_memory_history(True, context="invalid_context")

    def test_stacks_parameter_validation(self):
        """Test that invalid stacks parameter raises ValueError."""
        with pytest.raises(ValueError):
            memory._record_memory_history(True, stacks="invalid_stacks")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMemorySnapshotIntegration:
    """Integration tests for memory snapshot functionality."""

    def test_allocation_creates_trace_entry(self):
        """Test that allocating a tensor creates an ALLOC trace entry."""
        memory._record_memory_history(True, context="alloc", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()

        snapshot = memory._snapshot()
        alloc_entries = [
            t for traces in snapshot["device_traces"] for t in traces if t.get("action") == "alloc"
        ]
        assert len(alloc_entries) > 0, "Expected at least one ALLOC trace entry"

        del x

    def test_deallocation_creates_trace_entries(self):
        """Test that freeing a tensor creates FREE trace entries."""
        memory._record_memory_history(True, context="all", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()
        del x
        sync()

        snapshot = memory._snapshot()
        free_entries = [
            t
            for traces in snapshot["device_traces"]
            for t in traces
            if t.get("action") == "free_requested"
        ]
        assert len(free_entries) > 0, "Expected at least one FREE_REQUESTED trace entry"

    def test_dump_snapshot_creates_pickle_file(self):
        """Test that _dump_snapshot() creates a valid pickle file."""
        memory._record_memory_history(True, context="all", max_entries=100)

        x = torch.randn(1024, device="neuron")
        sync()

        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=True) as f:
            filename = f.name
            memory._dump_snapshot(filename)

            with open(filename, "rb") as pf:
                loaded = pickle.load(pf)

            assert isinstance(loaded, dict)
            assert "segments" in loaded
            assert "device_traces" in loaded

        del x

    def test_max_entries_limits_trace_buffer(self):
        """Test that max_entries parameter limits the trace buffer size."""
        max_entries = 5
        memory._record_memory_history(True, context="alloc", max_entries=max_entries)

        tensors = []
        for _ in range(max_entries + 3):
            t = torch.randn(256, device="neuron")
            sync()
            tensors.append(t)

        snapshot = memory._snapshot()

        for traces in snapshot["device_traces"]:
            assert (
                len(traces) <= max_entries + 1
            ), f"Trace buffer exceeded max_entries: {len(traces)} > {max_entries}"

        for t in tensors:
            del t


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSnapshotAllocTracking:
    """Tests for allocation/deallocation tracking via snapshots."""

    def test_basic_allocation_deallocation(self):
        """Test basic allocation and deallocation tracking."""
        memory.reset_peak_memory_stats()
        initial_stats = memory.memory_stats(0)

        size = 1024 * 1024  # 1MB
        t1 = torch.zeros(size // 4, dtype=torch.float32, device="neuron")
        sync()

        after_alloc = memory.memory_stats(0)
        assert after_alloc["allocation_requests"] > initial_stats["allocation_requests"]
        assert after_alloc["active_bytes.all.current"] > initial_stats["active_bytes.all.current"]

        del t1
        sync()

        after_free = memory.memory_stats(0)
        assert after_free["num_tensor_frees"] > initial_stats["num_tensor_frees"]

    def test_no_double_counting(self):
        """Test that deallocations are not double-counted."""
        memory.reset_peak_memory_stats()
        initial_stats = memory.memory_stats(0)

        size = 2 * 1024 * 1024  # 2MB
        num_iterations = 10

        for _ in range(num_iterations):
            t = torch.zeros(size // 4, dtype=torch.float32, device="neuron")
            sync()
            del t
            sync()

        final_stats = memory.memory_stats(0)

        total_allocs = final_stats["allocation_requests"] - initial_stats["allocation_requests"]
        total_frees = final_stats["num_tensor_frees"] - initial_stats["num_tensor_frees"]

        # Key invariant: frees should not exceed allocs (no double counting)
        assert (
            total_frees <= total_allocs + 1
        ), f"Double counting detected: more frees ({total_frees}) than allocs ({total_allocs})"
        # Also check we're not missing frees entirely
        assert (
            total_frees >= total_allocs - 2
        ), f"Missing frees: {total_allocs} allocs but only {total_frees} frees"


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestRecyclingStackTraces:
    """Tests for stack trace capture with tensor recycling."""

    def test_recycled_tensor_gets_new_stack_trace(self):
        """
        Test that recycled tensors get the NEW owner's stack trace.

        When a tensor is recycled, the allocation trace should show
        the call site of the new allocation, not the original.
        """
        memory._record_memory_history(enabled=True, context="all", stacks="python")
        memory.reset_peak_memory_stats()

        size = 1024 * 1024  # 1MB - same size to enable recycling

        def allocate_from_function_alpha():
            return torch.zeros(size // 4, dtype=torch.float32, device="neuron")

        def allocate_from_function_beta():
            return torch.zeros(size // 4, dtype=torch.float32, device="neuron")

        # First allocation
        t1 = allocate_from_function_alpha()
        sync()
        addr1 = t1.data_ptr()

        # Release first tensor
        del t1
        sync()

        # Second allocation (same size - should recycle)
        t2 = allocate_from_function_beta()
        sync()
        addr2 = t2.data_ptr()

        recycled = addr1 == addr2

        # Get snapshot and find the alpha/beta alloc traces by stack frames
        # (independent of the trace addr, which is the NRT VA and may not
        # match the Python-visible data_ptr() under lazy alloc).
        snapshot = memory._snapshot()

        alpha_alloc_traces = []
        beta_alloc_traces = []
        for traces in snapshot["device_traces"]:
            for t in traces:
                if t.get("action") != "alloc":
                    continue
                frame_names = [f.get("name", "") for f in t.get("frames", [])]
                if "allocate_from_function_alpha" in frame_names:
                    alpha_alloc_traces.append(t)
                if "allocate_from_function_beta" in frame_names:
                    beta_alloc_traces.append(t)

        del t2
        sync()

        # The two allocations must each appear in the trace with the
        # call-site stack we recorded at allocation time. This is the
        # invariant the test is really checking — that recycling does
        # not overwrite the stack on the recycled trace entry.
        assert (
            len(alpha_alloc_traces) >= 1
        ), "First alloc should have 'allocate_from_function_alpha'"
        assert len(beta_alloc_traces) >= 1, "Second alloc should have 'allocate_from_function_beta'"

        if recycled:
            # When recycled, both allocs should land on the same NRT VA in
            # the trace. Verify the trace's address (NRT VA) is identical
            # across the two events.
            alpha_trace_addrs = {t["addr"] for t in alpha_alloc_traces}
            beta_trace_addrs = {t["addr"] for t in beta_alloc_traces}
            shared = alpha_trace_addrs & beta_trace_addrs
            assert shared, (
                "Recycled alloc should reuse the same trace addr; "
                f"alpha addrs={alpha_trace_addrs}, beta addrs={beta_trace_addrs}"
            )


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSnapshotSerialization:
    """Tests for snapshot serialization and round-trip."""

    def test_snapshot_serialization_roundtrip(self):
        """Test that snapshot can be serialized and deserialized without data loss."""
        memory._record_memory_history(True, context="all", max_entries=50)

        x = torch.randn(512, device="neuron")
        y = torch.randn(256, device="neuron")
        sync()

        original_snapshot = memory._snapshot()

        # Serialize and deserialize
        serialized = pickle.dumps(original_snapshot)
        restored_snapshot = pickle.loads(serialized)

        # Verify structure is preserved
        assert len(restored_snapshot["segments"]) == len(original_snapshot["segments"])
        assert len(restored_snapshot["device_traces"]) == len(original_snapshot["device_traces"])

        # Verify segment data is preserved
        for orig_seg, rest_seg in zip(
            original_snapshot["segments"], restored_snapshot["segments"], strict=False
        ):
            assert orig_seg["device"] == rest_seg["device"]
            assert orig_seg["address"] == rest_seg["address"]
            assert orig_seg["total_size"] == rest_seg["total_size"]
            assert len(orig_seg["blocks"]) == len(rest_seg["blocks"])

        # Verify trace data is preserved
        for orig_traces, rest_traces in zip(
            original_snapshot["device_traces"], restored_snapshot["device_traces"], strict=False
        ):
            assert len(orig_traces) == len(rest_traces)
            for orig_entry, rest_entry in zip(orig_traces, rest_traces, strict=False):
                assert orig_entry["action"] == rest_entry["action"]
                assert orig_entry["addr"] == rest_entry["addr"]
                assert orig_entry["size"] == rest_entry["size"]

        del x, y
