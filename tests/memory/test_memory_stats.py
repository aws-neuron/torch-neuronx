"""
Integration tests for Neuron memory statistics API.

Tests the memory stats functions exposed through torch_neuronx.
"""

import os

import pytest
import torch

import torch_neuronx

_LAZY_ALLOC_ENABLED = os.environ.get("TORCH_NEURONX_ENABLE_LAZY_ALLOC") == "1"
_requires_lazy_alloc = pytest.mark.skipif(
    not _LAZY_ALLOC_ENABLED, reason="Requires TORCH_NEURONX_ENABLE_LAZY_ALLOC=1"
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMemoryStats:
    """Test cases for memory statistics API."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset memory state before and after each test."""
        torch_neuronx.empty_cache()
        torch_neuronx.reset_peak_memory_stats()
        torch_neuronx.reset_accumulated_memory_stats()
        yield
        torch_neuronx.empty_cache()

    def test_memory_stats_returns_dict_with_expected_keys(self):
        """Test that memory_stats() returns a dict with all expected keys."""
        stats = torch_neuronx.memory_stats(device=0)

        assert isinstance(stats, dict)
        assert "allocated_bytes" in stats
        assert "reserved_bytes" in stats
        assert "active_bytes" in stats
        assert "num_alloc_retries" in stats
        assert "num_ooms" in stats
        assert "num_tensor_frees" in stats
        assert "allocation_requests" in stats

    def test_memory_stats_byte_fields_have_current_and_peak(self):
        """Test that byte stat fields have current and peak sub-keys."""
        stats = torch_neuronx.memory_stats(device=0)

        for field in ["allocated_bytes", "reserved_bytes", "active_bytes"]:
            assert "current" in stats[field], f"{field} missing 'current'"
            assert "peak" in stats[field], f"{field} missing 'peak'"
            assert isinstance(stats[field]["current"], int)
            assert isinstance(stats[field]["peak"], int)

    def test_memory_allocated_returns_int(self):
        """Test that memory_allocated() returns an integer."""
        allocated = torch_neuronx.memory_allocated()
        assert isinstance(allocated, int)
        assert allocated >= 0

    def test_memory_reserved_returns_int(self):
        """Test that memory_reserved() returns an integer."""
        reserved = torch_neuronx.memory_reserved()
        assert isinstance(reserved, int)
        assert reserved >= 0

    def test_max_memory_reserved_returns_int(self):
        """Test that max_memory_reserved() returns an integer."""
        max_reserved = torch_neuronx.max_memory_reserved()
        assert isinstance(max_reserved, int)
        assert max_reserved >= 0

    @_requires_lazy_alloc
    def test_lazy_alloc_memory_stats_after_dealloc(self):
        """After alloc + dealloc + sync, the block is freed back to the segment pool.

        allocated/active bytes drop to 0, but reserved bytes reflect the segment
        that still holds cached free blocks.
        """
        # 1024 float32 = 4096 bytes
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats()
        assert stats["allocated_bytes"]["current"] == 0
        assert stats["active_bytes"]["current"] == 0
        # Segment is still reserved (not freed until empty_cache)
        assert stats["reserved_bytes"]["current"] > 0
        assert torch_neuronx.memory_allocated() == 0
        assert torch_neuronx.memory_reserved() > 0

    @_requires_lazy_alloc
    def test_empty_cache_frees_reserved_bytes(self):
        """empty_cache() releases unused segments, freeing reserved bytes to NRT."""
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

        # Block freed to pool but segment still reserved
        assert torch_neuronx.memory_reserved() > 0

        # Drain cache
        torch_neuronx.empty_cache()

        stats = torch_neuronx.memory_stats()
        assert stats["reserved_bytes"]["current"] == 0
        assert stats["allocated_bytes"]["current"] == 0
        assert stats["active_bytes"]["current"] == 0
        assert torch_neuronx.memory_reserved() == 0

    @_requires_lazy_alloc
    def test_allocated_exceeds_active_for_rounded_up_size(self):
        """A request that does not hit a block boundary should show
        allocated_bytes > active_bytes.
        """
        initial_allocated = torch_neuronx.memory_allocated()
        initial_stats = torch_neuronx.memory_stats()
        initial_active = initial_stats["active_bytes"]["current"]

        # 3 float32 elements = 12 bytes requested; rounds up to min block size.
        x = torch.empty(3, dtype=torch.float32, device="neuron")
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats()
        active = stats["active_bytes"]["current"] - initial_active
        allocated = torch_neuronx.memory_allocated() - initial_allocated

        # Requested (active) should match the tensor's byte size
        assert active == 12
        # At the minimum rounded size, allocated block is strictly larger.
        assert allocated > active

        del x

    def test_max_memory_allocated_tracks_peak(self):
        """Test that max_memory_allocated() tracks peak allocation."""
        # Allocate a tensor
        initial = torch_neuronx.memory_allocated()

        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()

        peak_with_tensor = torch_neuronx.max_memory_allocated()
        allocated_with_tensor = torch_neuronx.memory_allocated()
        assert allocated_with_tensor > initial

        # Delete tensor
        del x
        torch_neuronx.synchronize()

        current_after_del = torch_neuronx.memory_allocated()
        peak_after_del = torch_neuronx.max_memory_allocated()

        # Peak should remain at the high water mark
        assert peak_after_del >= peak_with_tensor
        # Current must drop after deletion (memory was actually freed)
        assert current_after_del < allocated_with_tensor

    def test_reset_peak_memory_stats_resets_peaks(self):
        """Test that reset_peak_memory_stats() resets peak to current."""
        # Allocate to create a peak
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

        # Drain cache so reserved_bytes.current is 0 before reset
        torch_neuronx.empty_cache()

        # Reset peaks
        torch_neuronx.reset_peak_memory_stats()

        # Get stats after reset
        stats_after = torch_neuronx.memory_stats()

        # After reset, peak should equal current
        assert stats_after["allocated_bytes"]["peak"] == stats_after["allocated_bytes"]["current"]
        assert stats_after["reserved_bytes"]["peak"] == stats_after["reserved_bytes"]["current"]
        assert stats_after["active_bytes"]["peak"] == stats_after["active_bytes"]["current"]

    def test_memory_summary_returns_formatted_string(self):
        """Test that memory_summary() returns a formatted string."""
        summary = torch_neuronx.memory_summary()

        assert isinstance(summary, str)
        assert "Neuron Memory Summary" in summary
        assert "Device" in summary

    def test_memory_summary_abbreviated(self):
        """Test that memory_summary(abbreviated=True) returns shorter output."""
        full_summary = torch_neuronx.memory_summary(abbreviated=False)
        abbrev_summary = torch_neuronx.memory_summary(abbreviated=True)

        assert isinstance(abbrev_summary, str)
        # Abbreviated should generally be shorter or equal
        assert len(abbrev_summary) <= len(full_summary)

    def test_allocation_increases_allocated_bytes(self):
        """Test that allocating a tensor increases allocated memory stats."""
        initial_active = torch_neuronx.memory_allocated()
        initial_requests = torch_neuronx.memory_stats()["allocation_requests"]

        # Allocate a tensor: 1024 floats = 4096 bytes
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        expected_size = 1024 * 4  # float32 = 4 bytes

        new_active = torch_neuronx.memory_allocated()
        new_requests = torch_neuronx.memory_stats()["allocation_requests"]

        # Allocation requests should increase
        assert new_requests > initial_requests

        # Active bytes should increase by at least the tensor size
        assert new_active >= initial_active + expected_size

        # Clean up
        del x

    def test_deallocation_increases_tensor_frees(self):
        """Test that deleting a tensor increments num_tensor_frees."""
        # Allocate a tensor
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()

        frees_before = torch_neuronx.memory_stats()["num_tensor_frees"]

        # Delete the tensor
        del x
        torch_neuronx.synchronize()

        frees_after = torch_neuronx.memory_stats()["num_tensor_frees"]

        # num_tensor_frees should increase
        assert frees_after > frees_before

    def test_device_parameter_accepts_int(self):
        """Test that device parameter accepts integer."""
        stats = torch_neuronx.memory_stats(device=0)
        assert isinstance(stats, dict)

    def test_device_parameter_accepts_none(self):
        """Test that device parameter accepts None (uses current device)."""
        stats = torch_neuronx.memory_stats(device=None)
        assert isinstance(stats, dict)

    def test_reset_accumulated_memory_stats(self):
        """Test that reset_accumulated_memory_stats() resets counters."""
        # Trigger some allocations first
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

        # Reset accumulated stats
        torch_neuronx.reset_accumulated_memory_stats()

        stats = torch_neuronx.memory_stats()

        # All counters should be reset
        assert stats["num_alloc_retries"] == 0
        assert stats["num_ooms"] == 0
        assert stats["num_tensor_frees"] == 0
        assert stats["allocation_requests"] == 0

    def test_multiple_allocations_accumulate(self):
        """Test that multiple allocations accumulate in active memory."""
        initial_active = torch_neuronx.memory_allocated()

        tensors = []
        for _ in range(5):
            tensors.append(torch.randn(256, device="neuron"))
        torch_neuronx.synchronize()

        final_active = torch_neuronx.memory_allocated()

        # Should have allocated at least 5 * 256 * 4 = 5120 bytes
        expected_min = 5 * 256 * 4
        assert final_active >= initial_active + expected_min

        # Clean up
        del tensors

    def test_sequential_ops_free_intermediate_tensors(self):
        """Test that intermediate tensors from sequential ops are freed."""
        baseline = torch_neuronx.memory_allocated()

        # Run two ops that reuse the same cached kernel (aten::add)
        a = torch.randn(1024, device="neuron")
        b = a + a
        del a
        torch_neuronx.synchronize()

        # Only b should be alive
        mem_one = torch_neuronx.memory_allocated()

        c = b + b
        del b
        torch_neuronx.synchronize()

        # Only c should be alive — b's memory must be freed
        mem_two = torch_neuronx.memory_allocated()
        assert mem_two <= mem_one

        del c
        torch_neuronx.synchronize()

        mem_final = torch_neuronx.memory_allocated()
        assert mem_final <= baseline

    def test_peak_memory_stats_valid_after_allocation(self):
        """Test that peak stats is captured after an allocation."""
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        peak = torch_neuronx.memory.get_peak_memory_stats(0)
        assert peak["valid"], "Peak stats should be valid after allocation"
        assert peak["allocated_bytes"]["current"] > 0
        assert peak["allocated_bytes"]["peak"] > 0
        assert peak["active_bytes"]["current"] > 0
        assert peak["allocated_bytes"]["current"] >= peak["active_bytes"]["current"]
        del x
        torch_neuronx.synchronize()

    def test_peak_memory_stats_not_cleared_by_reset_peak(self):
        """Test that reset_peak_memory_stats does NOT clear peak stats."""
        x = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        stats_before = torch_neuronx.memory.get_peak_memory_stats(0)
        assert stats_before["valid"]

        torch_neuronx.reset_peak_memory_stats()

        stats_after = torch_neuronx.memory.get_peak_memory_stats(0)
        assert stats_after["valid"], "Peak stats should survive reset_peak_memory_stats"
        assert stats_after["allocated_bytes"]["peak"] == stats_before["allocated_bytes"]["peak"]
        del x
        torch_neuronx.synchronize()

    def test_peak_memory_stats_tracks_increasing_peak(self):
        """Test that peak stats updates when a new peak is reached."""
        x = torch.randn(512, device="neuron")
        torch_neuronx.synchronize()
        t1 = torch_neuronx.memory.get_peak_memory_stats(0)

        y = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        t2 = torch_neuronx.memory.get_peak_memory_stats(0)

        assert t2["allocated_bytes"]["peak"] >= t1["allocated_bytes"]["peak"]
        del x, y
        torch_neuronx.synchronize()

    def test_peak_memory_stats_internal_fragmentation(self):
        """Test that internal fragmentation is non-negative."""
        x = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        peak = torch_neuronx.memory.get_peak_memory_stats(0)
        assert peak["internal_fragmentation"] >= 0
        del x
        torch_neuronx.synchronize()

    def test_segment_pool_analysis_returns_valid_stats(self):
        """Test that segment_pool_analysis returns expected keys with valid values."""
        x = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        analysis = torch_neuronx.memory.segment_pool_analysis(0)

        assert analysis["allocated_bytes"] > 0
        assert analysis["reserved_bytes"] >= analysis["allocated_bytes"]
        assert analysis["active_bytes"] > 0
        assert analysis["total_hbm"] > 0
        assert analysis["free_chunks"] >= 0
        assert analysis["largest_free"] >= 0
        assert analysis["num_segments"] >= 1
        assert 0 <= analysis["fragmentation_index"] <= 1
        assert analysis["memory_efficiency"] >= 0
        assert analysis["internal_fragmentation"] >= 0
        assert analysis["cached_free"] >= 0
        del x
        torch_neuronx.synchronize()

    def test_segment_pool_analysis_summary_returns_string(self):
        """Test that segment_pool_analysis_summary returns a formatted string."""
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        summary = torch_neuronx.memory.segment_pool_analysis_summary(0)
        assert isinstance(summary, str)
        assert "Segment Pool Analysis" in summary
        assert "Frag index" in summary
        del x
        torch_neuronx.synchronize()

    def test_peak_memory_stats_invalid_before_allocation(self):
        """Test that peak stats returns a dict with expected keys even if no new peak."""
        # Note: valid may be True from prior tests since peak stats is
        # intentionally not cleared by reset_peak_memory_stats.
        peak = torch_neuronx.memory.get_peak_memory_stats(0)
        assert "valid" in peak
        assert "allocated_bytes" in peak
        assert "reserved_bytes" in peak
        assert "active_bytes" in peak
        assert "internal_fragmentation" in peak

    def test_segment_pool_analysis_empty_state(self):
        """Test that segment_pool_analysis handles no-allocation state without crashing."""
        analysis = torch_neuronx.memory.segment_pool_analysis(0)
        assert analysis["allocated_bytes"] == 0
        assert analysis["active_bytes"] == 0
        assert analysis["free_chunks"] >= 0
        assert analysis["fragmentation_index"] == 0.0
        assert analysis["memory_efficiency"] == 0.0
        assert analysis["num_segments"] == 0

    def test_segment_pool_analysis_summary_empty_state(self):
        """Test that segment_pool_analysis_summary handles no-allocation state."""
        summary = torch_neuronx.memory.segment_pool_analysis_summary(0)
        assert isinstance(summary, str)
        assert "Segment Pool Analysis" in summary

    def test_peak_stats_alloc_dealloc_stress(self):
        """Test peak stats across repeated alloc/matmul/dealloc cycles."""
        peak_before = torch_neuronx.memory.get_peak_memory_stats(0)

        for _ in range(50):
            x = torch.randn(256, 256, device="neuron")
            torch_neuronx.synchronize()
            for _ in range(20):
                y = x @ x.T
            torch_neuronx.synchronize()
            del x, y
            torch_neuronx.synchronize()

        peak_after = torch_neuronx.memory.get_peak_memory_stats(0)
        assert peak_after["valid"], "Peak stats should be valid after stress test"
        assert peak_after["allocated_bytes"]["peak"] > 0

        # Peak should be at least as large as before (monotonically increasing)
        if peak_before["valid"]:
            assert peak_after["allocated_bytes"]["peak"] >= peak_before["allocated_bytes"]["peak"]

        # Verify analysis still works after stress
        torch_neuronx.synchronize()
        analysis = torch_neuronx.memory.segment_pool_analysis(0)
        assert analysis["num_segments"] >= 1
        assert analysis["reserved_bytes"] >= analysis["allocated_bytes"]
