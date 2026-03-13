"""
Integration tests for Neuron memory statistics API.

Tests the memory stats functions exposed through torch_neuronx.
"""

import pytest
import torch

import torch_neuronx


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
        # Currently always 0 since caching is not implemented
        assert reserved >= 0

    def test_max_memory_reserved_returns_int(self):
        """Test that max_memory_reserved() returns an integer."""
        max_reserved = torch_neuronx.max_memory_reserved()
        assert isinstance(max_reserved, int)
        # Currently always 0 since caching is not implemented
        assert max_reserved >= 0

    def test_reserved_bytes_currently_zero(self):
        """Test that reserved_bytes is always 0 (no caching implemented)."""
        # Allocate and deallocate some tensors
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats()

        # reserved_bytes should be 0 since we don't have caching
        assert stats["reserved_bytes"]["current"] == 0
        assert stats["reserved_bytes"]["peak"] == 0
        assert torch_neuronx.memory_reserved() == 0
        assert torch_neuronx.max_memory_reserved() == 0

    def test_max_memory_allocated_tracks_peak(self):
        """Test that max_memory_allocated() tracks peak allocation."""
        # Allocate a tensor
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()

        peak_with_tensor = torch_neuronx.max_memory_allocated()

        # Delete tensor
        del x
        torch_neuronx.synchronize()

        current_after_del = torch_neuronx.memory_allocated()
        peak_after_del = torch_neuronx.max_memory_allocated()

        # Peak should remain at the high water mark
        assert peak_after_del >= peak_with_tensor
        # Current should be less than peak after deletion
        assert current_after_del <= peak_after_del

    def test_reset_peak_memory_stats_resets_peaks(self):
        """Test that reset_peak_memory_stats() resets peak to current."""
        # Allocate to create a peak
        x = torch.randn(1024, device="neuron")
        torch_neuronx.synchronize()
        del x
        torch_neuronx.synchronize()

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
