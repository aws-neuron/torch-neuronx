"""
Integration tests for Neuron memory statistics API.

Tests the memory stats functions exposed through torch_neuronx.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.defer_retry_helpers import (
    big_mib,
    floats_for_mib,
    n_chunks,
    oversize_mib,
    stage_pending_kdeallocs,
    staged_mib,
    vnc_hbm_mib,
)

_skip_nrt_oom_report_regression = pytest.mark.skip(
    reason=(
        "NRT-6217: NRT 2.x.77388.0 aborts in nrt_debug_log_oom_report when the "
        "process-local device index does not identify the owned host-global core; "
        "the stable 2.x.68105.0 baseline does not"
    )
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
        assert "allocated_bytes.all.current" in stats
        assert "reserved_bytes.all.current" in stats
        assert "active_bytes.all.current" in stats
        assert "num_alloc_retries" in stats
        assert "num_ooms" in stats
        assert "num_tensor_frees" in stats
        assert "allocation_requests" in stats
        assert "num_alloc_deferred" in stats
        assert "num_alloc_drain_retry_ok" in stats
        assert "num_alloc_drain_retry_fail" in stats

    def test_memory_stats_byte_fields_have_current_and_peak(self):
        """Test that byte stat fields have flat keys for all pools."""
        stats = torch_neuronx.memory_stats(device=0)

        for field in ["allocated_bytes", "reserved_bytes", "active_bytes"]:
            for pool in ["all", "large_pool", "small_pool"]:
                assert f"{field}.{pool}.current" in stats, f"{field}.{pool}.current missing"
                assert f"{field}.{pool}.peak" in stats, f"{field}.{pool}.peak missing"
                assert isinstance(stats[f"{field}.{pool}.current"], int)
                assert isinstance(stats[f"{field}.{pool}.peak"], int)

    def test_memory_stats_as_nested_dict(self):
        """Test that memory_stats_as_nested_dict() returns nested structure."""
        nested = torch_neuronx.memory_stats_as_nested_dict(device=0)

        assert isinstance(nested, dict)
        for field in ["allocated_bytes", "reserved_bytes", "active_bytes"]:
            assert field in nested
            assert isinstance(nested[field], dict)
            for pool in ["all", "large_pool", "small_pool"]:
                assert pool in nested[field], f"{field} missing pool '{pool}'"
                assert "current" in nested[field][pool]
                assert "peak" in nested[field][pool]
                assert isinstance(nested[field][pool]["current"], int)
        assert "num_alloc_retries" in nested
        assert "num_ooms" in nested

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
        assert stats["allocated_bytes.all.current"] == 0
        assert stats["active_bytes.all.current"] == 0
        # Segment is still reserved (not freed until empty_cache)
        assert stats["reserved_bytes.all.current"] > 0
        assert torch_neuronx.memory_allocated() == 0
        assert torch_neuronx.memory_reserved() > 0

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
        assert stats["reserved_bytes.all.current"] == 0
        assert stats["allocated_bytes.all.current"] == 0
        assert stats["active_bytes.all.current"] == 0
        assert torch_neuronx.memory_reserved() == 0

    def test_allocated_exceeds_active_for_rounded_up_size(self):
        """A request that does not hit a block boundary should show
        allocated_bytes > active_bytes.
        """
        initial_allocated = torch_neuronx.memory_allocated()
        initial_stats = torch_neuronx.memory_stats()
        initial_active = initial_stats["active_bytes.all.current"]

        # 3 float32 elements = 12 bytes requested; rounds up to min block size.
        x = torch.empty(3, dtype=torch.float32, device="neuron")
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats()
        active = stats["active_bytes.all.current"] - initial_active
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
        assert stats_after["allocated_bytes.all.peak"] == stats_after["allocated_bytes.all.current"]
        assert stats_after["reserved_bytes.all.peak"] == stats_after["reserved_bytes.all.current"]
        assert stats_after["active_bytes.all.peak"] == stats_after["active_bytes.all.current"]

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
        peak = torch_neuronx.memory.peak_memory_stats(0)
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
        stats_before = torch_neuronx.memory.peak_memory_stats(0)
        assert stats_before["valid"]

        torch_neuronx.reset_peak_memory_stats()

        stats_after = torch_neuronx.memory.peak_memory_stats(0)
        assert stats_after["valid"], "Peak stats should survive reset_peak_memory_stats"
        assert stats_after["allocated_bytes"]["peak"] == stats_before["allocated_bytes"]["peak"]
        del x
        torch_neuronx.synchronize()

    def test_peak_memory_stats_tracks_increasing_peak(self):
        """Test that peak stats updates when a new peak is reached."""
        x = torch.randn(512, device="neuron")
        torch_neuronx.synchronize()
        t1 = torch_neuronx.memory.peak_memory_stats(0)

        y = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        t2 = torch_neuronx.memory.peak_memory_stats(0)

        assert t2["allocated_bytes"]["peak"] >= t1["allocated_bytes"]["peak"]
        del x, y
        torch_neuronx.synchronize()

    def test_peak_memory_stats_internal_fragmentation(self):
        """Test that internal fragmentation is non-negative."""
        x = torch.randn(1024, 1024, device="neuron")
        torch_neuronx.synchronize()
        peak = torch_neuronx.memory.peak_memory_stats(0)
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
        assert analysis["memory_density"] >= 0
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
        peak = torch_neuronx.memory.peak_memory_stats(0)
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
        assert analysis["memory_density"] == 0.0
        assert analysis["num_segments"] == 0

    def test_segment_pool_analysis_summary_empty_state(self):
        """Test that segment_pool_analysis_summary handles no-allocation state."""
        summary = torch_neuronx.memory.segment_pool_analysis_summary(0)
        assert isinstance(summary, str)
        assert "Segment Pool Analysis" in summary

    def test_peak_stats_alloc_dealloc_stress(self):
        """Test peak stats across repeated alloc/matmul/dealloc cycles."""
        peak_before = torch_neuronx.memory.peak_memory_stats(0)

        for _ in range(50):
            x = torch.randn(256, 256, device="neuron")
            torch_neuronx.synchronize()
            for _ in range(20):
                y = x @ x.T
            torch_neuronx.synchronize()
            del x, y
            torch_neuronx.synchronize()

        peak_after = torch_neuronx.memory.peak_memory_stats(0)
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


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.usefixtures("clean_device_state")
@pytest.mark.xdist_group("oom_serial")
@pytest.mark.needs_process_isolation(
    reason="Five of these drive an oversize_mib() alloc past total HBM to force a hard "
    "OOM, and the OOM-report path can hit a KaenaRuntime assertion (tdrv/db.c:235) that "
    "SIGABRTs the process. xdist_group serializes them but does not contain an abort: it "
    "kills the whole worker, and the session wedges after xdist replaces it."
)
class TestLazyAllocDeferRetry:
    """End-to-end tests for the lazy-alloc defer-retry path: parks an OOM'd
    kAlloc as Deferred, waits for the stream to drain so preceding kDeallocs
    release NRT memory, then retries. Sizing + staging helpers live in
    ``tests.utils.defer_retry_helpers``."""

    def test_defer_retry_silent_recovery(self):
        """A pressure kAlloc with kDeallocs queued behind compute defers,
        recovers silently via stage-2 drain-retry or stage-3 cache-drain,
        and completes without raising. Timing-dependent: which stage wins
        is not pinned."""
        stage_pending_kdeallocs()

        big = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
        big.fill_(0.0)
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats(device=0)
        assert stats["num_alloc_deferred"] == 1, (
            f"defer must engage exactly once; got {stats['num_alloc_deferred']}. "
            f"VNC={vnc_hbm_mib()} MiB, staged={staged_mib()} MiB / {n_chunks()} "
            f"chunks, big={big_mib()} MiB. Bump the fractions if pressure missed."
        )
        assert stats["num_ooms"] == 0, f"unexpected hard OOM: stats={stats}"
        stage2_ok = stats["num_alloc_drain_retry_ok"] == 1
        stage3_ok = stats["num_alloc_drain_retry_fail"] == 1
        assert stage2_ok or stage3_ok, f"retry funnel missed: stats={stats}"
        assert stage2_ok != stage3_ok, f"stages must be mutually exclusive: stats={stats}"
        del big

    def test_defer_retry_counters_sync_baseline(self):
        """Negative control: synchronizing before the pressure alloc must
        not engage defer-retry, catching regressions that inflate counters
        on unrelated allocations."""
        stage_pending_kdeallocs()
        torch_neuronx.synchronize()  # full drain before the pressure alloc

        big = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
        big.fill_(0.0)
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats(device=0)
        assert stats["num_alloc_deferred"] == 0, f"unexpected defer: stats={stats}"
        assert stats["num_alloc_drain_retry_ok"] == 0, f"unexpected drain-retry: stats={stats}"
        assert (
            stats["num_alloc_drain_retry_fail"] == 0
        ), f"unexpected drain-retry-fail: stats={stats}"
        assert stats["num_ooms"] == 0, f"unexpected OOM: stats={stats}"
        del big

    @pytest.mark.xfail(
        reason="Flaky signal-6 crash from a KaenaRuntime assertion (tdrv/db.c:235) in the "
        "OOM-report path under parallel CI; passes serially. Xfailing to unblock CI."
    )
    def test_alone_on_stream_oom_propagates_cleanly(self):
        """An oversize kAlloc on an empty stream surfaces a RuntimeError
        and records the terminal-failure counters."""
        torch_neuronx.synchronize()

        floats = floats_for_mib(oversize_mib())

        with pytest.raises(RuntimeError) as excinfo:
            big = torch.empty(floats, device="neuron", dtype=torch.float32)
            big.fill_(0.0)
            torch_neuronx.synchronize()

        message = str(excinfo.value)
        assert "NRT_RESOURCE" in message or "OOM" in message, f"not OOM-flavored: {message!r}"

        stats = torch_neuronx.memory_stats(device=0)
        assert stats["num_alloc_retries"] == 1, f"stats={stats}"
        assert stats["num_ooms"] == 1, f"stats={stats}"

    @_skip_nrt_oom_report_regression
    def test_hard_oom_records_counters(self):
        """Bottom of the funnel: defer → RetryReady → AttemptOOMRecovery →
        cache-drain → final failure on a request that exceeds total HBM."""
        stage_pending_kdeallocs()

        floats = floats_for_mib(oversize_mib())  # > total HBM

        with pytest.raises(RuntimeError):
            big = torch.empty(floats, device="neuron", dtype=torch.float32)
            big.fill_(0.0)
            torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats(device=0)
        assert stats["num_alloc_retries"] >= 1, f"stats={stats}"
        assert stats["num_ooms"] == 1, f"stats={stats}"

    def test_repeated_defer_retry_cycles(self):
        """Per-OperationContext AllocState must not leak across allocations:
        N pressure cycles must each defer once and recover silently."""
        n_iters = 3

        for _ in range(n_iters):
            stage_pending_kdeallocs()
            big = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
            big.fill_(0.0)
            del big

        torch_neuronx.synchronize()
        stats = torch_neuronx.memory_stats(device=0)

        assert (
            stats["num_alloc_deferred"] == n_iters
        ), f"expected {n_iters} defers, got {stats['num_alloc_deferred']}. stats={stats}"
        assert stats["num_ooms"] == 0, f"unexpected hard OOM: stats={stats}"
        stage2_total = stats["num_alloc_drain_retry_ok"] + stats["num_alloc_drain_retry_fail"]
        assert stage2_total == stats["num_alloc_deferred"], (
            f"every defer must resolve in stage 2; got ok="
            f"{stats['num_alloc_drain_retry_ok']}, fail="
            f"{stats['num_alloc_drain_retry_fail']}, deferred="
            f"{stats['num_alloc_deferred']}"
        )

    def test_synchronize_returns_when_deferred_alloc_completes(self):
        """synchronize() must return once a Deferred kAlloc completes or escalates."""
        stage_pending_kdeallocs()

        big = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
        big.fill_(0.0)
        torch_neuronx.synchronize()

        assert big.shape[0] == floats_for_mib(big_mib())
        del big

    def test_compute_behind_deferred_alloc_sees_bound_memory(self):
        """FIFO invariant: a compute op submitted behind a Deferred kAlloc
        must run against the bound buffer after retry, not before."""
        stage_pending_kdeallocs()

        floats = floats_for_mib(big_mib())
        big = torch.empty(floats, device="neuron", dtype=torch.float32)
        sentinel = 7.0
        big.fill_(sentinel)
        torch_neuronx.synchronize()

        stats = torch_neuronx.memory_stats(device=0)
        assert (
            stats["num_alloc_deferred"] == 1
        ), f"defer must engage to cover FIFO invariant; stats={stats}"
        assert stats["num_ooms"] == 0, f"unexpected hard OOM: stats={stats}"
        # Endpoints would be garbage if fill_() ran against unmapped memory.
        cpu = big.cpu()
        assert float(cpu[0].item()) == pytest.approx(
            sentinel
        ), f"FIFO violated: cpu[0]={cpu[0].item()} (expected {sentinel})"
        assert float(cpu[-1].item()) == pytest.approx(
            sentinel
        ), f"FIFO violated: cpu[-1]={cpu[-1].item()} (expected {sentinel})"
        del big

    @_skip_nrt_oom_report_regression
    def test_alone_on_stream_oom_surfaces_oom_error_first(self):
        """First-exception ordering: when an OOM'd kAlloc has a dependent
        compute behind it, the user-visible RuntimeError must describe the
        OOM, not the downstream 'Tensor not bound' from the failed bind."""
        torch_neuronx.synchronize()  # empty stream

        floats = floats_for_mib(oversize_mib())  # > HBM total

        with pytest.raises(RuntimeError) as excinfo:
            big = torch.empty(floats, device="neuron", dtype=torch.float32)
            big.fill_(1.0)  # would throw "Tensor not bound" if it ran
            torch_neuronx.synchronize()

        message = str(excinfo.value)
        assert "NRT_RESOURCE" in message or "OOM" in message, f"not OOM-flavored: {message!r}"
        assert "Tensor not bound" not in message, f"first-exception ordering regressed: {message!r}"

        stats = torch_neuronx.memory_stats(device=0)
        assert stats["num_alloc_retries"] == 1, f"stats={stats}"
        assert stats["num_ooms"] == 1, f"stats={stats}"

    @_skip_nrt_oom_report_regression
    def test_oom_report_fires_on_unrecoverable_oom(self):
        """Unrecoverable OOM should produce a detailed NRT diagnostic dump
        (log_dev_mem) for debugging, not just the basic nrt_infodump."""
        import os
        import tempfile

        # Capture stderr at fd level (NRT writes via C fprintf)
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            stderr_fd = os.dup(2)
            os.dup2(tmp.fileno(), 2)

            try:
                floats = floats_for_mib(oversize_mib())
                with pytest.raises(RuntimeError):
                    big = torch.empty(floats, device="neuron", dtype=torch.float32)
                    big.fill_(0.0)
                    torch_neuronx.synchronize()
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)

            tmp.seek(0)
            captured = tmp.read().decode("utf-8", errors="replace")

        assert "log_dev_mem" in captured, "Detailed OOM report should fire on unrecoverable OOM"

    def test_oom_report_suppressed_on_recoverable_oom(self):
        """A recoverable OOM (defer-retry succeeds) must NOT produce a detailed
        NRT dump — only true terminal OOMs should report."""
        import os
        import tempfile

        with tempfile.TemporaryFile(mode="w+b") as tmp:
            stderr_fd = os.dup(2)
            os.dup2(tmp.fileno(), 2)

            try:
                stage_pending_kdeallocs()
                big = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
                big.fill_(0.0)
                torch_neuronx.synchronize()
                del big
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)

            tmp.seek(0)
            captured = tmp.read().decode("utf-8", errors="replace")

        assert (
            "log_dev_mem" not in captured
        ), "Detailed OOM report fired on recoverable OOM — should only fire on terminal OOM"
