"""Test cross-process caching using distributed multi-process framework.

These tests verify that persistent cache entries created by one process
can be correctly read by another process, proving cache keys are
deterministic and don't include process-specific data (like PID).
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from tests.distributed.utils import DistributedTester


def _cross_process_cache_test(rank, world_size, func_args):
    """Test function run by each rank.

    Rank 0: Creates cache entry by running operation
    Rank 1: Should hit persistent cache (proves cross-process sharing works)
    """

    import torch_neuronx

    # Clear compilation cache for this process
    torch_neuronx._C._clear_compilation_cache()

    # Use barrier to ensure rank 0 completes first
    if rank == 1:
        dist.barrier()  # Rank 1 waits for rank 0 to finish

    # Run the same operation on all ranks
    base_size = 1000
    num_tensors = 16
    sizes = [base_size + j * 10 for j in range(num_tensors)]
    tensors = [torch.randn(s, device="neuron") for s in sizes]
    torch.cat(tensors, dim=0)
    torch.neuron.synchronize()

    if rank == 0:
        dist.barrier()  # Signal rank 1 that rank 0 is done

    stats = torch_neuronx._C._get_compilation_cache_stats()

    if rank == 0:
        # Rank 0 should compile (cache miss)
        assert stats["persistent_misses"] >= 1, (
            f"Rank 0 expected cache misses (compilation), "
            f"got {stats['persistent_misses']} misses"
        )
    elif rank == 1:
        # Rank 1 should hit cache from rank 0
        assert stats["persistent_hits"] >= 1, (
            f"Cross-process caching failed. Rank 1 expected persistent hits "
            f"but got {stats['persistent_hits']} hits, "
            f"{stats['persistent_misses']} misses."
        )


def _cross_version_cache_test(rank, world_size, func_args):
    """Test function run by each rank.

    Rank 0 populates the cache. Rank 1 then runs with a different
    global_version_hash(), which simulates an edit to a lowering source file, and must
    still hit the persistent cache.

    The persistent NEFF key is content-addressed on the lowered IR, so a source or
    version change must not orphan the cache. The legacy key folded
    global_version_hash() into base_key and therefore failed this test.
    """

    from unittest.mock import patch

    import torch_neuronx

    torch_neuronx._C._clear_compilation_cache()

    # Patch every module that binds global_version_hash at import time. A plain patch of
    # torch_neuronx.utils.global_version_hash does NOT reach these: `from ... import x`
    # copies the reference, so the module keeps the original function.
    edited = [
        patch(f"{m}.global_version_hash", return_value="0" * 64)
        for m in (
            "torch_neuronx.utils",
            "torch_neuronx.python_ops.torch_mlir.kernel",
            "torch_neuronx.python_ops.torch_mlir.nki_op_impl",
        )
    ]

    if rank == 1:
        dist.barrier()
        for p in edited:
            p.start()

    try:
        a = torch.randn(64, 128, device="neuron")
        b = torch.randn(128, 32, device="neuron")
        torch.matmul(a, b)
        torch.neuron.synchronize()
    finally:
        if rank == 1:
            for p in edited:
                p.stop()

    if rank == 0:
        dist.barrier()

    stats = torch_neuronx._C._get_compilation_cache_stats()

    if rank == 0:
        assert stats["persistent_misses"] >= 1, (
            f"Rank 0 expected cache misses (compilation), "
            f"got {stats['persistent_misses']} misses"
        )
    elif rank == 1:
        # Strict: the version change must cost NOTHING. Any miss means an entry was
        # orphaned, which is the bug. ">= 1 hit" would pass even with most keys orphaned.
        assert stats["persistent_hits"] >= 1 and stats["persistent_misses"] == 0, (
            f"A version change orphaned the persistent cache. Rank 1 expected only "
            f"persistent hits but got {stats['persistent_hits']} hits and "
            f"{stats['persistent_misses']} misses. The persistent NEFF key must not "
            f"depend on global_version_hash()."
        )


@pytest.mark.skipif(
    os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
    reason="Async pipeline persistent caching tests",
)
class TestCrossProcessCaching:
    """Tests for cross-process cache sharing using distributed framework."""

    def test_cross_process_cache_sharing(self):
        """Test that cache entries created by one process are usable by another.

        Uses mp.spawn to create separate processes with different neuron cores.
        Rank 0 compiles first, Rank 1 should hit persistent cache.

        Parameterized by mlir_mode fixture to run with both XLA and MLIR paths.
        """

        with tempfile.TemporaryDirectory() as cache_dir:
            # Set env vars in parent - inherited by spawned children
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = cache_dir
            os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"

            tester = DistributedTester(world_size=2)
            tester.run_test(_cross_process_cache_test, cache_dir=cache_dir)

    def test_cache_survives_a_version_change(self):
        """A source or version change must not orphan the persistent NEFF cache.

        Rank 0 compiles. Rank 1 runs with a different global_version_hash() and must
        still hit the cache, because the persistent key comes from the lowered IR.
        """

        with tempfile.TemporaryDirectory() as cache_dir:
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = cache_dir
            os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"

            tester = DistributedTester(world_size=2)
            tester.run_test(_cross_version_cache_test, cache_dir=cache_dir)
