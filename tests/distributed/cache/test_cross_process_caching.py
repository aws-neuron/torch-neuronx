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
