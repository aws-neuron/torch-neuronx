"""
Test that cache keys are consistent across ranks for torch.compile (SPMD).

Verifies both in-memory and persistent cache keys are identical across all ranks,
enabling proper NEFF sharing in distributed training.
"""

import glob
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import torch_neuronx
from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.distributed.utils import broadcast_object_from_rank0
from torch_neuronx.neuron_dynamo_backend.compile import (
    ArtifactType,
    clear_compiled_cache_keys,
    get_compiled_cache_keys,
)


@pytest.fixture(autouse=True)
def reset_compile_state():
    """Reset dynamo and cache key tracking before each test."""
    torch._dynamo.reset()
    clear_compiled_cache_keys()
    yield
    torch._dynamo.reset()
    clear_compiled_cache_keys()


def _compile_and_run(device):
    """Create, compile, and run a simple model. Returns after synchronization."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.linear(x)

    compiled = torch.compile(Model().to(device), backend="neuron", fullgraph=True)
    with torch.no_grad():
        compiled(torch.randn(2, 64).to(device))
    torch.neuron.synchronize()


def run_in_memory_cache_key_test(rank, world_size, kwargs):
    """Test that in-memory cache keys match across all ranks."""
    device = torch.device(f"neuron:{torch.neuron.current_device()}")
    _compile_and_run(device)

    cache_keys = get_compiled_cache_keys(ArtifactType.NEURON_NEFF)
    assert len(cache_keys) == 1, f"Expected 1 cache key, got {len(cache_keys)}"
    local_key = next(iter(cache_keys))

    # Broadcast rank 0's key and verify all ranks match (host-side; see helper)
    rank0_key = [broadcast_object_from_rank0(local_key, rank, world_size, tag="cache_key")]

    assert (
        local_key == rank0_key[0]
    ), f"Cache key mismatch: rank 0='{rank0_key[0][:40]}...', rank {rank}='{local_key[:40]}...'"


def run_persistent_cache_test(rank, world_size, kwargs):
    """Test that all ranks share a single NEFF in persistent cache."""
    device = torch.device(f"neuron:{torch.neuron.current_device()}")

    # Compilation cache should be cleared from any previous NEFFs for this check
    torch_neuronx._C._clear_compilation_cache()
    dist.barrier()

    _compile_and_run(device)

    if rank == 0:
        neff_files = glob.glob(os.path.join(kwargs["cache_dir"], "**", "*.neff"), recursive=True)
        assert (
            len(neff_files) == 1
        ), f"Expected 1 NEFF (shared), found {len(neff_files)}. SPMD cache sharing is broken."


class TestCacheKeySPMD(BaseCollectiveOpTest):
    """Test cache key consistency across ranks for torch.compile."""

    @pytest.mark.multi_device
    def test_in_memory_cache_keys_match(self):
        """Verify all ranks produce identical in-memory cache keys."""
        self.distributed_tester.run_test(run_in_memory_cache_key_test)

    @pytest.mark.multi_device
    def test_persistent_cache_single_neff(self):
        """Verify all ranks share one NEFF file in persistent cache."""
        with tempfile.TemporaryDirectory() as cache_dir:
            os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = cache_dir
            self.distributed_tester.run_test(run_persistent_cache_test, cache_dir=cache_dir)
