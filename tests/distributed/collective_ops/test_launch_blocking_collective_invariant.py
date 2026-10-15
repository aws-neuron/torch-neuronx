"""Invariant test: NEURON_LAUNCH_BLOCKING=1 must use the OperationExecutionEngine for collectives.

This is the distributed counterpart to tests/operations/test_launch_blocking_invariant.py.
It verifies that collective ops (which go through XLABuilderKernel) route through the
C++ OperationExecutionEngine when NEURON_LAUNCH_BLOCKING=1 is set.

This test MUST be run with:
  TORCH_NEURONX_CPP_DISPATCH=0 NEURON_LAUNCH_BLOCKING=1

The run-test-parallel script handles setting these env vars and running this test
in isolation (see SKIP_TESTS + the dedicated run_and_record block).
"""

import os

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest

_CPP_DISPATCH_OFF = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "0"
_LAUNCH_BLOCKING_ON = os.environ.get("NEURON_LAUNCH_BLOCKING", "0") == "1"

_SKIP_REASON = (
    "Requires TORCH_NEURONX_CPP_DISPATCH=0 and NEURON_LAUNCH_BLOCKING=1. "
    "Run via: TORCH_NEURONX_CPP_DISPATCH=0 NEURON_LAUNCH_BLOCKING=1 "
    "pytest tests/distributed/collective_ops/test_launch_blocking_collective_invariant.py -vs"
)


def run_all_reduce_uses_engine(rank, world_size, kwargs):
    """Verify all_reduce goes through the engine (CompilationCache gains entries)."""
    import torch_neuronx

    stats_before = torch_neuronx._C._get_compilation_cache_stats()
    before_entries = stats_before["total_entries"]

    tensor = torch.ones(10, dtype=torch.float32).to("neuron")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.neuron.synchronize()

    stats_after = torch_neuronx._C._get_compilation_cache_stats()
    after_entries = stats_after["total_entries"]

    assert after_entries > before_entries, (
        f"Rank {rank}: C++ CompilationCache was not used for all_reduce under "
        f"NEURON_LAUNCH_BLOCKING=1. This indicates the collective bypassed the "
        f"OperationExecutionEngine. Before: {before_entries}, After: {after_entries}"
    )

    expected = torch.ones(10, dtype=torch.float32) * world_size
    assert torch.allclose(tensor.cpu(), expected), (
        f"Rank {rank}: all_reduce result mismatch. " f"Expected {expected}, got {tensor.cpu()}"
    )


@pytest.mark.skipif(not (_CPP_DISPATCH_OFF and _LAUNCH_BLOCKING_ON), reason=_SKIP_REASON)
class TestLaunchBlockingCollectiveInvariant(BaseCollectiveOpTest):
    """Verify NEURON_LAUNCH_BLOCKING=1 routes collectives through the OperationExecutionEngine."""

    def test_all_reduce_uses_engine(self):
        """all_reduce goes through XLABuilderKernel and into the engine."""
        self.distributed_tester.run_test(run_all_reduce_uses_engine)
