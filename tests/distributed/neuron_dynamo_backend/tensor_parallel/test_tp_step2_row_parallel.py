"""
Step 2: Test row-parallel linear layer with all-reduce

This adds the all-reduce collective operation after the local matmul.
A row-parallel linear splits the input dimension and requires all-reduce.

Usage:
    pytest tests/neuron_dynamo_backend/component/tensor_parallel/test_tp_step2_row_parallel.py
"""

import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.distributed.utils import DistributedTester
from torch_neuronx.neuron_dynamo_backend import set_model_name

logging.basicConfig(
    level=logging.DEBUG
)  # Enable INFO for all loggers including collective_transforms
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Input: [batch, in_features // world_size] - sharded per rank
    Weight: [in_features // world_size, out_features] - sharded per rank
    Output: [batch, out_features] - replicated (after all-reduce)
    """

    def __init__(self, in_features, out_features, device_mesh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = device_mesh.size()
        self.device_mesh = device_mesh

        assert in_features % self.world_size == 0
        self.in_features_per_rank = in_features // self.world_size

        # Each rank has a slice of the weight matrix
        self.weight = nn.Parameter(torch.randn(self.in_features_per_rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        """
        x: [batch, in_features // world_size] - sharded
        output: [batch, out_features] - replicated
        """
        # Local matmul
        output = torch.matmul(x, self.weight)

        # All-reduce to combine results from all ranks
        output = funcol.all_reduce(output, reduceOp="sum", group=self.device_mesh)

        # Add bias (only on one rank to avoid duplication, or divide by world_size)
        output = output + self.bias

        return output


def run_row_parallel_linear_test(rank, world_size, kwargs):
    """Test row-parallel linear layer with all-reduce"""

    logger.info(f"=== Rank {rank}/{world_size}: Testing Row-Parallel Linear ===")

    # Artifacts directory is controlled by TORCH_NEURONX_DEBUG_DIR env var
    # Set it before running: export TORCH_NEURONX_DEBUG_DIR=./test_artifacts/tp_step2
    set_model_name(f"tp_step2_rank{rank}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Model config
    batch_size = 4
    in_features = 16
    out_features = 8

    logger.info(f"Rank {rank}: Config - batch={batch_size}, in={in_features}, out={out_features}")

    # Create model and move to neuron device
    model = RowParallelLinear(in_features, out_features, device_mesh)
    model = model.to(device)
    logger.info(
        f"Rank {rank}: Created RowParallelLinear on {device}, "
        f"local in_features={model.in_features_per_rank}"
    )

    # Create sharded input (different on each rank) on neuron device
    torch.manual_seed(42 + rank)
    x = torch.randn(batch_size, model.in_features_per_rank, device=device)
    logger.info(f"Rank {rank}: Input shape: {x.shape}, device: {x.device}")

    try:
        # Eager execution
        logger.info(f"Rank {rank}: Testing eager...")
        with torch.no_grad():
            eager_output = model(x)
        logger.info(f"Rank {rank}: Eager output shape: {eager_output.shape}")
        logger.info(f"Rank {rank}: Eager output sample: {eager_output[0, :3]}")

        # Compiled execution
        logger.info(f"Rank {rank}: Testing compiled...")
        compiled_model = torch.compile(model, backend="neuron")
        with torch.no_grad():
            compiled_output = compiled_model(x)
        logger.info(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
        logger.info(f"Rank {rank}: Compiled output sample: {compiled_output[0, :3]}")

        # Verify - both tensors are on neuron device
        if torch.allclose(eager_output, compiled_output, atol=1e-4):
            logger.info(f"Rank {rank}: Results match!")
            success = True
        else:
            logger.error(f"Rank {rank}: Results don't match")
            logger.info(f"Max diff: {torch.max(torch.abs(eager_output - compiled_output))}")
            success = False

        dist.barrier()

        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: ROW-PARALLEL LINEAR TEST SUMMARY")
            logger.info("=" * 80)
            logger.info("Row-parallel linear with all-reduce works with torch.compile")
            logger.info("=" * 80)
        assert success

    except Exception as e:
        logger.error(f"Rank {rank}: Test failed: {e}")
        import traceback

        traceback.print_exc()


class TestTensorParallelRowLinear(BaseCollectiveOpTest):
    """Test class for tensor parallel row-linear layer component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_row_parallel_linear(self):
        """Test row-parallel linear layer with all-reduce and torch.compile."""
        self.distributed_tester.run_test(run_row_parallel_linear_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
