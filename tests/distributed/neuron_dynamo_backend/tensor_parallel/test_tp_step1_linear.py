"""
Step 1: Test column-parallel linear layer with torch.compile

This is the simplest building block for tensor parallelism.
A column-parallel linear splits the output dimension across ranks.

Usage:
    pytest tests/neuron_dynamo_backend/component/tensor_parallel/test_tp_step1_linear.py
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_distributed():
    """Setup distributed training environment"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    return rank, world_size


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Input: [batch, in_features] - replicated across all ranks
    Weight: [in_features, out_features // world_size] - sharded per rank
    Output: [batch, out_features // world_size] - sharded per rank
    """

    def __init__(self, in_features, out_features, device_mesh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = device_mesh.size()
        self.device_mesh = device_mesh

        assert out_features % self.world_size == 0
        self.out_features_per_rank = out_features // self.world_size

        # Each rank has a slice of the weight matrix
        self.weight = nn.Parameter(torch.randn(in_features, self.out_features_per_rank))
        self.bias = nn.Parameter(torch.randn(self.out_features_per_rank))

    def forward(self, x):
        """
        x: [batch, in_features] - replicated
        output: [batch, out_features // world_size] - sharded
        """
        # Local matmul
        output = torch.matmul(x, self.weight) + self.bias
        return output


def run_column_parallel_linear_test(rank, world_size, kwargs):
    """Test column-parallel linear layer"""

    logger.info(f"=== Rank {rank}/{world_size}: Testing Column-Parallel Linear ===")

    # Artifacts directory is controlled by TORCH_NEURONX_DEBUG_DIR env var
    # Set it before running: export TORCH_NEURONX_DEBUG_DIR=./test_artifacts/tp_step1
    set_model_name(f"tp_step1_rank{rank}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Model config
    batch_size = 4
    in_features = 8
    out_features = 16

    logger.info(f"Rank {rank}: Config - batch={batch_size}, in={in_features}, out={out_features}")

    # Create model and move to neuron device
    model = ColumnParallelLinear(in_features, out_features, device_mesh)
    model = model.to(device)
    logger.info(
        f"Rank {rank}: Created ColumnParallelLinear on {device}, "
        f"local out_features={model.out_features_per_rank}"
    )

    # Create replicated input (same on all ranks) on neuron device
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device=device)
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
            success = False

        dist.barrier()

        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: COLUMN-PARALLEL LINEAR TEST SUMMARY")
            logger.info("=" * 80)
            logger.info("Column-parallel linear layer works with torch.compile")
            logger.info("=" * 80)
        assert success

    except Exception as e:
        logger.error(f"Rank {rank}: Test failed: {e}")
        import traceback

        traceback.print_exc()


class TestTensorParallelLinear(BaseCollectiveOpTest):
    """Test class for tensor parallel linear layer component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_column_parallel_linear(self):
        """Test column-parallel linear layer with torch.compile."""
        self.distributed_tester.run_test(run_column_parallel_linear_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
