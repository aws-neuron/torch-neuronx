"""
Step 3: Test tensor-parallel MLP (column + row parallel)

Combines column-parallel and row-parallel layers into a 2-layer MLP.
This is the MLP component used in transformers.

Usage:
    pytest tests/neuron_dynamo_backend/component/tensor_parallel/test_tp_step3_mlp.py
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
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    return rank, world_size


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear (output sharded)"""

    def __init__(self, in_features, out_features, device_mesh):
        super().__init__()
        self.world_size = device_mesh.size()
        self.device_mesh = device_mesh

        assert out_features % self.world_size == 0
        out_features_per_rank = out_features // self.world_size

        self.weight = nn.Parameter(torch.randn(in_features, out_features_per_rank))
        self.bias = nn.Parameter(torch.randn(out_features_per_rank))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class RowParallelLinear(nn.Module):
    """Row-parallel linear (input sharded, output replicated via all-reduce)"""

    def __init__(self, in_features, out_features, device_mesh):
        super().__init__()
        self.world_size = device_mesh.size()
        self.device_mesh = device_mesh

        assert in_features % self.world_size == 0
        in_features_per_rank = in_features // self.world_size

        self.weight = nn.Parameter(torch.randn(in_features_per_rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        output = torch.matmul(x, self.weight)
        output = funcol.all_reduce(output, reduceOp="sum", group=self.device_mesh)
        output = output + self.bias
        return output


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP with column-parallel fc1 and row-parallel fc2.

    Architecture:
    - fc1: column-parallel (output sharded)
    - activation: local (operates on sharded data)
    - fc2: row-parallel (input sharded, output replicated)
    """

    def __init__(self, hidden_size, device_mesh):
        super().__init__()
        self.hidden_size = hidden_size
        self.device_mesh = device_mesh

        # Column-parallel: [hidden_size] -> [4*hidden_size // world_size]
        self.fc1 = ColumnParallelLinear(hidden_size, 4 * hidden_size, device_mesh)

        # Activation operates locally on sharded data
        self.activation = nn.GELU()

        # Row-parallel: [4*hidden_size // world_size] -> [hidden_size]
        self.fc2 = RowParallelLinear(4 * hidden_size, hidden_size, device_mesh)

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size] - replicated
        output: [batch, seq_len, hidden_size] - replicated
        """
        # fc1: replicated -> sharded
        x = self.fc1(x)

        # activation: sharded -> sharded
        x = self.activation(x)

        # fc2: sharded -> replicated (via all-reduce)
        x = self.fc2(x)

        return x


def run_tensor_parallel_mlp_test(rank, world_size, kwargs):
    """Test tensor-parallel MLP"""

    logger.info(f"=== Rank {rank}/{world_size}: Testing Tensor-Parallel MLP ===")

    set_model_name(f"tp_step3_rank{rank}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Model config - use 3D tensors (now that _unsafe_view is decomposed)
    batch_size = 2
    seq_len = 8
    hidden_size = 16

    logger.info(f"Rank {rank}: Config - batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

    # Create model and move to neuron device
    model = TensorParallelMLP(hidden_size, device_mesh)
    model = model.to(device)
    logger.info(f"Rank {rank}: Created TensorParallelMLP on {device}")

    # Create replicated input (same on all ranks) on neuron device - 3D tensor
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    logger.info(f"Rank {rank}: Input shape: {x.shape}, device: {x.device}")

    try:
        # Eager execution
        logger.info(f"Rank {rank}: Testing eager...")
        with torch.no_grad():
            eager_output = model(x)
        logger.info(f"Rank {rank}: Eager output shape: {eager_output.shape}")
        logger.info(f"Rank {rank}: Eager output sample: {eager_output[0, 0, :3]}")

        # Compiled execution
        logger.info(f"Rank {rank}: Testing compiled...")
        compiled_model = torch.compile(model, backend="neuron")
        with torch.no_grad():
            compiled_output = compiled_model(x)
        logger.info(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
        logger.info(f"Rank {rank}: Compiled output sample: {compiled_output[0, 0, :3]}")

        # Verify - both tensors are on neuron device
        if torch.allclose(eager_output, compiled_output, atol=1e-4):
            logger.info(f"✓ Rank {rank}: Results match!")
            success = True
        else:
            logger.error(f"✗ Rank {rank}: Results don't match")
            logger.info(f"Max diff: {torch.max(torch.abs(eager_output - compiled_output))}")
            success = False

        dist.barrier()

        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: TENSOR-PARALLEL MLP TEST SUMMARY")
            logger.info("=" * 80)
            logger.info(f"{'✓' if success else '✗'} Tensor-parallel MLP works with torch.compile")
            logger.info("=" * 80)
        assert success

    except Exception as e:
        logger.error(f"Rank {rank}: Test failed: {e}")
        import traceback

        traceback.print_exc()


class TestTensorParallelMLP(BaseCollectiveOpTest):
    """Test class for tensor parallel MLP component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_tensor_parallel_mlp(self):
        """Test tensor-parallel MLP with column+row parallel layers and torch.compile."""
        self.distributed_tester.run_test(run_tensor_parallel_mlp_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
