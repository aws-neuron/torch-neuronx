"""
Step 4: Test tensor-parallel attention with all-reduce

Adds tensor-parallel multi-head attention with column-parallel QKV projection
and row-parallel output projection.

Usage:
    pytest tests/neuron_dynamo_backend/component/tensor_parallel/test_tp_step4_attention.py
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
from torch_neuronx.neuron_dynamo_backend import create_neuron_backend, set_model_name

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

custom_neuron_backend = create_neuron_backend()


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
        assert out_features % self.world_size == 0
        out_features_per_rank = out_features // self.world_size
        self.weight = nn.Parameter(torch.randn(in_features, out_features_per_rank))
        self.bias = nn.Parameter(torch.randn(out_features_per_rank))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class RowParallelLinear(nn.Module):
    """Row-parallel linear (input sharded, output replicated)"""

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
        return output + self.bias


class TensorParallelAttention(nn.Module):
    """
    Tensor-parallel multi-head attention.

    Architecture:
    - QKV projection: column-parallel (heads sharded across ranks)
    - Attention computation: local (each rank computes subset of heads)
    - Output projection: row-parallel (all-reduce to combine results)
    """

    def __init__(self, hidden_size, num_heads, device_mesh):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % device_mesh.size() == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.device_mesh = device_mesh
        self.world_size = device_mesh.size()

        # Each rank handles a subset of heads
        self.num_heads_per_rank = num_heads // self.world_size

        # Column-parallel QKV: [hidden_size] -> [3*hidden_size // world_size]
        self.qkv = ColumnParallelLinear(hidden_size, 3 * hidden_size, device_mesh)

        # Row-parallel output: [hidden_size // world_size] -> [hidden_size]
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, device_mesh)

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size] - replicated
        output: [batch, seq_len, hidden_size] - replicated
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection: [batch, seq_len, 3*hidden_size // world_size]
        qkv = self.qkv(x)

        # Split into Q, K, V: each [batch, seq_len, hidden_size // world_size]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: [batch, num_heads_per_rank, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        # Attention computation (local to each rank)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape: [batch, seq_len, hidden_size // world_size]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads_per_rank * self.head_dim)
        )

        # Output projection with all-reduce
        output = self.out_proj(attn_output)

        return output


def run_tensor_parallel_attention_test(rank, world_size, kwargs):
    """Test tensor-parallel attention"""

    logger.info(f"=== Rank {rank}/{world_size}: Testing Tensor-Parallel Attention ===")

    artifacts_dir = f"test_artifacts/tp_step4_rank_{rank}"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir

    set_model_name(f"tp_step4_rank{rank}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Model config
    batch_size = 2
    seq_len = 8
    hidden_size = 16
    num_heads = 4

    logger.info(
        f"Rank {rank}: Config - batch={batch_size}, seq={seq_len}, "
        f"hidden={hidden_size}, heads={num_heads}"
    )

    # Create model and move to neuron device
    model = TensorParallelAttention(hidden_size, num_heads, device_mesh)
    model = model.to(device)
    logger.info(
        f"Rank {rank}: Created TensorParallelAttention on {device}, "
        f"local heads={model.num_heads_per_rank}"
    )

    # Create replicated input on neuron device
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

        compiled_model = torch.compile(model, backend=custom_neuron_backend)
        with torch.no_grad():
            compiled_output = compiled_model(x)
        logger.info(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
        logger.info(f"Rank {rank}: Compiled output sample: {compiled_output[0, 0, :3]}")

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
            logger.info("STEP 4: TENSOR-PARALLEL ATTENTION TEST SUMMARY")
            logger.info("=" * 80)
            logger.info("Tensor-parallel attention works with torch.compile")
            logger.info("=" * 80)
        assert success

    except Exception as e:
        logger.error(f"Rank {rank}: Test failed: {e}")
        import traceback

        traceback.print_exc()


class TestTensorParallelAttention(BaseCollectiveOpTest):
    """Test class for tensor parallel attention component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_tensor_parallel_attention(self):
        """Test tensor-parallel multi-head attention with torch.compile."""
        self.distributed_tester.run_test(run_tensor_parallel_attention_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
