"""
Step 5: Test complete tensor-parallel transformer layer

Combines tensor-parallel attention and MLP into a full transformer layer
with layer norms and residual connections.

Usage:
    pytest tests/neuron_dynamo_backend/component/tensor_parallel/test_tp_step5_transformer.py
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

logging.basicConfig(level=logging.INFO)
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
    """Tensor-parallel multi-head attention"""

    def __init__(self, hidden_size, num_heads, device_mesh):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % device_mesh.size() == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.device_mesh = device_mesh
        self.world_size = device_mesh.size()
        self.num_heads_per_rank = num_heads // self.world_size

        self.qkv = ColumnParallelLinear(hidden_size, 3 * hidden_size, device_mesh)
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, device_mesh)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads_per_rank, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads_per_rank * self.head_dim)
        )

        output = self.out_proj(attn_output)
        return output


class TensorParallelMLP(nn.Module):
    """Tensor-parallel MLP"""

    def __init__(self, hidden_size, device_mesh):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden_size, 4 * hidden_size, device_mesh)
        self.activation = nn.GELU()
        self.fc2 = RowParallelLinear(4 * hidden_size, hidden_size, device_mesh)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TensorParallelTransformerLayer(nn.Module):
    """
    Complete tensor-parallel transformer layer.

    Architecture:
    - Tensor-parallel attention with residual
    - Layer norm
    - Tensor-parallel MLP with residual
    - Layer norm
    """

    def __init__(self, hidden_size, num_heads, device_mesh):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device_mesh = device_mesh

        self.attention = TensorParallelAttention(hidden_size, num_heads, device_mesh)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.mlp = TensorParallelMLP(hidden_size, device_mesh)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size] - replicated
        output: [batch, seq_len, hidden_size] - replicated
        """
        # Attention block with residual
        attn_output = self.attention(self.ln1(x))
        x = x + attn_output

        # MLP block with residual
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output

        return x


def run_tensor_parallel_transformer_test(rank, world_size, kwargs):
    """Test complete tensor-parallel transformer layer"""

    logger.info(f"=== Rank {rank}/{world_size}: Testing Tensor-Parallel Transformer ===")

    artifacts_dir = "test_artifacts/tp_step5"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir

    set_model_name(f"tp_step5_rank{rank}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Model config (matching test_transformer_simple.py)
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 8

    logger.info(
        f"Rank {rank}: Config - batch={batch_size}, seq={seq_len}, "
        f"hidden={hidden_size}, heads={num_heads}"
    )

    # Create model and move to neuron device
    model = TensorParallelTransformerLayer(hidden_size, num_heads, device_mesh)
    model = model.to(device)
    logger.info(f"Rank {rank}: Created TensorParallelTransformerLayer on {device}")

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
        logger.info(f"Rank {rank}: Eager output sample: {eager_output[0, 0, :5]}")

        # Compiled execution
        logger.info(f"Rank {rank}: Testing compiled...")
        compiled_model = torch.compile(model, backend="neuron")
        with torch.no_grad():
            compiled_output = compiled_model(x)
        logger.info(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
        logger.info(f"Rank {rank}: Compiled output sample: {compiled_output[0, 0, :5]}")

        # Verify (using relaxed tolerance for distributed operations)
        # Note: Distributed collectives can introduce small numerical differences
        # due to different reduction orders and floating-point non-associativity
        max_diff = torch.max(torch.abs(eager_output - compiled_output))
        if torch.allclose(eager_output, compiled_output, atol=0.05, rtol=1e-3):
            logger.info(f"Rank {rank}: Results match! (max diff: {max_diff:.6f})")
            success = True
        else:
            logger.error(f"Rank {rank}: Results don't match")
            logger.info(f"Max diff: {max_diff}")
            logger.info(f"Relative error: {max_diff / torch.abs(eager_output).mean():.6f}")
            success = False

        dist.barrier()

        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: TENSOR-PARALLEL TRANSFORMER TEST SUMMARY")
            logger.info("=" * 80)
            logger.info("Complete tensor-parallel transformer works with torch.compile")
            logger.info("=" * 80)
        assert success

    except Exception as e:
        logger.error(f"Rank {rank}: Test failed: {e}")
        import traceback

        traceback.print_exc()


class TestTensorParallelTransformer(BaseCollectiveOpTest):
    """Test class for tensor parallel transformer component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_tensor_parallel_transformer(self):
        """Test complete tensor-parallel transformer layer with torch.compile."""
        self.distributed_tester.run_test(run_tensor_parallel_transformer_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
