import logging
import os

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

torch._dynamo.config.automatic_dynamic_shapes = False


class ReduceScatterModel(nn.Module):
    """Simple model that uses reduce_scatter"""

    def __init__(self, device_mesh, reduce_op="sum"):
        super().__init__()
        self.device_mesh = device_mesh
        self.reduce_op = reduce_op

    def forward(self, x):
        """
        x: [N, D] where N is divisible by world_size
        output: [N/world_size, D] - each rank gets a slice
        """
        return funcol.reduce_scatter_tensor(
            x, reduceOp=self.reduce_op, scatter_dim=0, group=(self.device_mesh, 0)
        )


class ReduceScatter2DModel(nn.Module):
    """Model that uses reduce_scatter on 2D tensors"""

    def __init__(self, device_mesh, reduce_op="sum", scatter_dim=0):
        super().__init__()
        self.device_mesh = device_mesh
        self.reduce_op = reduce_op
        self.scatter_dim = scatter_dim

    def forward(self, x):
        """
        x: [N, H, W] where dimension at scatter_dim is divisible by world_size
        output: tensor with scatter_dim reduced by world_size
        """
        return funcol.reduce_scatter_tensor(
            x, reduceOp=self.reduce_op, scatter_dim=self.scatter_dim, group=(self.device_mesh, 0)
        )


class ReduceScatterWithComputeModel(nn.Module):
    """Model that combines reduce_scatter with computation"""

    def __init__(self, device_mesh, hidden_size):
        super().__init__()
        self.device_mesh = device_mesh
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def forward(self, x):
        """
        x: [N, hidden_size] where N is divisible by world_size
        output: [N/world_size, hidden_size]
        """
        # First do some computation
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)

        # Then reduce_scatter
        x = funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=(self.device_mesh, 0)
        )

        return x


def _run_reduce_scatter_basic(rank, world_size, kwargs):
    """Test 1: Basic reduce_scatter with sum"""
    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"\n{'='*80}")
    logger.debug(f"TEST 1: Basic reduce_scatter with sum (Rank {rank}/{world_size})")
    logger.debug(f"{'='*80}")

    artifacts_dir = f"test_artifacts/reduce_scatter_basic_rank_{rank}"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir
    set_model_name(f"reduce_scatter_basic_rank{rank}")

    # Create model and move to neuron device
    model = ReduceScatterModel(device_mesh, reduce_op="sum")
    model = model.to(device)

    # Create input: each rank contributes different values
    # For world_size=2, input_size=4:
    # Rank 0: [0, 1, 2, 3]
    # Rank 1: [10, 11, 12, 13]
    # After reduce_scatter:
    # Rank 0 gets: [0+10, 1+11] = [10, 12]
    # Rank 1 gets: [2+12, 3+13] = [14, 16]
    input_size = 32
    torch.manual_seed(rank)
    x = torch.arange(input_size, dtype=torch.float32, device=device) + rank * 10

    logger.debug(f"Rank {rank}: Input: {x}, device: {x.device}")

    # Eager execution
    logger.debug(f"Rank {rank}: Testing eager...")
    with torch.no_grad():
        eager_output = model(x)
    logger.debug(f"Rank {rank}: Eager output: {eager_output}")

    # Compiled execution
    logger.debug(f"Rank {rank}: Testing compiled...")
    compiled_model = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_output = compiled_model(x)
    logger.debug(f"Rank {rank}: Compiled output: {compiled_output}")

    # Verify
    if torch.allclose(eager_output, compiled_output, atol=1e-4, rtol=1e-3):
        logger.debug(f"Rank {rank}: Results match!")
        success = True
    else:
        logger.error(f"Rank {rank}: Results don't match")
        logger.error(f"  Eager: {eager_output}")
        logger.error(f"  Compiled: {compiled_output}")
        logger.error(f"  Diff: {torch.abs(eager_output - compiled_output)}")
        success = False

    dist.barrier()

    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST 1 RESULT: {'PASSED' if success else 'FAILED'}")
        logger.info(f"{'='*80}")

    assert success


def _run_reduce_scatter_2d(rank, world_size, kwargs):
    """Test 2: reduce_scatter with 2D tensors"""
    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"\n{'='*80}")
    logger.debug(f"TEST 2: reduce_scatter with 2D tensors (Rank {rank}/{world_size})")
    logger.debug(f"{'='*80}")

    artifacts_dir = f"test_artifacts/reduce_scatter_2d_rank_{rank}"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir
    set_model_name(f"reduce_scatter_2d_rank{rank}")

    # Create model and move to neuron device
    model = ReduceScatter2DModel(device_mesh, reduce_op="sum")
    model = model.to(device)

    # Create 2D input: [world_size * 2, 3] - first dim must be divisible by world_size
    # Each rank has different values
    torch.manual_seed(rank)
    x = torch.ones(world_size * 2, 3, device=device) * (rank + 1)

    logger.debug(f"Rank {rank}: Input shape: {x.shape}, device: {x.device}")
    logger.debug(f"Rank {rank}: Input:\n{x}")

    # Eager execution
    logger.debug(f"Rank {rank}: Testing eager...")
    with torch.no_grad():
        eager_output = model(x)
    logger.debug(f"Rank {rank}: Eager output shape: {eager_output.shape}")
    logger.debug(f"Rank {rank}: Eager output:\n{eager_output}")

    # Compiled execution
    logger.debug(f"Rank {rank}: Testing compiled...")
    compiled_model = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_output = compiled_model(x)
    logger.debug(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
    logger.debug(f"Rank {rank}: Compiled output:\n{compiled_output}")

    # Verify
    if torch.allclose(eager_output, compiled_output, atol=1e-4, rtol=1e-3):
        logger.debug(f"Rank {rank}: Results match!")
        success = True
    else:
        logger.error(f"Rank {rank}: Results don't match")
        logger.error(f"  Max diff: {torch.max(torch.abs(eager_output - compiled_output))}")
        success = False

    dist.barrier()

    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST 2 RESULT: {'PASSED' if success else 'FAILED'}")
        logger.info(f"{'='*80}")

    assert success


def _run_reduce_scatter_with_compute(rank, world_size, kwargs):
    """Test 3: reduce_scatter combined with computation"""
    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"\n{'='*80}")
    logger.debug(f"TEST 3: reduce_scatter with computation (Rank {rank}/{world_size})")
    logger.debug(f"{'='*80}")

    artifacts_dir = f"test_artifacts/reduce_scatter_compute_rank_{rank}"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir
    set_model_name(f"reduce_scatter_compute_rank{rank}")

    # Create model
    hidden_size = 8
    model = ReduceScatterWithComputeModel(device_mesh, hidden_size)
    model = model.to(device)

    # Create input - first dim must be divisible by world_size
    input_size = world_size * 2
    torch.manual_seed(42)  # Same seed for all ranks
    x = torch.randn(input_size, hidden_size, device=device)

    logger.debug(f"Rank {rank}: Input shape: {x.shape}, device: {x.device}")

    # Eager execution
    logger.debug(f"Rank {rank}: Testing eager...")
    with torch.no_grad():
        eager_output = model(x)
    logger.debug(f"Rank {rank}: Eager output shape: {eager_output.shape}")
    logger.debug(f"Rank {rank}: Eager output sample: {eager_output[0, :3]}")

    # Compiled execution
    logger.debug(f"Rank {rank}: Testing compiled...")
    compiled_model = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_output = compiled_model(x)
    logger.debug(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
    logger.debug(f"Rank {rank}: Compiled output sample: {compiled_output[0, :3]}")

    # Verify - both tensors are on neuron device
    max_diff = torch.max(torch.abs(eager_output - compiled_output))
    if torch.allclose(eager_output, compiled_output, atol=1e-3, rtol=1e-2):
        logger.debug(f"Rank {rank}: Results match! (max diff: {max_diff:.6f})")
        success = True
    else:
        logger.error(f"Rank {rank}: Results don't match")
        logger.error(f"  Max diff: {max_diff}")
        logger.error(f"  Relative error: {max_diff / torch.abs(eager_output).mean():.6f}")
        success = False

    dist.barrier()

    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST 3 RESULT: {'PASSED' if success else 'FAILED'}")
        logger.info(f"{'='*80}")

    assert success


def _run_reduce_scatter_avg(rank, world_size, kwargs):
    """Test 4: reduce_scatter with avg reduction (tests transformation)"""
    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"\n{'='*80}")
    logger.debug(f"TEST 4: reduce_scatter with avg reduction (Rank {rank}/{world_size})")
    logger.debug(f"{'='*80}")

    artifacts_dir = f"test_artifacts/reduce_scatter_seqdim_rank_{rank}"
    os.environ["TORCH_NEURONX_DEBUG_DIR"] = artifacts_dir
    set_model_name(f"reduce_scatter_avg_rank{rank}")

    # Create model with avg reduction and move to neuron device
    model = ReduceScatterModel(device_mesh, reduce_op="avg")
    model = model.to(device)

    # Create input on neuron device: uniform values per rank
    # For world_size=2, input_size=4:
    # Rank 0: [1, 1, 1, 1]
    # Rank 1: [2, 2, 2, 2]
    # After reduce_scatter with avg:
    # Rank 0 gets: [(1+2)/2, (1+2)/2] = [1.5, 1.5]
    # Rank 1 gets: [(1+2)/2, (1+2)/2] = [1.5, 1.5]
    input_size = 8
    x = torch.full((input_size,), float(rank + 1), device=device)

    logger.debug(f"Rank {rank}: Input: {x}, device: {x.device}")

    # Eager execution (note: PyTorch's native avg may not work correctly)
    logger.debug(f"Rank {rank}: Testing eager...")
    with torch.no_grad():
        eager_output = model(x)
    logger.debug(f"Rank {rank}: Eager output: {eager_output}")

    # Compiled execution (should use our transformation: sum + divide)
    logger.debug(f"Rank {rank}: Testing compiled...")
    compiled_model = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_output = compiled_model(x)
    logger.debug(f"Rank {rank}: Compiled output: {compiled_output}")

    # Expected value: average of all ranks = (1 + 2 + ... + world_size) / world_size
    expected_value = sum(range(1, world_size + 1)) / world_size
    expected_output = torch.full((input_size // world_size,), expected_value, device=device)
    logger.debug(f"Rank {rank}: Expected output: {expected_output}")

    # Verify compiled output against expected (not eager, since eager may be wrong)
    if torch.allclose(compiled_output, expected_output, atol=1e-4, rtol=1e-3):
        logger.debug(f"Rank {rank}: Compiled output matches expected!")
        success = True
    else:
        logger.error(f"Rank {rank}: Compiled output doesn't match expected")
        logger.error(f"  Compiled: {compiled_output}")
        logger.error(f"  Expected: {expected_output}")
        logger.error(f"  Diff: {torch.abs(compiled_output - expected_output)}")
        success = False

    # Also check if eager matches (it probably won't, which is expected)
    if not torch.allclose(eager_output, expected_output, atol=1e-4, rtol=1e-3):
        logger.warning(
            f"Rank {rank}: Eager output doesn't match expected (this is expected "
            f"- PyTorch's native avg may not work)"
        )
        logger.warning(f"  Eager: {eager_output}")
        logger.warning(f"  Expected: {expected_output}")

    dist.barrier()

    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST 4 RESULT: {'PASSED' if success else 'FAILED'}")
        logger.info("Note: This test validates that our StableHLO transformation")
        logger.info("      correctly implements avg as sum + divide by world_size")
        logger.info(f"{'='*80}")

    assert success


def _run_reduce_scatter_sequence_dim(rank, world_size, kwargs):
    """Test 5: reduce_scatter on sequence dimension (Llama3 pattern)"""
    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"\n{'='*80}")
    logger.debug(
        f"TEST 5: reduce_scatter on sequence dim (Llama3 pattern) (Rank {rank}/{world_size})"
    )
    logger.debug(f"{'='*80}")

    set_model_name(f"reduce_scatter_seqdim_rank{rank}")

    # Create model that scatters on dim=1 (sequence dimension) and move to neuron device
    model = ReduceScatter2DModel(device_mesh, reduce_op="sum", scatter_dim=1)
    model = model.to(device)

    # Create 3D input matching Llama3 pattern: [batch, seq_len, hidden_dim]
    # seq_len must be divisible by world_size
    batch_size = 2
    seq_len = world_size * 16  # e.g., 128 for world_size=8
    hidden_dim = 64

    torch.manual_seed(rank)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device) * (rank + 1)

    logger.debug(f"Rank {rank}: Input shape: {x.shape}, device: {x.device}")
    logger.debug(f"Rank {rank}: Input stats - mean: {x.mean():.6f}, std: {x.std():.6f}")

    # Eager execution
    logger.debug(f"Rank {rank}: Testing eager...")
    with torch.no_grad():
        eager_output = model(x)
    logger.debug(f"Rank {rank}: Eager output shape: {eager_output.shape}")
    logger.debug(
        f"Rank {rank}: Eager output stats - mean: {eager_output.mean():.6f}, "
        f"std: {eager_output.std():.6f}"
    )

    # Compiled execution
    logger.debug(f"Rank {rank}: Testing compiled...")
    compiled_model = torch.compile(model, backend="neuron")
    with torch.no_grad():
        compiled_output = compiled_model(x)
    logger.debug(f"Rank {rank}: Compiled output shape: {compiled_output.shape}")
    logger.debug(
        f"Rank {rank}: Compiled output stats - mean: {compiled_output.mean():.6f}, "
        f"std: {compiled_output.std():.6f}"
    )

    # Verify shapes (seq_len=32, world_size=2, so output seq_len=16)
    expected_shape = (batch_size, 16, hidden_dim)
    if compiled_output.shape != expected_shape:
        logger.error(f"Rank {rank}: Output shape mismatch!")
        logger.error(f"  Expected: {expected_shape}")
        logger.error(f"  Got: {compiled_output.shape}")
        success = False
    elif torch.allclose(eager_output, compiled_output, atol=1e-4, rtol=1e-3):
        logger.debug(f"Rank {rank}: Results match!")
        success = True
    else:
        logger.error(f"Rank {rank}: Results don't match")
        max_diff = torch.max(torch.abs(eager_output - compiled_output))
        logger.error(f"  Max diff: {max_diff}")
        success = False

    dist.barrier()

    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST 5 RESULT: {'PASSED' if success else 'FAILED'}")
        logger.info("Note: This test matches the Llama3 TP pattern where")
        logger.info("      reduce_scatter operates on the sequence dimension (dim=1)")
        logger.info(f"{'='*80}")

    assert success


class TestReduceScatterBasic(BaseCollectiveOpTest):
    @pytest.mark.multi_device
    def test_reduce_scatter_compile(self):
        """Test reduce_scatter operations with torch.compile."""
        self.distributed_tester.run_test(_run_reduce_scatter_basic)


class TestReduceScatter2D(BaseCollectiveOpTest):
    @pytest.mark.multi_device
    def test_reduce_scatter_compile(self):
        """Test reduce_scatter operations with torch.compile."""
        self.distributed_tester.run_test(_run_reduce_scatter_2d)


class TestReduceScatterCompute(BaseCollectiveOpTest):
    @pytest.mark.multi_device
    def test_reduce_scatter_compile(self):
        """Test reduce_scatter operations with torch.compile."""
        self.distributed_tester.run_test(_run_reduce_scatter_with_compute)


class TestReduceScatterAverage(BaseCollectiveOpTest):
    @pytest.mark.multi_device
    def test_reduce_scatter_compile(self):
        """Test reduce_scatter operations with torch.compile."""
        self.distributed_tester.run_test(_run_reduce_scatter_avg)


class TestReduceScatterSequenceDim(BaseCollectiveOpTest):
    """Test class for reduce_scatter compile tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_reduce_scatter_compile(self):
        """Test reduce_scatter operations with torch.compile."""
        self.distributed_tester.run_test(_run_reduce_scatter_sequence_dim)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
