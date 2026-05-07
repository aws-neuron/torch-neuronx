"""
End-to-end tests for shard_dim_alltoall collective operation legalization.

Tests the complete compilation pipeline with torch.compile and the neuron backend
for models containing torch.ops._dtensor.shard_dim_alltoall operations.
"""

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from torch_neuronx.neuron_dynamo_backend import set_model_name


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset dynamo compilation cache before each test"""
    torch._dynamo.reset()


def run_shard_dim_alltoall_basic_test(rank, world_size, kwargs):
    """Test basic shard_dim_alltoall with gather_dim=1, shard_dim=0"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD
    group_name = funcol._resolve_group_name(process_group)

    set_model_name(f"shard_dim_alltoall_basic_rank{rank}")

    rows_per_rank = 4 // world_size
    x = torch.full((4, 8), float(rank), device=device)

    def model(t):
        return torch.ops._dtensor.shard_dim_alltoall(
            t, gather_dim=0, shard_dim=0, group_name=group_name
        )

    compiled_fn = torch.compile(model, backend="neuron")
    output = compiled_fn(x.clone())

    torch.neuron.synchronize()

    expected = torch.cat([torch.full((rows_per_rank, 8), float(r)) for r in range(world_size)])
    torch.testing.assert_close(output.cpu(), expected)


def run_shard_dim_alltoall_with_matmul_test(rank, world_size, kwargs):
    """Test shard_dim_alltoall followed by matmul"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD
    group_name = funcol._resolve_group_name(process_group)

    set_model_name(f"shard_dim_alltoall_matmul_rank{rank}")

    cols = 16 // world_size
    x = torch.full((8, cols), float(rank), device=device)
    weight = torch.eye(16, 32, device=device)

    def model(t, w):
        t = torch.ops._dtensor.shard_dim_alltoall(
            t, gather_dim=1, shard_dim=0, group_name=group_name
        )
        return torch.matmul(t, w)

    compiled_fn = torch.compile(model, backend="neuron")
    output = compiled_fn(x.clone(), weight)

    torch.neuron.synchronize()

    rows_out = 8 // world_size
    alltoall_expected = torch.cat(
        [torch.full((rows_out, cols), float(r)) for r in range(world_size)], dim=1
    )
    expected = torch.zeros(rows_out, 32)
    expected[:, :16] = alltoall_expected
    torch.testing.assert_close(output.cpu(), expected)


def run_shard_dim_alltoall_different_dims_test(rank, world_size, kwargs):
    """Test shard_dim_alltoall with gather_dim=0, shard_dim=1"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD
    group_name = funcol._resolve_group_name(process_group)

    set_model_name(f"shard_dim_alltoall_diff_dims_rank{rank}")

    x = torch.full((16, 8), float(rank), device=device)

    def model(t):
        return torch.ops._dtensor.shard_dim_alltoall(
            t, gather_dim=0, shard_dim=1, group_name=group_name
        )

    compiled_fn = torch.compile(model, backend="neuron")
    output = compiled_fn(x.clone())

    torch.neuron.synchronize()

    cols_out = 8 // world_size
    expected = torch.cat([torch.full((16, cols_out), float(r)) for r in range(world_size)], dim=0)
    torch.testing.assert_close(output.cpu(), expected)


def run_shard_dim_alltoall_with_relu_test(rank, world_size, kwargs):
    """Test shard_dim_alltoall with ReLU activation"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD
    group_name = funcol._resolve_group_name(process_group)

    set_model_name(f"shard_dim_alltoall_relu_rank{rank}")

    x = torch.full((8, 16), float(rank) - world_size / 2.0, device=device)

    def model(t):
        t = torch.ops._dtensor.shard_dim_alltoall(
            t, gather_dim=1, shard_dim=0, group_name=group_name
        )
        return torch.relu(t)

    compiled_fn = torch.compile(model, backend="neuron")
    output = compiled_fn(x.clone())

    torch.neuron.synchronize()

    rows_out = 8 // world_size
    alltoall_expected = torch.cat(
        [torch.full((rows_out, 16), float(r) - world_size / 2.0) for r in range(world_size)], dim=1
    )
    expected = torch.relu(alltoall_expected)
    torch.testing.assert_close(output.cpu(), expected)


def run_shard_dim_alltoall_3d_tensor_test(rank, world_size, kwargs):
    """Test shard_dim_alltoall with 3D tensor"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD
    group_name = funcol._resolve_group_name(process_group)

    set_model_name(f"shard_dim_alltoall_3d_rank{rank}")

    x = torch.full((4, 8, 16), float(rank), device=device)

    def model(t):
        return torch.ops._dtensor.shard_dim_alltoall(
            t, gather_dim=2, shard_dim=0, group_name=group_name
        )

    compiled_fn = torch.compile(model, backend="neuron")
    output = compiled_fn(x.clone())

    torch.neuron.synchronize()

    d0 = 4 // world_size
    expected = torch.cat([torch.full((d0, 8, 16), float(r)) for r in range(world_size)], dim=2)
    torch.testing.assert_close(output.cpu(), expected)


class TestShardDimAllToAll(BaseCollectiveOpTest):
    """Test class for shard_dim_alltoall collective operation using DistributedTester."""

    @pytest.mark.multi_device
    def test_shard_dim_alltoall_basic(self):
        """Test basic shard_dim_alltoall operation."""
        self.distributed_tester.run_test(run_shard_dim_alltoall_basic_test)

    @pytest.mark.multi_device
    def test_shard_dim_alltoall_with_matmul(self):
        """Test shard_dim_alltoall followed by matmul."""
        self.distributed_tester.run_test(run_shard_dim_alltoall_with_matmul_test)

    @pytest.mark.multi_device
    def test_shard_dim_alltoall_different_dims(self):
        """Test shard_dim_alltoall with different gather/shard dimensions."""
        self.distributed_tester.run_test(run_shard_dim_alltoall_different_dims_test)

    @pytest.mark.multi_device
    def test_shard_dim_alltoall_with_relu(self):
        """Test shard_dim_alltoall with ReLU activation."""
        self.distributed_tester.run_test(run_shard_dim_alltoall_with_relu_test)

    @pytest.mark.multi_device
    def test_shard_dim_alltoall_3d_tensor(self):
        """Test shard_dim_alltoall with 3D tensor."""
        self.distributed_tester.run_test(run_shard_dim_alltoall_3d_tensor_test)
