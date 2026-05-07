import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def verify_reduce_result(rank, root_rank, tensor_neuron, expected, orig_tensor):
    if rank == root_rank:
        assert torch.allclose(
            tensor_neuron.cpu(), expected
        ), f"Root rank {rank}: Expected {expected}, got {tensor_neuron.cpu()}"
    else:
        assert torch.allclose(
            tensor_neuron.cpu(), orig_tensor
        ), f"Non-root rank {rank}: Expected unchanged {orig_tensor}, got {tensor_neuron.cpu()}"


def run_basic_reduce_test(rank, world_size, kwargs):
    """Test basic reduce functionality with SUM operation."""
    root_rank = kwargs.get("root_rank", 0)

    # Create input tensor with rank-specific values
    tensor = torch.ones(10) * (rank + 1)  # rank 0: [1,1,1...], rank 1: [2,2,2...]
    expected_sum = torch.ones(10) * sum(range(world_size)) + torch.ones(10) * world_size
    # For world_size=2: rank0=[1,1,1], rank1=[2,2,2] -> sum=[3,3,3]

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_different_root_test(rank, world_size, kwargs):
    """Test reduce with different root ranks."""
    root_rank = kwargs.get("root_rank", 1)

    tensor = torch.ones(5) * (rank + 1)
    expected_sum = torch.ones(5) * sum(range(world_size)) + torch.ones(5) * world_size

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_different_dtype_test(rank, world_size, kwargs):
    """Test reduce with different data types."""
    dtype = kwargs["dtype"]
    root_rank = 0

    tensor = torch.ones(10, dtype=dtype) * (rank + 1)
    expected_sum = (
        torch.ones(10, dtype=dtype) * sum(range(world_size))
        + torch.ones(10, dtype=dtype) * world_size
    )

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_different_shape_test(rank, world_size, kwargs):
    """Test reduce with different tensor shapes."""
    shape = kwargs["shape"]
    root_rank = 0

    tensor = torch.ones(shape) * (rank + 1)
    expected_sum = torch.ones(shape) * sum(range(world_size)) + torch.ones(shape) * world_size

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_multiple_tensors_test(rank, world_size, kwargs):
    """Test reduce with multiple tensors."""
    root_rank = 0
    tensors = [torch.ones(5) * (rank + 1) for _ in range(3)]
    expected_sums = [
        torch.ones(5) * sum(range(world_size)) + torch.ones(5) * world_size for _ in range(3)
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]

    for i, tensor_neuron in enumerate(tensors_neuron):
        dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)
        verify_reduce_result(rank, root_rank, tensor_neuron, expected_sums[i], tensors[i])


def run_zero_size_tensor_test(rank, world_size, kwargs):
    """Test reduce with zero-size tensor."""
    root_rank = 0
    tensor = torch.ones(0).to("neuron")
    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)
    assert tensor.size(0) == 0


def run_inplace_operation_test(rank, world_size, kwargs):
    """Test that reduce operates in-place."""
    root_rank = 0
    tensor = torch.ones(10).to("neuron")
    tensor_id = id(tensor)
    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)
    assert id(tensor) == tensor_id  # Should be the same object


def run_async_operation_test(rank, world_size, kwargs):
    """Test asynchronous reduce operation."""
    root_rank = 0
    tensor = torch.ones(10) * (rank + 1)
    expected_sum = torch.ones(10) * sum(range(world_size)) + torch.ones(10) * world_size

    tensor_neuron = tensor.to("neuron")
    work = dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM, async_op=True)
    assert work is not None
    work.wait()

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_group_argument_test(rank, world_size, kwargs):
    """Test reduce with specific process groups."""
    group = dist.new_group([0, 1])
    root_rank = 0

    tensor = torch.ones(10) * (rank + 1)
    expected_sum = torch.ones(10) * sum(range(world_size)) + torch.ones(10) * world_size

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM, group=group)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_invalid_root_rank_test(rank, world_size, kwargs):
    """Test error handling for invalid root rank."""
    tensor = torch.ones(10).to("neuron")
    # This should raise an error due to invalid root rank
    dist.reduce(tensor, dst=world_size + 1, op=dist.ReduceOp.SUM)


def run_large_tensor_test(rank, world_size, kwargs):
    """Test reduce with large tensors."""
    root_rank = 0

    # Create large tensor (1M elements)
    tensor = torch.ones(1000000) * (rank + 1)
    expected_sum = torch.ones(1000000) * sum(range(world_size)) + torch.ones(1000000) * world_size

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_dtype_mismatch_test_at_runtime(rank, world_size, kwargs):
    """Test reduce with mismatched dtypes across ranks (should fail)."""
    root_rank = 0

    # Different ranks use different dtypes
    if rank == 0:
        tensor = torch.ones(10, dtype=torch.float32).to("neuron")
    else:
        tensor = torch.ones(10, dtype=torch.float16).to("neuron")

    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)


def run_shape_mismatch_test_at_runtime(rank, world_size, kwargs):
    """Test reduce with mismatched shapes across ranks (should fail)."""
    root_rank = 0

    # Different ranks use different shapes
    tensor = torch.ones(10).to("neuron") if rank == 0 else torch.ones(5).to("neuron")

    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)


def run_with_inf_inputs_test(rank, world_size, kwargs):
    """Test reduce with tensors containing Inf values."""
    root_rank = 0

    # Create tensor with inf values
    tensor = torch.tensor([torch.inf] * 10).to("neuron")

    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)

    if rank == root_rank:
        # Root rank should have inf values (inf + inf = inf)
        assert torch.isinf(
            tensor.cpu()
        ).all(), f"Root rank {rank}: Expected all inf values, got {tensor.cpu()}"
    else:
        # Non-root ranks should still have inf values
        assert torch.isinf(
            tensor.cpu()
        ).all(), f"Non-root rank {rank}: Expected unchanged inf values, got {tensor.cpu()}"


def run_with_nan_inputs_test(rank, world_size, kwargs):
    """Test reduce with tensors containing NaN values."""
    root_rank = 0

    # Create tensor with nan values
    tensor = torch.tensor([torch.nan] * 10).to("neuron")

    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)

    if rank == root_rank:
        # Root rank should have nan values (nan + nan = nan)
        assert torch.isnan(
            tensor.cpu()
        ).all(), f"Root rank {rank}: Expected all nan values, got {tensor.cpu()}"
    else:
        # Non-root ranks should still have nan values
        assert torch.isnan(
            tensor.cpu()
        ).all(), f"Non-root rank {rank}: Expected unchanged nan values, got {tensor.cpu()}"


def run_with_mixed_special_values_test(rank, world_size, kwargs):
    """Test reduce with tensors containing mixed special values (inf, nan, normal)."""
    root_rank = 0

    # Create tensor with mixed values based on rank
    if rank == 0:
        tensor = torch.tensor([1.0, torch.inf, torch.nan, 2.0, torch.inf]).to("neuron")
    else:
        tensor = torch.tensor([3.0, torch.inf, torch.nan, 4.0, torch.inf]).to("neuron")

    dist.reduce(tensor, dst=root_rank, op=dist.ReduceOp.SUM)

    if rank == root_rank:
        result = tensor.cpu()
        # Check expected results: normal values sum, inf+inf=inf, nan+nan=nan
        assert result[0] == 4.0, f"Expected 4.0, got {result[0]}"  # 1+3=4
        assert torch.isinf(result[1]), f"Expected inf, got {result[1]}"  # inf+inf=inf
        assert torch.isnan(result[2]), f"Expected nan, got {result[2]}"  # nan+nan=nan
        assert result[3] == 6.0, f"Expected 6.0, got {result[3]}"  # 2+4=6
        assert torch.isinf(result[4]), f"Expected inf, got {result[4]}"  # inf+inf=inf
    else:
        # Non-root ranks should have unchanged values
        if rank == 1:
            result = tensor.cpu()
            assert result[0] == 3.0 and torch.isinf(result[1]) and torch.isnan(result[2])


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test reduce with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    root_rank = 0

    # Create large tensor
    tensor = torch.ones(tensor_size, dtype=torch.float32) * (rank + 1)
    expected_sum = (
        torch.ones(tensor_size, dtype=torch.float32) * sum(range(world_size))
        + torch.ones(tensor_size, dtype=torch.float32) * world_size
    )

    tensor_neuron = tensor.to("neuron")
    dist.reduce(tensor_neuron, dst=root_rank, op=dist.ReduceOp.SUM)

    verify_reduce_result(rank, root_rank, tensor_neuron, expected_sum, tensor)


def run_different_device_test(rank, world_size, kwargs):
    """Test reduce with tensors on different devices (should fail)."""
    tensor = torch.ones(10)  # CPU tensor
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)


def run_group_reduce_world_size_4_test(rank, world_size, kwargs):
    """Test reduce within group [2,3] where all group members reduce to rank 3."""
    group: dist.ProcessGroup = dist.new_group([2, 3])

    if rank in [2, 3]:
        # Each rank starts with tensor filled with its rank value
        tensor = torch.ones(10, device="neuron") * rank

        # Reduce to rank 3 (group_dst=1 since rank 3 is index 1 in group [2,3])
        dist.reduce(tensor, group_dst=1, group=group)

        if rank == 3:
            # Rank 3 should have the sum: 2 + 3 = 5
            expected = torch.ones(10) * 5
            assert torch.allclose(tensor.cpu(), expected)
        else:  # rank == 2
            # Rank 2's tensor should remain unchanged
            expected = torch.ones(10) * rank
            assert torch.allclose(tensor.cpu(), expected)

    dist.destroy_process_group(group)


class TestReduce(BaseCollectiveOpTest):
    """Test cases for torch.distributed.reduce."""

    def test_basic_reduce(self):
        """Test basic reduce functionality."""
        self.distributed_tester.run_test(run_basic_reduce_test, root_rank=0)

    @pytest.mark.parametrize("root_rank", [0, 1])
    def test_different_root_ranks(self, root_rank):
        """Test reduce with different root ranks."""
        self.distributed_tester.run_test(run_different_root_test, root_rank=root_rank)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16],
    )
    def test_different_dtypes(self, dtype):
        """Test reduce with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape):
        """Test reduce with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    def test_multiple_tensors(self):
        """Test reduce with multiple tensors."""
        self.distributed_tester.run_test(run_multiple_tensors_test)

    @assert_raises(
        RuntimeError, match=r".*tensors cannot be empty, found empty tensors at indices.*"
    )
    def test_zero_size_tensor(self):
        """Test reduce with zero-size tensor."""
        self.distributed_tester.run_test(run_zero_size_tensor_test)

    def test_inplace_operation(self):
        """Test that reduce operates in-place."""
        self.distributed_tester.run_test(run_inplace_operation_test)

    def test_async_operation(self):
        """Test asynchronous reduce operation."""
        self.distributed_tester.run_test(run_async_operation_test)

    def test_group_operation(self):
        """Test reduce with specific process groups."""
        self.distributed_tester.run_test(run_group_argument_test)

    def test_large_tensor(self):
        """Test reduce with large tensors."""
        self.distributed_tester.run_test(run_large_tensor_test)

    @assert_raises(RuntimeError, match=r".*Invalid rootRank .* for broadcast with world size.*")
    def test_error_handling_invalid_root(self):
        """Test error cases with invalid root rank."""
        self.distributed_tester.run_test(run_invalid_root_rank_test)

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
        reason="Error message for Neuron execution only",
    )
    @assert_raises(
        RuntimeError,
        match=r".*(Failed to execute model|"
        r"NRT Execution error occurred on Neuron for operation=all_reduce).*",
    )
    def test_dtype_mismatch_error(self):
        """Test reduce error handling with dtype mismatch across ranks."""
        self.distributed_tester.run_test(run_dtype_mismatch_test_at_runtime)

    @pytest.mark.skipif(
        os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
        reason="Error message for Neuron execution only",
    )
    @assert_raises(
        RuntimeError,
        match=r".*(Failed to execute model|"
        r"NRT Execution error occurred on Neuron for operation=all_reduce).*",
    )
    def test_shape_mismatch_error(self):
        """Test reduce error handling with shape mismatch across ranks."""
        self.distributed_tester.run_test(run_shape_mismatch_test_at_runtime)

    def test_with_inf_inputs(self):
        """Test reduce with tensors containing Inf values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_with_nan_inputs(self):
        """Test reduce with tensors containing NaN values."""
        self.distributed_tester.run_test(run_with_nan_inputs_test)

    def test_with_mixed_special_values(self):
        """Test reduce with tensors containing mixed special values (inf, nan, normal)."""
        self.distributed_tester.run_test(run_with_mixed_special_values_test)

    @assert_raises(RuntimeError, match=r".*Expected neuron device, got cpu.*")
    def test_different_devices(self):
        """Test reduce with tensors on different devices."""
        self.distributed_tester.run_test(run_different_device_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test reduce with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)


class TestReduceTensorWorldSize4(BaseCollectiveOpTest):
    """Test cases for torch.distributed.scatter with a world size of 4."""

    world_size = 4  # Set at class level instead of in setup_method

    def test_group_reduce_with_rank_data(self):
        """Test scatter within group [2,3] with group_src=1 (global rank 3)."""
        self.distributed_tester.run_test(run_group_reduce_world_size_4_test)
