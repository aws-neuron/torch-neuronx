import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_different_dtype_test(rank, world_size, kwargs):
    tensor = torch.ones(10, dtype=kwargs["dtype"])
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor_neuron.cpu(), expected * 2)  # * 2 because of 2 processes


def run_different_shape_test(rank, world_size, kwargs):
    tensor = torch.ones(kwargs["shape"])
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor_neuron.cpu(), expected * 2)


def run_multiple_tensors_test(rank, world_size, kwargs):
    tensors = [torch.ones(10) for _ in range(3)]
    expected = [tensor.clone() for tensor in tensors]
    tensors_neuron = [tensor.to("neuron") for tensor in tensors]

    for t in tensors_neuron:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e * 2)


def run_inplace_operation_test(rank, world_size, kwargs):
    tensor = torch.ones(10).to("neuron")
    tensor_id = id(tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert id(tensor) == tensor_id  # Should be the same object


def run_min_reduce_test(rank, world_size, kwargs):
    # Test MIN with different values per rank
    tensor = torch.full((10,), rank * 3 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 4
    expected = torch.full((10,), 1.0, dtype=torch.float32)  # Minimum: 1
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.MIN)
    assert torch.allclose(tensor_neuron.cpu(), expected)


def run_max_reduce_test(rank, world_size, kwargs):
    # Test MAX with different values per rank
    tensor = torch.full((10,), rank * 3 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 4
    expected = torch.full((10,), 4.0, dtype=torch.float32)  # Maximum: 4
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.MAX)
    assert torch.allclose(tensor_neuron.cpu(), expected)


def run_avg_reduce_test(rank, world_size, kwargs):
    # Test AVG with different values per rank
    tensor = torch.full((10,), rank * 3 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 4
    expected = torch.full((10,), (1 + 4) / 2)  # Average: 2.5
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.AVG)
    assert torch.allclose(tensor_neuron.cpu(), expected)


def run_avg_reduce_integer_fractional_test(rank, world_size, kwargs):
    # Test case that exposes integer truncation
    # rank 0: [1], rank 1: [2] -> average should be 1 as input is int32, same behavior on CUDA
    tensor = torch.tensor([rank + 1], dtype=torch.int32)  # rank 0: 1, rank 1: 2
    tensor_neuron = tensor.to("neuron")

    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.AVG)
    result = tensor_neuron.cpu()

    # CUDA Behavior: (1 + 2) / 2 = 1
    expected = 1
    assert result.item() == expected, f"Expected {expected}, got {result.item()}"


def run_zero_size_tensor_test(rank, world_size, kwargs):
    tensor = torch.ones(0).to("neuron")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert tensor.size(0) == 0


def run_large_tensor_test(rank, world_size, kwargs):
    tensor = torch.ones(1000000)  # 1M elements
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor_neuron.to("cpu"), expected * 2)


def run_invalid_op_test(rank, world_size, kwargs):
    tensor = torch.ones(10).to("neuron")
    dist.all_reduce(tensor, op="invalid_op")


def run_with_inf_inputs_test(rank, world_size, kwargs):
    tensor = torch.tensor([torch.inf]).clone().to("neuron")
    expected = tensor.cpu().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor.cpu(), expected * 2)


def run_async_operation_test(rank, world_size, kwargs):
    tensor = torch.ones(10)
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")

    work = dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM, async_op=True)
    assert work is not None
    work.wait()  # Wait for completion
    assert torch.allclose(tensor_neuron.cpu(), expected * 2)


def run_group_argument_test(rank, world_size, kwargs):
    group = dist.new_group([0, 1])
    tensor = torch.ones(10)
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")

    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM, group=group)
    assert torch.allclose(tensor_neuron.cpu(), expected * 2)


def run_different_device_test(rank, world_size, kwargs):
    tensor = torch.ones(10)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_reduce with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    tensor = torch.ones(tensor_size, dtype=torch.float32)
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")
    dist.all_reduce(tensor_neuron, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor_neuron.cpu(), expected * 2)


def run_premul_sum_scalar_all_reduce_test(rank, world_size, kwargs):
    """Test all_reduce with PREMUL_SUM operation using scalar factor."""
    # rank 0: [1, 2, 3, 4], rank 1: [3, 4, 5, 6]
    tensor = torch.tensor(
        [rank * 2 + 1.0, rank * 2 + 2.0, rank * 2 + 3.0, rank * 2 + 4.0], dtype=torch.float32
    ).to("neuron")

    # Create PREMUL_SUM operation with scalar factor 2.5
    premul_factor = 2.5
    reduce_op = dist._make_nccl_premul_sum(premul_factor)

    # For PREMUL_SUM with factor 2.5:
    # rank 0: [1, 2, 3, 4] * 2.5 = [2.5, 5.0, 7.5, 10.0]
    # rank 1: [3, 4, 5, 6] * 2.5 = [7.5, 10.0, 12.5, 15.0]
    # After SUM across all ranks:
    #   All ranks get: [2.5+7.5, 5.0+10.0, 7.5+12.5, 10.0+15.0] = [10.0, 15.0, 20.0, 25.0]
    expected = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float32)

    dist.all_reduce(tensor, op=reduce_op)
    assert torch.allclose(
        tensor.cpu(), expected, rtol=1e-5
    ), f"Rank {rank}: Expected {expected}, got {tensor.cpu()}"


def run_premul_sum_tensor_all_reduce_test(rank, world_size, kwargs):
    """Test all_reduce with PREMUL_SUM operation using tensor factor."""
    # rank 0: [1, 2, 3, 4], rank 1: [3, 4, 5, 6]
    tensor = torch.tensor(
        [rank * 2 + 1.0, rank * 2 + 2.0, rank * 2 + 3.0, rank * 2 + 4.0], dtype=torch.float32
    ).to("neuron")

    # Create PREMUL_SUM operation with tensor factor 2.5
    tensor_factor = torch.tensor([2.5], dtype=torch.float32).to("neuron")
    reduce_op = dist._make_nccl_premul_sum(tensor_factor)

    # For PREMUL_SUM with factor 2.5:
    # rank 0: [1, 2, 3, 4] * 2.5 = [2.5, 5.0, 7.5, 10.0]
    # rank 1: [3, 4, 5, 6] * 2.5 = [7.5, 10.0, 12.5, 15.0]
    # After SUM across all ranks:
    #   All ranks get: [2.5+7.5, 5.0+10.0, 7.5+12.5, 10.0+15.0] = [10.0, 15.0, 20.0, 25.0]
    expected = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float32)

    dist.all_reduce(tensor, op=reduce_op)
    assert torch.allclose(
        tensor.cpu(), expected, rtol=1e-5
    ), f"Rank {rank}: Expected {expected}, got {tensor.cpu()}"


def run_all_reduce_partial_group_test(rank, world_size, kwargs):
    """Test all_reduce with a partial group in a larger world.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        kwargs: Dictionary containing:
            - partial_group: List of ranks in the partial group
            - expected_output: Expected output tensor for each rank in the group
            - input_shape: Shape of input tensor (default: 4)
            - dtype: Tensor dtype (default: torch.long)
    """
    partial_group = kwargs.get("partial_group", [0, 1])
    expected_output = kwargs.get("expected_output")
    dtype = torch.long
    input_shape = 4

    # Create the partial group
    group = dist.new_group(partial_group)

    if rank in partial_group:
        # Create input tensor with rank-specific values
        input_tensor = torch.tensor(
            [rank * 10 + i + 1 for i in range(input_shape)],
            dtype=dtype,
        ).to("neuron")

        # Perform all_reduce within the partial group
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=group)

        # Verify results if expected output is provided
        rank_idx = partial_group.index(rank)
        expected = expected_output[rank_idx]
        assert torch.allclose(
            input_tensor.cpu(), expected
        ), f"Rank {rank}: Output {input_tensor.cpu()} doesn't match expected {expected}"
    else:
        # Ranks not in partial_group don't participate
        pass
    torch.distributed.barrier()


class TestAllReduce(BaseCollectiveOpTest):
    """Test cases for torch.distributed.all_reduce."""

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16, torch.int8],
    )
    def test_different_dtypes(self, dtype):
        """Test all_reduce with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape):
        """Test all_reduce with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    def test_multiple_tensors(self):
        """Test all_reduce with multiple tensors."""
        self.distributed_tester.run_test(run_multiple_tensors_test)

    def test_inplace_operation(self):
        """Test that all_reduce operates in-place."""
        self.distributed_tester.run_test(run_inplace_operation_test)

    def test_avg_reduce(self):
        """Test all_reduce with AVG operation."""
        self.distributed_tester.run_test(run_avg_reduce_test)

    def test_avg_reduce_integer_fractional(self):
        """Test all_reduce with AVG operation that should produce fractional results."""
        self.distributed_tester.run_test(run_avg_reduce_integer_fractional_test)

    def test_min_reduce(self):
        """Test all_reduce with MIN operation."""
        self.distributed_tester.run_test(run_min_reduce_test)

    def test_max_reduce(self):
        """Test all_reduce with MAX operation."""
        self.distributed_tester.run_test(run_max_reduce_test)

    @assert_raises(
        RuntimeError, match=r".*tensors cannot be empty, found empty tensors at indices.*"
    )
    def test_zero_size_tensor(self):
        """Test all_reduce with zero-size tensor."""
        self.distributed_tester.run_test(run_zero_size_tensor_test)

    def test_large_tensor(self):
        """Test all_reduce with a large tensor."""
        self.distributed_tester.run_test(run_large_tensor_test)

    @assert_raises(RuntimeError, match=r".*incompatible function arguments.*")
    def test_error_handling(self):
        """Test error cases. Error thrown by torch directly"""
        self.distributed_tester.run_test(run_invalid_op_test)

    @pytest.mark.xfail(reason="Inf is not handled correctly by the compiler")
    def test_with_inf_inputs(self, values):
        """Test all_reduce with specific input values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_async_operation(self):
        """Test asynchronous all_reduce operation."""
        self.distributed_tester.run_test(run_async_operation_test)

    def test_group_operation(self):
        """Test all_reduce with specific process groups."""
        self.distributed_tester.run_test(run_group_argument_test)

    @assert_raises(RuntimeError, match=r".*Expected neuron device, got cpu.*")
    def test_different_devices(self):
        """Test all_reduce with tensors on different devices."""
        self.distributed_tester.run_test(run_different_device_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test all_reduce with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    def test_all_reduce_premul_sum_scalar(self):
        """Test all_reduce with PREMUL_SUM operation using scalar factor."""
        self.distributed_tester.run_test(run_premul_sum_scalar_all_reduce_test)

    def test_all_reduce_premul_sum_tensor(self):
        """Test all_reduce with PREMUL_SUM operation using tensor factor."""
        self.distributed_tester.run_test(run_premul_sum_tensor_all_reduce_test)


class TestAllReduceWorldSize4(BaseCollectiveOpTest):
    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.parametrize(
        "group_config",
        [
            {
                "partial_group": [0, 1],
                "expected_output": [
                    torch.tensor([12, 14, 16, 18]),  # sum of rank 0 and rank 1 values
                    torch.tensor([12, 14, 16, 18]),  # same result for both ranks
                ],
            },
        ],
    )
    def test_all_reduce_partial_group_test(self, group_config):
        partial_group = group_config["partial_group"]
        expected_output = group_config["expected_output"]
        self.distributed_tester.run_test(
            run_all_reduce_partial_group_test,
            partial_group=partial_group,
            expected_output=expected_output,
        )
