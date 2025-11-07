import os

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest


def run_different_dtype_test(rank, world_size, kwargs):
    input_tensor = torch.ones(20, dtype=kwargs["dtype"]) * (rank + 1)  # 10 per rank
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, dtype=kwargs["dtype"], device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.SUM)

    # Each rank gets sum of corresponding portions
    expected = torch.ones(10, dtype=kwargs["dtype"]) * (1 + 2)
    assert torch.allclose(output_tensor.cpu(), expected)


def run_different_shape_test(rank, world_size, kwargs):
    shape = kwargs["shape"]
    input_shape = [*list(shape[:-1]), shape[-1] * world_size]
    input_tensor = torch.ones(input_shape) * (rank + 1)
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(shape, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.SUM)

    expected = torch.ones(shape) * (1 + 2)  # Sum of values from rank 0 and 1
    assert torch.allclose(output_tensor.cpu(), expected)


def run_large_tensor_test(rank, world_size, kwargs):
    size_per_rank = 1000000  # 1M elements per rank
    input_tensor = torch.ones(size_per_rank * world_size) * (rank + 1)
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(size_per_rank, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.SUM)

    expected = torch.ones(size_per_rank) * (1 + 2)  # Sum of values from rank 0 and 1
    assert torch.allclose(output_tensor.cpu(), expected)


def run_avg_reduce_test(rank, world_size, kwargs):
    # Test AVG with different values per rank
    input_tensor = torch.full((20,), rank * 2 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 3
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.AVG)

    # Average: (1 + 3) / 2 = 2
    expected = torch.full((10,), 2.0)
    assert torch.allclose(output_tensor.cpu(), expected)


def run_min_reduce_test(rank, world_size, kwargs):
    # Test MIN with different values per rank
    input_tensor = torch.full((20,), rank * 2 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 3
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.MIN)

    # Minimum: min(1, 3) = 1
    expected = torch.full((10,), 1.0)
    assert torch.allclose(output_tensor.cpu(), expected)


def run_max_reduce_test(rank, world_size, kwargs):
    # Test MAX with different values per rank
    input_tensor = torch.full((20,), rank * 2 + 1, dtype=torch.float32)  # rank 0: 1, rank 1: 3
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.MAX)

    # Maximum: max(1, 3) = 3
    expected = torch.full((10,), 3.0)
    assert torch.allclose(output_tensor.cpu(), expected)


def run_unsupported_op_test(rank, world_size, kwargs):
    input_tensor = torch.ones(20).to("neuron")
    output_tensor = torch.zeros(10, device="neuron")
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.PRODUCT)


def run_with_inf_inputs_test(rank, world_size, kwargs):
    input_tensor = torch.tensor([float("inf")] * 20).to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    assert torch.isinf(output_tensor).all()


def run_async_operation_test(rank, world_size, kwargs):
    input_tensor = torch.ones(20).to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    work = dist.reduce_scatter_tensor(
        output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
    )
    assert work is not None
    work.wait()

    expected = torch.ones(10) * 2
    assert torch.allclose(output_tensor.cpu(), expected)


def run_group_argument_test(rank, world_size, kwargs):
    group = dist.new_group([0, 1])
    if rank <= 1:
        input_tensor = torch.ones(20).to("neuron")  # 10 per rank
        output_tensor = torch.zeros(10, device="neuron")

        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=group)
        expected = torch.ones(10) * 2
        assert torch.allclose(output_tensor.cpu(), expected)


def run_size_mismatch_test(rank, world_size, kwargs):
    # Input size not divisible by world_size
    input_tensor = torch.ones(21).to("neuron")  # 21 is not divisible by world_size
    output_tensor = torch.zeros(10, device="neuron")
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)


def run_output_size_mismatch_test(rank, world_size, kwargs):
    # Output size doesn't match input_size/world_size
    input_tensor = torch.ones(20).to("neuron")
    output_tensor = torch.zeros(5, device="neuron")  # Should be 10
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)


def run_reduce_scatter_partial_group_test(rank, world_size, kwargs):
    """Test reduce_scatter with a partial group in a larger world.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        kwargs: Dictionary containing:
            - partial_group: List of ranks in the partial group
            - expected_output: Expected output tensor for each rank in the group
            - input_shape: Shape of input tensor per rank (default: 4)
            - dtype: Tensor dtype (default: torch.long)
    """
    partial_group = kwargs.get("partial_group", [0, 1])
    expected_output = kwargs.get("expected_output")
    dtype = torch.long
    input_shape_per_rank = 4

    # Create the partial group
    group = dist.new_group(partial_group)

    if rank in partial_group:
        group_size = len(partial_group)

        # Create input tensor: total size = input_shape_per_rank * group_size
        # Each rank contributes different values
        input_tensor = torch.tensor(
            [rank * 10 + i + 1 for i in range(input_shape_per_rank * group_size)],
            dtype=dtype,
        ).to("neuron")

        # Create output tensor to receive the scattered result
        output_tensor = torch.zeros(input_shape_per_rank, dtype=dtype, device="neuron")

        # Perform reduce_scatter within the partial group
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=group)

        # Verify results if expected output is provided
        rank_idx = partial_group.index(rank)
        expected = expected_output[rank_idx]
        assert torch.allclose(output_tensor.cpu(), expected), (
            f"Rank {rank}: Output {output_tensor.cpu()} doesn't match " f"expected {expected}"
        )
    else:
        # Ranks not in partial_group don't participate
        pass


class TestReduceScatterTensor(BaseCollectiveOpTest):
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.int8,
        ],
    )
    def test_different_dtypes(self, dtype):
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape):
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    def test_large_tensor(self):
        self.distributed_tester.run_test(run_large_tensor_test)

    def test_avg_reduce(self):
        self.distributed_tester.run_test(run_avg_reduce_test)

    def test_min_reduce(self):
        self.distributed_tester.run_test(run_min_reduce_test)

    def test_max_reduce(self):
        self.distributed_tester.run_test(run_max_reduce_test)

    def test_unsupported_op(self):
        with pytest.raises(
            RuntimeError,
            match=r".*Operation reduce_scatter is not implemented for Neuron*",
        ):
            self.distributed_tester.run_test(run_unsupported_op_test)

    # @pytest.mark.xfail(reason="Inf is not handled correctly by the compiler")
    def test_with_inf_inputs(self):
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_async_operation(self):
        self.distributed_tester.run_test(run_async_operation_test)

    def test_group_operation(self):
        self.distributed_tester.run_test(run_group_argument_test)

    def test_size_validation(self):
        with pytest.raises(RuntimeError, match=r".*Operation reduce_scatter is not implemented.*"):
            self.distributed_tester.run_test(run_size_mismatch_test)
        with pytest.raises(RuntimeError, match=r".*Operation reduce_scatter is not implemented.*"):
            self.distributed_tester.run_test(run_output_size_mismatch_test)


class TestReduceScatterTensorWorldSize4(BaseCollectiveOpTest):
    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.parametrize(
        "group_config",
        [
            {
                "partial_group": [0, 1],
                "expected_output": [
                    torch.tensor([12, 14, 16, 18]),
                    torch.tensor([20, 22, 24, 26]),
                ],
            },
        ],
    )
    def test_reduce_scatter_partial_group_test(self, group_config):
        partial_group = group_config["partial_group"]
        expected_output = group_config["expected_output"]
        self.distributed_tester.run_test(
            run_reduce_scatter_partial_group_test,
            partial_group=partial_group,
            expected_output=expected_output,
        )
