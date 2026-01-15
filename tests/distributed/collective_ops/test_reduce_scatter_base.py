import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

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


def run_premul_sum_scalar_test(rank, world_size, kwargs):
    # Test PREMUL_SUM with scalar factor
    input_tensor = torch.ones(20, dtype=torch.float32) * (rank + 1)  # rank 0: 1, rank 1: 2
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    premul_factor = 2.5
    reduce_op = dist._make_nccl_premul_sum(premul_factor)

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=reduce_op)

    # PREMUL_SUM: (1 * 2.5) + (2 * 2.5) = 2.5 + 5.0 = 7.5
    expected = torch.full((10,), 7.5)
    assert torch.allclose(output_tensor.cpu(), expected)


def run_premul_sum_tensor_test(rank, world_size, kwargs):
    # Test PREMUL_SUM with tensor factor
    input_tensor = torch.ones(20, dtype=torch.float32) * (rank + 1)  # rank 0: 1, rank 1: 2
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(10, device="neuron")

    tensor_factor = torch.tensor([2.5], dtype=torch.float32).to("neuron")
    reduce_op = dist._make_nccl_premul_sum(tensor_factor)

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=reduce_op)

    # PREMUL_SUM: (1 * 2.5) + (2 * 2.5) = 2.5 + 5.0 = 7.5
    expected = torch.full((10,), 7.5)
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


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test reduce_scatter_tensor with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    size_per_rank = tensor_size // world_size

    input_tensor = torch.ones(tensor_size, dtype=torch.float32) * (rank + 1)
    input_tensor_neuron = input_tensor.to("neuron")
    output_tensor = torch.zeros(size_per_rank, dtype=torch.float32, device="neuron")

    dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.SUM)

    expected = torch.ones(size_per_rank, dtype=torch.float32) * (
        1 + 2
    )  # Sum of values from rank 0 and 1
    assert torch.allclose(output_tensor.cpu(), expected)


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
        assert torch.allclose(
            output_tensor.cpu(), expected
        ), f"Rank {rank}: Output {output_tensor.cpu()} doesn't match expected {expected}"
    else:
        # Ranks not in partial_group don't participate
        pass


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test reduce_scatter_tensor with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    with open(f"./neuron_core_id_{rank}.txt", "w") as f:
        f.write(str(core_id))
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/"
        "model_shared_scratchpad/peak"
    )
    for i in range(2):
        tensor_size = 134217732 + i * 134217732
        size_per_rank = tensor_size // world_size
        input_tensor = torch.ones(tensor_size, dtype=torch.float32) * (rank + 1)
        input_tensor_neuron = input_tensor.to("neuron")
        output_tensor = torch.zeros(size_per_rank, dtype=torch.float32, device="neuron")

        dist.reduce_scatter_tensor(output_tensor, input_tensor_neuron, op=dist.ReduceOp.SUM)

        expected = torch.ones(size_per_rank, dtype=torch.float32) * (1 + 2)
        assert torch.allclose(output_tensor.cpu(), expected)

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/"
                "model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak <= 1073741824, f"Memory {peak} exceeds 1GB"


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

    def test_premul_sum_scalar(self):
        self.distributed_tester.run_test(run_premul_sum_scalar_test)

    def test_premul_sum_tensor(self):
        self.distributed_tester.run_test(run_premul_sum_tensor_test)

    @assert_raises(RuntimeError, match=r".*unsupported reduce operation.*supported ops.*")
    def test_unsupported_op(self):
        self.distributed_tester.run_test(run_unsupported_op_test)

    # @pytest.mark.xfail(reason="Inf is not handled correctly by the compiler")
    def test_with_inf_inputs(self):
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_async_operation(self):
        self.distributed_tester.run_test(run_async_operation_test)

    def test_group_operation(self):
        self.distributed_tester.run_test(run_group_argument_test)

    @assert_raises(
        RuntimeError,
        match=(
            r".*scatter dimension.*input size.*"
            r"must be evenly divisible by replica group size.*Expected output size.*"
            r"actual output size.*"
        ),
    )
    def test_input_size_validation(self):
        self.distributed_tester.run_test(run_size_mismatch_test)

    @assert_raises(
        RuntimeError,
        match=(
            r".*input size.*must be evenly "
            r"divisible by replica group size.*Expected output size.*"
            r"actual output size.*"
        ),
    )
    def test_output_size_validation(self):
        self.distributed_tester.run_test(run_output_size_mismatch_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test reduce_scatter_tensor with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test reduce_scatter_tensor with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)


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
