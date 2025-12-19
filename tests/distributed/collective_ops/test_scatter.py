import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_basic_scatter_test(rank, world_size, kwargs):
    """Basic scatter operation test."""
    # Root process creates input tensors for all processes
    if rank == 0:
        input_tensors = [torch.ones(10, device="neuron") * i for i in range(world_size)]
    else:
        input_tensors = None

    # Output tensor for receiving scattered data
    output = torch.zeros(10, device="neuron")

    # Scatter from rank 0
    dist.scatter(output, input_tensors, src=0)

    # Each rank should receive its corresponding tensor
    expected = torch.ones(10) * rank
    assert torch.allclose(output.cpu(), expected)


def run_different_dtype_test(rank, world_size, kwargs):
    """Test scatter with different data types."""
    dtype = kwargs["dtype"]

    if rank == 0:
        input_tensors = [
            torch.ones(10, dtype=dtype, device="neuron") * i for i in range(world_size)
        ]
    else:
        input_tensors = None

    output = torch.zeros(10, dtype=dtype, device="neuron")
    dist.scatter(output, input_tensors, src=0)

    expected = torch.ones(10, dtype=dtype) * rank
    assert torch.allclose(output.cpu(), expected)


def run_different_shape_test(rank, world_size, kwargs):
    """Test scatter with different tensor shapes."""
    shape = kwargs["shape"]

    if rank == 0:
        input_tensors = [torch.ones(shape, device="neuron") * i for i in range(world_size)]
    else:
        input_tensors = None

    output = torch.zeros(shape, device="neuron")
    dist.scatter(output, input_tensors, src=0)

    expected = torch.ones(shape) * rank
    assert torch.allclose(output.cpu(), expected)


def run_multiple_tensors_test(rank, world_size, kwargs):
    """Test scatter with multiple tensors."""
    if rank == 0:
        input_tensors_list = [
            [torch.ones(10, device="neuron") * (i + j) for i in range(world_size)] for j in range(3)
        ]
    else:
        input_tensors_list = None

    output_tensors = [torch.zeros(10, device="neuron") for _ in range(3)]

    for i, output in enumerate(output_tensors):
        dist.scatter(output, None if rank != 0 else input_tensors_list[i], src=0)
        expected = torch.ones(10) * (rank + i)
        assert torch.allclose(output.cpu(), expected)


def run_different_root_test(rank, world_size, kwargs):
    """Test scatter from different root ranks."""
    root_rank = kwargs["root_rank"]

    if rank == root_rank:
        input_tensors = [torch.ones(10, device="neuron") * i for i in range(world_size)]
    else:
        input_tensors = None

    output = torch.zeros(10, device="neuron")
    dist.scatter(output, input_tensors, src=root_rank)

    expected = torch.ones(10) * rank
    assert torch.allclose(output.cpu(), expected)


def run_async_operation_test(rank, world_size, kwargs):
    """Test asynchronous scatter operation."""
    if rank == 0:
        input_tensors = [torch.ones(10, device="neuron") * i for i in range(world_size)]
    else:
        input_tensors = None

    output = torch.zeros(10, device="neuron")

    work = dist.scatter(output, input_tensors, src=0, async_op=True)
    assert work is not None
    work.wait()

    expected = torch.ones(10) * rank
    assert torch.allclose(output.cpu(), expected)


def run_mismatched_shapes_test(rank, world_size, kwargs):
    if rank == 0:
        input_tensors = [torch.ones(10 + i, device="neuron") for i in range(world_size)]
    else:
        input_tensors = None
    output = torch.zeros(10, device="neuron")
    dist.scatter(output, input_tensors, src=0)


def run_invalid_size_test(rank, world_size, kwargs):
    if rank == 0:
        # Create wrong number of input tensors
        input_tensors = [torch.ones(10, device="neuron") for _ in range(world_size + 1)]
    else:
        input_tensors = None
    output = torch.zeros(10, device="neuron")
    dist.scatter(output, input_tensors, src=0)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test scatter with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Root process creates input tensors for all processes
    if rank == 0:
        input_tensors = [
            torch.ones(tensor_size, device="neuron", dtype=torch.float32) * i
            for i in range(world_size)
        ]
    else:
        input_tensors = None

    # Output tensor for receiving scattered data
    output = torch.zeros(tensor_size, device="neuron", dtype=torch.float32)

    # Scatter from rank 0
    dist.scatter(output, input_tensors, src=0)

    # Each rank should receive its corresponding tensor
    expected = torch.ones(tensor_size, dtype=torch.float32) * rank
    assert torch.allclose(output.cpu(), expected)


def run_group_scatter_world_size_4_test(rank, world_size, kwargs):
    """Test scatter within group [2,3] where rank 3 scatters data to group members."""
    group: dist.ProcessGroup = dist.new_group([2, 3])

    if rank in [2, 3]:
        # Root process (rank 3) creates input tensors for group processes
        input_tensors = [torch.ones(10, device="neuron") * i for i in [2, 3]] if rank == 3 else None

        # Output tensor for receiving scattered data
        output = torch.zeros(10, device="neuron")

        # Scatter from rank 3 (group_src=1 since rank 3 is index 1 in group [2,3])
        dist.scatter(output, input_tensors, group_src=1, group=group)

        # Each rank should receive its corresponding tensor
        expected = torch.ones(10) * rank
        assert torch.allclose(output.cpu(), expected)

    dist.destroy_process_group(group)


class TestScatter(BaseCollectiveOpTest):
    """Test cases for torch.distributed.scatter."""

    def test_basic_scatter(self):
        """Test basic scatter operation."""
        self.distributed_tester.run_test(run_basic_scatter_test)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16]
    )
    def test_different_dtypes(self, dtype):
        """Test scatter with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape):
        """Test scatter with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    def test_multiple_tensors(self):
        """Test scatter with multiple tensors."""
        self.distributed_tester.run_test(run_multiple_tensors_test)

    @pytest.mark.parametrize("root_rank", [0, 1])
    def test_different_root(self, root_rank):
        """Test scatter from different root ranks."""
        self.distributed_tester.run_test(run_different_root_test, root_rank=root_rank)

    def test_async_operation(self):
        """Test asynchronous scatter operation."""
        self.distributed_tester.run_test(run_async_operation_test)

    @assert_raises(RuntimeError)
    def test_invalid_input_size(self):
        """Test scatter with invalid input size."""
        self.distributed_tester.run_test(run_invalid_size_test)

    @assert_raises(RuntimeError)
    def test_invalid_root(self):
        """Test scatter with invalid root rank."""
        self.distributed_tester.run_test(run_different_root_test, root_rank=self.world_size + 1)

    @assert_raises(RuntimeError)
    def test_mismatched_shapes(self):
        """Test scatter with mismatched tensor shapes."""
        self.distributed_tester.run_test(run_mismatched_shapes_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test scatter with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)


class TestScatterTensorWorldSize4(BaseCollectiveOpTest):
    """Test cases for torch.distributed.scatter with a world size of 4."""

    world_size = 4  # Set at class level instead of in setup_method

    def test_group_scatter_with_rank_data(self):
        """Test scatter within group [2,3] with group_src=1 (global rank 3)."""
        self.distributed_tester.run_test(run_group_scatter_world_size_4_test)
