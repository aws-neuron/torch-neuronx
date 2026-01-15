import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_all_gather_into_tensor_dtype_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with different dtypes."""
    input_tensor = torch.ones(10, dtype=kwargs["dtype"]) * (rank + 1)
    input_tensor = input_tensor.to("neuron")

    # Create output tensor with correct size (input_size * world_size)
    output_tensor = torch.zeros(world_size * 10, dtype=kwargs["dtype"]).to("neuron")

    # Perform all_gather_into_tensor
    dist.all_gather_into_tensor(output_tensor, input_tensor)

    # Expected: concatenated tensor with values [1,1,...,2,2,...,n,n,...]
    expected = torch.cat(
        [torch.ones(10, dtype=kwargs["dtype"]) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(output_tensor.cpu(), expected)


def run_all_gather_into_tensor_shape_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with different shapes."""
    shape = kwargs["shape"]
    input_tensor = torch.ones(shape) * (rank + 1)
    input_tensor = input_tensor.to("neuron")

    # Calculate output shape (first dimension multiplied by world_size)
    output_shape = list(shape)
    output_shape[0] = output_shape[0] * world_size
    output_tensor = torch.zeros(output_shape).to("neuron")

    dist.all_gather_into_tensor(output_tensor, input_tensor)

    expected = torch.cat([torch.ones(shape) * (i + 1) for i in range(world_size)])
    assert torch.allclose(output_tensor.cpu(), expected)


@pytest.mark.xfail(reason="Test has been flakey, needs more investigation. Xfailing to unblock CI")
def run_all_gather_into_tensor_async_test(rank, world_size, kwargs):
    """Test async operation of all_gather_into_tensor."""
    input_tensor = torch.ones(10) * (rank + 1)
    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros(world_size * 10).to("neuron")

    work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
    assert work is not None
    work.wait()

    expected = torch.cat([torch.ones(10) * (i + 1) for i in range(world_size)])
    assert torch.allclose(output_tensor.cpu(), expected)


def run_all_gather_into_tensor_group_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with process groups."""
    group = dist.new_group([0, 1])
    input_tensor = torch.ones(10) * (rank + 1)
    input_tensor = input_tensor.to("neuron")

    if rank <= 1:  # Only ranks 0 and 1 are in the group
        output_tensor = torch.zeros(20).to("neuron")  # size 2 * 10 for group [0, 1]
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=group)

        expected = torch.cat([torch.ones(10) * (i + 1) for i in range(2)])
        assert torch.allclose(output_tensor.cpu(), expected)
    else:
        raise AssertionError("Ranks greater than 2, only need 2 ranks for this test")


def run_all_gather_into_tensor_wrong_size_test(rank, world_size, kwargs):
    """Test error handling for incorrect output tensor size."""
    input_tensor = torch.ones(10).to("neuron")
    # Wrong size output tensor (should be world_size * 10)
    output_tensor = torch.zeros(15).to("neuron")
    dist.all_gather_into_tensor(output_tensor, input_tensor)


def run_all_gather_negative_values_test(rank, world_size, kwargs):
    """Test all_gather with zero and negative values on neuron device."""
    # Create input tensor with zero and negative values specific to each rank
    tensor = torch.tensor([0, -1, -2, -3]) if rank == 0 else torch.tensor([-4, 0, -5, -6])

    tensor_neuron = tensor.to("neuron")

    # Create list of output tensors, one for each process
    output_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

    # Perform all_gather
    dist.all_gather(output_list, tensor_neuron)

    # Expected values for each rank
    expected_list = [
        torch.tensor([0, -1, -2, -3]),  # rank 0's values
        torch.tensor([-4, 0, -5, -6]),  # rank 1's values
    ]

    # Verify each gathered tensor matches expected
    for output, expected in zip(output_list, expected_list, strict=False):
        assert torch.allclose(output.cpu(), expected)


def run_all_gather_base_debug_test(rank, world_size, kwargs):
    """Test _allgather_base debug output and shape handling."""
    input_tensor = torch.ones(5).to("neuron")
    output_tensor = torch.zeros(1, 10).to("neuron")

    dist.all_gather_into_tensor(output_tensor, input_tensor)

    expected = torch.ones(1, 10)
    assert torch.allclose(output_tensor.cpu(), expected)
    assert input_tensor.shape == torch.Size(
        [5]
    ), f"Input tensor shape mismatch: Expected {torch.Size([5])}, got {input_tensor.shape}"
    assert output_tensor.shape == torch.Size(
        [1, 10]
    ), f"Output tensor shape mismatch: Expected {torch.Size([1, 10])}, got {output_tensor.shape}"


def run_all_gather_base_reverse_test(rank, world_size, kwargs):
    """Test _allgather_base with input having more dims than output."""
    input_tensor = torch.ones(1, 5).to("neuron")
    output_tensor = torch.zeros(10).to("neuron")

    dist.all_gather_into_tensor(output_tensor, input_tensor)
    expected = torch.ones(10)
    assert torch.allclose(output_tensor.cpu(), expected)
    assert input_tensor.shape == torch.Size(
        [1, 5]
    ), f"Input tensor shape mismatch: Expected {torch.Size([1, 5])}, got {input_tensor.shape}"
    assert output_tensor.shape == torch.Size(
        [10]
    ), f"Output tensor shape mismatch: Expected {torch.Size([10])}, got {output_tensor.shape}"


def run_all_gather_stack_outputs_test(rank, world_size, kwargs):
    tensor_in = torch.arange(2, dtype=torch.int64, device="neuron") + 1 + 2 * rank
    tensor_out = torch.zeros(world_size, 2, dtype=torch.int64, device="neuron")
    dist.all_gather_into_tensor(tensor_out, tensor_in)

    expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert torch.allclose(tensor_out.cpu(), expected)


def run_all_gather_multiple_single_dim_test(rank, world_size, kwargs):
    input_tensor = torch.ones((1, 1, 10)) * (rank + 1)
    input_tensor = input_tensor.to("neuron")
    output_tensor = torch.zeros((1, world_size * 10)).to("neuron")
    work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
    assert work is not None
    work.wait()

    expected = torch.cat([torch.ones(10) * (i + 1) for i in range(world_size)])
    assert torch.allclose(output_tensor.cpu(), expected)
    assert input_tensor.shape == torch.Size(
        [1, 1, 10]
    ), f"Input tensor shape mismatch: Expected {torch.Size([1, 1, 10])}, got {input_tensor.shape}"
    assert output_tensor.shape == torch.Size([1, world_size * 10]), (
        f"Output tensor shape mismatch: Expected {torch.Size([1, world_size * 10])}, "
        f"got {output_tensor.shape}"
    )


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    input_tensor = torch.ones(tensor_size, dtype=torch.float32) * (rank + 1)
    input_tensor = input_tensor.to("neuron")

    # Create output tensor with correct size (input_size * world_size)
    output_tensor = torch.zeros(world_size * tensor_size, dtype=torch.float32).to("neuron")

    # Perform all_gather_into_tensor
    dist.all_gather_into_tensor(output_tensor, input_tensor)

    # Expected: concatenated tensor with values [1,1,...,2,2,...,n,n,...]
    expected = torch.cat(
        [torch.ones(tensor_size, dtype=torch.float32) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(output_tensor.cpu(), expected)


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        tensor = torch.ones(tensor_size, dtype=torch.float32)
        tensor_neuron = tensor.to("neuron")
        output_tensor = torch.zeros((world_size * tensor_size,), device="neuron")

        dist.all_gather_into_tensor(output_tensor, tensor_neuron)

        expected = torch.ones(tensor_size)
        for j in range(world_size):
            start_idx = j * tensor_size
            end_idx = (j + 1) * tensor_size
            assert torch.allclose(output_tensor[start_idx:end_idx].cpu(), expected)

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak <= 1073741824, f"Memory {peak} exceeds 1GB"


def run_bucketing_memory_flat_2d_tensor_test(rank, world_size, kwargs):
    """Test all_gather_into_tensor with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        tensor = torch.ones(1, tensor_size, dtype=torch.float32)
        tensor_neuron = tensor.to("neuron")
        gather_list = [torch.zeros(1, tensor_size, device="neuron") for _ in range(world_size)]

        dist.all_gather(gather_list, tensor_neuron)
        expected = tensor.clone()
        for gathered_tensor in gather_list:
            assert torch.allclose(gathered_tensor.cpu(), expected)

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak <= 1073741824, f"Memory {peak} exceeds 1GB"


class TestAllGather(BaseCollectiveOpTest):
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.int8,
            torch.uint8,
        ],
    )
    def test_all_gather_into_tensor_dtypes(self, dtype):
        """Test all_gather_into_tensor with different data types."""
        self.distributed_tester.run_test(run_all_gather_into_tensor_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_all_gather_into_tensor_shapes(self, shape):
        """Test all_gather_into_tensor with different tensor shapes."""
        self.distributed_tester.run_test(run_all_gather_into_tensor_shape_test, shape=shape)

    def test_all_gather_into_tensor_async(self):
        """Test asynchronous all_gather_into_tensor operation."""
        self.distributed_tester.run_test(run_all_gather_into_tensor_async_test)

    def test_all_gather_into_tensor_group(self):
        """Test all_gather_into_tensor with process groups."""
        self.distributed_tester.run_test(run_all_gather_into_tensor_group_test)

    @assert_raises(RuntimeError, match=r".*output shape mismatch, expected.*got.*")
    def test_all_gather_into_tensor_wrong_size(self):
        """Test error handling for incorrect output tensor size."""
        self.distributed_tester.run_test(run_all_gather_into_tensor_wrong_size_test)

    def test_all_gather_negative_values_test(self):
        """Test all_gather with tensors on different devices."""
        self.distributed_tester.run_test(run_all_gather_negative_values_test)

    def test_all_gather_base_debug_output(self):
        """Test _allgather_base debug output and functionality."""
        self.distributed_tester.run_test(run_all_gather_base_debug_test)

    def test_all_gather_base_reverse_dims(self):
        """Test _allgather_base with input having more dims than output."""
        self.distributed_tester.run_test(run_all_gather_base_reverse_test)

    @pytest.mark.xfail(reason="AG base with stacking outputs not supported yet")
    def test_all_gather_stack_outputs(self):
        """Test all_gather with stacked output tensors."""
        self.distributed_tester.run_test(run_all_gather_stack_outputs_test)

    def test_all_gather_multiple_single_dim(self):
        """Test all_gather with multiple single-dimension tensors."""
        self.distributed_tester.run_test(run_all_gather_multiple_single_dim_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test all_gather_into_tensor with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test all_gather_into_tensor with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory_flat_2d_tensor_test(self):
        """Test all_gather_into_tensor with bucketing for memory efficiency with flat 2d tensor"""
        self.distributed_tester.run_test(run_bucketing_memory_flat_2d_tensor_test)
