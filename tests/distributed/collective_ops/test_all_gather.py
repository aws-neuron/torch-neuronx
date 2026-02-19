import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_different_dtype_test(rank, world_size, kwargs):
    tensor = torch.ones(10, dtype=kwargs["dtype"]) * (rank + 1)
    tensor_neuron = tensor.to("neuron")

    # Create list of output tensors, one for each process
    output_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

    # Perform all_gather
    dist.all_gather(output_list, tensor_neuron)

    # Expected: list of tensors, each filled with (rank + 1)
    expected_list = [torch.ones(10, dtype=kwargs["dtype"]) * (i + 1) for i in range(world_size)]

    # Verify each gathered tensor matches expected
    for output, expected in zip(output_list, expected_list, strict=False):
        assert torch.allclose(output.cpu(), expected)


def run_different_shape_test_gather(rank, world_size, kwargs):
    tensor = torch.ones(kwargs["shape"]) * (rank + 1)  # Different value for each rank
    tensor_neuron = tensor.to("neuron")

    # Create output list to store gathered tensors
    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

    # Perform all_gather operation
    dist.all_gather(gather_list, tensor_neuron)

    # Create expected output
    expected_list = [torch.ones(kwargs["shape"]) * (i + 1) for i in range(world_size)]

    # Verify results
    for gathered, expected in zip(gather_list, expected_list, strict=False):
        assert torch.allclose(gathered.cpu(), expected)


def run_empty_tensor_test(rank, world_size, kwargs):
    # Test empty tensor
    empty_tensor = torch.tensor([], dtype=torch.float32)
    empty_tensor_neuron = empty_tensor.to("neuron")
    gather_list = [torch.zeros_like(empty_tensor_neuron) for _ in range(world_size)]
    dist.all_gather(gather_list, empty_tensor_neuron)


def run_large_tensor_test(rank, world_size, kwargs):
    tensor = torch.ones(1000000)  # 1M elements
    tensor_neuron = tensor.to("neuron")
    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor_neuron)

    # Expected: each tensor in gather_list should be ones
    expected = tensor.clone()

    # Convert gathered tensors back to CPU and verify
    for gathered_tensor in gather_list:
        assert torch.allclose(gathered_tensor.to("cpu"), expected)


def run_with_inf_inputs_test(rank, world_size, kwargs):
    tensor = torch.tensor([torch.inf]).clone().to("neuron")
    expected = tensor.cpu().clone()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor)

    # Verify each gathered tensor contains inf and matches the original
    for gathered_tensor in gather_list:
        assert torch.allclose(gathered_tensor.cpu(), expected)
        assert torch.isinf(gathered_tensor).all()


def run_async_operations(rank, world_size, kwargs):
    tensor = torch.full((10,), rank, dtype=torch.float32)
    expected_list = [torch.full((10,), i, dtype=torch.float32) for i in range(world_size)]
    tensor_neuron = tensor.to("neuron")
    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

    # Test with async operations
    work = dist.all_gather(gather_list, tensor_neuron, async_op=True)
    assert work is not None
    work.wait()
    for gathered_tensor, expected in zip(gather_list, expected_list, strict=True):
        assert torch.allclose(gathered_tensor.cpu(), expected)


def run_group_argument_test(rank, world_size, kwargs):
    group = dist.new_group([0, 1])
    tensor = torch.ones(10)
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")

    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(2)]  # size 2 for group [0, 1]

    if rank <= 1:  # Only ranks 0 and 1 are in the group
        dist.all_gather(gather_list, tensor_neuron, group=group)

        # Verify gathered tensors
        for gathered_tensor in gather_list:
            assert torch.allclose(gathered_tensor.cpu(), expected)


def run_different_device_test(rank, world_size, kwargs):
    tensor = torch.ones(10)

    # Create output list to store gathered tensors
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Perform all_gather operation
    dist.all_gather(gather_list, tensor)


def run_different_dtypes_mismatch_error(rank, world_size, kwargs):
    tensor = torch.randn(10, dtype=torch.float32).to("neuron")

    # Create gather list with different dtype tensors
    gather_list = [torch.zeros(10, dtype=torch.float16).to("neuron") for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)


def run_all_gather_with_uneven_tensors_test(rank, world_size, kwargs):
    input_tensor = torch.tensor([float(rank) for _ in range(rank + 1)]).to("neuron")
    output_tensor_list = [torch.zeros_like(input_tensor).to("neuron") for _ in range(world_size)]
    # Perform all_gather
    dist.all_gather(output_tensor_list, input_tensor)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_gather with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    tensor = torch.ones(tensor_size, dtype=torch.float32)
    expected = tensor.clone()
    tensor_neuron = tensor.to("neuron")
    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_neuron)
    for gathered_tensor in gather_list:
        assert torch.allclose(gathered_tensor.cpu(), expected)


def run_all_gather_partial_group_test(rank, world_size, kwargs):
    """Test all_gather with a partial group in a larger world.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        kwargs: Dictionary containing:
            - partial_group: List of ranks in the partial group
            - expected_values: List of expected tensors for each rank in the group
            - input_shape: Shape of input tensor (default: [2])
            - dtype: Tensor dtype (default: torch.long)
    """
    partial_group = kwargs.get("partial_group", [0, 1])
    expected_values = kwargs.get("expected_values")
    dtype = torch.long
    input_shape = 2

    # Create the partial group
    group = dist.new_group(partial_group)

    # Create input tensor based on rank
    input_tensor = torch.tensor([rank * 10 + i + 1 for i in range(input_shape)], dtype=dtype).to(
        "neuron"
    )

    if rank in partial_group:
        # Create output tensors for the partial group
        group_size = len(partial_group)
        output_list = [
            torch.zeros(input_shape, dtype=dtype).to("neuron") for _ in range(group_size)
        ]

        # Perform all_gather within the partial group
        dist.all_gather(output_list, input_tensor, group=group)

        # Verify results if expected values are provided
        for output, expected in zip(output_list, expected_values, strict=False):
            assert torch.allclose(
                output.cpu(), expected
            ), f"Rank {rank}: Output {output} doesn't match expected {expected}"
    else:
        # Ranks not in partial_group don't participate
        pass


class TestAllGather(BaseCollectiveOpTest):
    """Test cases for torch.distributed.all_reduce."""

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.int8,
            torch.bool,
            torch.uint8,
        ],
    )
    def test_different_dtypes(self, dtype):
        """Test all_reduce with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes_gather(self, shape):
        """Test all_gather with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test_gather, shape=shape)

    def test_edge_cases(self):
        """Test all_gather with edge cases like empty tensors and large tensors."""
        self.distributed_tester.run_test(run_large_tensor_test)

    def test_with_inf_inputs(self):
        """Test all_gather with tensors containing Inf values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_group_argument(self):
        """Test all_gather with group argument."""
        self.distributed_tester.run_test(run_group_argument_test)

    def test_async_operation(self):
        """Test asynchronous all_gather operation."""
        self.distributed_tester.run_test(run_async_operations)

    def test_group_operation(self):
        """Test all_gather with specific process groups."""
        self.distributed_tester.run_test(run_group_argument_test)

    @assert_raises(RuntimeError, match=r".*Expected neuron device, got cpu.*")
    def test_different_devices(self):
        """Test all_gather with tensors on different devices."""
        self.distributed_tester.run_test(run_different_device_test)

    @assert_raises(RuntimeError, match=r".*tensors cannot have zero size in dimension 0, found.*")
    def test_empty_tensor(self):
        """Test all_gather with empty tensor."""
        self.distributed_tester.run_test(run_empty_tensor_test)

    @assert_raises(
        RuntimeError,
        match=(
            r".*Invalid usage of tensors with different dtypesFound "
            "torch.float16 and  torch.float32.*"
        ),
    )
    def test_dtype_mismatch_error(self):
        """Test all_gather error handling with dtype mismatch. Error thrown by torch directly"""
        self.distributed_tester.run_test(run_different_dtypes_mismatch_error)

    @pytest.mark.xfail(reason="Uneven sized tensors are not supported")
    def test_with_uneven_sized_tensors(self, values):
        """Test all_reduce with specific input values."""
        self.distributed_tester.run_test(run_all_gather_with_uneven_tensors_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test gather with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test all_gather with bucketing to ensure memory stays under 1GB."""

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
        tensor_size = 31250000 + i * 31250000
        tensor = torch.ones(tensor_size, dtype=torch.float32)
        tensor_neuron = tensor.to("neuron")
        gather_list = [torch.zeros_like(tensor_neuron) for _ in range(world_size)]

        dist.all_gather(gather_list, tensor_neuron)

        expected = tensor.clone()
        for gathered_tensor in gather_list:
            assert torch.allclose(gathered_tensor.cpu(), expected)

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


def run_bucketing_memory_flat_2d_tensor_test(rank, world_size, kwargs):
    """Test all_gather with bucketing to ensure memory stays under 1GB."""

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
        tensor_size = 31250000 + i * 31250000
        tensor = torch.ones(1, tensor_size, dtype=torch.float32)
        tensor_neuron = tensor.to("neuron")
        # Create gather list with tensors having batch dimension
        gather_list = [torch.zeros(1, tensor_size, device="neuron") for _ in range(world_size)]

        dist.all_gather(gather_list, tensor_neuron)

        expected = tensor.clone()
        for gathered_tensor in gather_list:
            assert torch.allclose(gathered_tensor.cpu(), expected)

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


class TestAllGatherWorldSize4(BaseCollectiveOpTest):
    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.parametrize(
        "group_config",
        [
            pytest.param(
                {
                    "partial_group": [1, 2],
                    "expected_values": [
                        torch.tensor([11, 12]),  # rank 1's values
                        torch.tensor([21, 22]),  # rank 2's values
                    ],
                },
                marks=pytest.mark.xfail(
                    reason="Topology connecting ranks 1 and 2 not supported on Neuron \
                        we should have group validation and error handling for the same"
                ),
            ),
            {
                "partial_group": [0, 1],
                "expected_values": [
                    torch.tensor([1, 2]),  # rank 0's values
                    torch.tensor([11, 12]),  # rank 1's values
                ],
            },
            {
                "partial_group": [2, 3],
                "expected_values": [
                    torch.tensor([21, 22]),  # rank 2's values
                    torch.tensor([31, 32]),  # rank 3's values
                ],
            },
        ],
    )
    def test_all_gather_partial_group_test(self, group_config):
        partial_group = group_config["partial_group"]
        expected_values = group_config["expected_values"]
        self.distributed_tester.run_test(
            run_all_gather_partial_group_test,
            partial_group=partial_group,
            expected_values=expected_values,
        )

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test all_gather with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory_flat_2d_tensor_test(self):
        """Test all_gather_into_tensor with bucketing for memory efficiency with flat 2d tensor"""
        self.distributed_tester.run_test(run_bucketing_memory_flat_2d_tensor_test)

    def test_delayed_wait_after_completion(self):
        """Regression test: wait() after watchdog may have processed completion.

        Previously, the watchdog's detachStashedTensorShelf() could race with
        synchronize()'s unstashTensors(), causing SIGSEGV in pthread_mutex_lock.
        """
        self.distributed_tester.run_test(run_delayed_wait_after_completion_test)


def run_delayed_wait_after_completion_test(rank, world_size, kwargs):
    """Wait on async work after giving watchdog time to process it."""
    import time

    for _ in range(20):
        tensor = torch.ones(4, 4, device="neuron")
        output_list = [torch.zeros(4, 4, device="neuron") for _ in range(world_size)]

        work = dist.all_gather(output_list, tensor, async_op=True)

        # Sleep to let watchdog detect completion and call handleCompletion
        time.sleep(0.1)

        # wait() must not crash even if watchdog already detached the shelf
        work.wait()

        # Verify results
        for gathered in output_list:
            assert torch.allclose(gathered.cpu(), torch.ones(4, 4))
