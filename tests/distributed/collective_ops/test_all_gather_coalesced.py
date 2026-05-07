import os

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_basic_coalesced_test(rank, world_size, kwargs):
    """Test basic functionality of all_gather_coalesced with same-shaped tensors."""
    # Create input tensors with rank-specific values
    tensor1 = torch.ones(10, dtype=torch.float32) * rank
    tensor2 = torch.ones(10, dtype=torch.float32) * (rank + 10)
    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists
    output_lists = [
        [torch.zeros(10, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(10, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        assert torch.allclose(output_lists[0][i].cpu(), torch.ones(10) * i)
        assert torch.allclose(output_lists[1][i].cpu(), torch.ones(10) * (i + 10))


def run_supported_dtypes_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with different supported dtypes."""
    dtype = kwargs.get("dtype", torch.float32)

    # Create tensors of the specified dtype
    tensor1 = torch.full((5,), rank, dtype=dtype).to("neuron")
    tensor2 = torch.full((5,), rank + 1, dtype=dtype).to("neuron")

    input_tensors = [tensor1, tensor2]

    # Create output lists
    output_lists = [
        [torch.zeros(5, dtype=dtype).to("neuron") for _ in range(world_size)],
        [torch.zeros(5, dtype=dtype).to("neuron") for _ in range(world_size)],
    ]
    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        assert torch.all(output_lists[0][i].cpu() == i)
        assert torch.all(output_lists[1][i].cpu() == i + 1)


def run_different_shapes_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with tensors of different shapes."""
    shape = kwargs["shape"]
    # Create two tensors of the same shape but different values
    tensor1 = torch.ones(shape, dtype=torch.float32) * rank
    tensor2 = torch.ones(shape, dtype=torch.float32) * (rank + 10)

    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists
    output_lists = [
        [torch.zeros(shape, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(shape, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    expected1 = [torch.ones(shape) * i for i in range(world_size)]
    expected2 = [torch.ones(shape) * (i + 10) for i in range(world_size)]

    for i in range(world_size):
        assert torch.allclose(output_lists[0][i].cpu(), expected1[i])
        assert torch.allclose(output_lists[1][i].cpu(), expected2[i])


def run_input_dtypes_mismatch_error(rank, world_size, kwargs):
    """Test error handling when input tensors have different dtypes."""
    dtype1 = kwargs.get("dtype1", torch.float32)
    dtype2 = kwargs.get("dtype2", torch.int64)  # Different dtype

    # Create input tensors with different data types
    tensor1 = torch.ones(5, dtype=dtype1) * rank
    tensor2 = torch.ones(5, dtype=dtype2) * rank

    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists matching input types
    output_lists = [
        [torch.zeros(5, dtype=dtype1).to("neuron") for _ in range(world_size)],
        [torch.zeros(5, dtype=dtype2).to("neuron") for _ in range(world_size)],
    ]

    # This should raise ValueError due to different dtypes
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_different_input_shapes_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with input tensors of different shapes."""
    # Create input tensors with different shapes
    input_tensors = [
        torch.full((2, 2), rank + 1, dtype=torch.float32).to("neuron"),  # 2x2 tensor
        torch.full((1,), rank + 5, dtype=torch.float32).to("neuron"),  # scalar tensor
        torch.full((3,), rank + 2, dtype=torch.float32).to("neuron"),  # 1D tensor with 3 elements
    ]

    # Create output tensors with matching shapes
    output_lists = [
        [torch.zeros(2, 2, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(1, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        # First tensor (2x2)
        expected_val1 = i + 1
        assert torch.all(
            output_lists[0][i] == expected_val1
        ), f"Rank {rank}: expected tensor 0 from rank {i} to be "
        f"{expected_val1}, got {output_lists[0][i]}"

        # Second tensor (scalar)
        expected_val2 = i + 5
        assert torch.all(
            output_lists[1][i] == expected_val2
        ), f"Rank {rank}: expected tensor 1 from rank {i} to be "
        f"{expected_val2}, got {output_lists[1][i]}"

        # Third tensor (1D with 3 elements)
        expected_val3 = i + 2
        assert torch.all(
            output_lists[2][i] == expected_val3
        ), f"Rank {rank}: expected tensor 2 from rank {i} to be "
        f"{expected_val3}, got {output_lists[2][i]}"


def run_input_output_shape_mismatch_test(rank, world_size, kwargs):
    """Test that all_gather_coalesced correctly detects shape mismatches."""
    # Create input tensors with SAME shapes
    input_tensors = [
        torch.full((3,), rank + 1, dtype=torch.float32).to("neuron"),  # 1D tensor with 3 elements
        torch.full((3,), rank + 2, dtype=torch.float32).to("neuron"),  # 1D tensor with 3 elements
    ]

    # Create output tensors with one wrong shape (second tensor for rank 0 has wrong shape)
    output_lists = [
        [torch.zeros(3, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [
            torch.zeros(3 if i != 0 else 4, dtype=torch.float32).to("neuron")
            for i in range(world_size)
        ],
    ]

    # This should raise an error due to shape mismatch
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_special_values_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with special values (inf, nan, very large, very small)."""
    # Create rank-specific special values
    # Each rank uses the same values but we'll verify they're gathered correctly
    tensor1 = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), 1e38, 1e-38], dtype=torch.float32
    ).to("neuron")
    tensor2 = torch.tensor([-1.0, 0.0, 1.0, -2.0, 2.0], dtype=torch.float32).to("neuron")

    input_tensors = [tensor1, tensor2]

    # Create output lists
    output_lists = [
        [torch.zeros(5, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(5, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # After all_gather_coalesced, every rank should have received values from all ranks
    # Verify results for all ranks
    for r in range(world_size):
        output1 = output_lists[0][r].cpu()

        # Check regular values with appropriate tolerances
        regular_mask = ~(torch.isnan(output1) | torch.isinf(output1))
        if regular_mask.any():
            # Compare with expected tensor1 values (regular values only)
            expected_regular = tensor1.cpu()[regular_mask]
            assert torch.allclose(output1[regular_mask], expected_regular, rtol=1e-5, atol=1e-5)

        # Check tensor2 (all regular values)
        assert torch.allclose(output_lists[1][r].cpu(), tensor2.cpu(), rtol=1e-5, atol=1e-5)

        # Check special values in tensor1

        # Check positive infinity
        inf_pos_mask = tensor1.cpu() == float("inf")
        if inf_pos_mask.any():
            assert torch.all(output1[inf_pos_mask] == float("inf"))

        # Check negative infinity
        inf_neg_mask = tensor1.cpu() == float("-inf")
        if inf_neg_mask.any():
            assert torch.all(output1[inf_neg_mask] == float("-inf"))

        # Check NaN values
        nan_mask = torch.isnan(tensor1.cpu())
        if nan_mask.any():
            assert torch.all(torch.isnan(output1[nan_mask]))


def run_large_tensors_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with large tensors."""
    # Create large tensors (100K elements each) - use SAME SIZE for both
    tensor_size = 100000
    tensor1 = torch.ones(tensor_size) * rank
    tensor2 = torch.ones(tensor_size) * (rank + 1)  # Same size as tensor1

    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists
    output_lists = [
        [torch.zeros(tensor_size).to("neuron") for _ in range(world_size)],
        [torch.zeros(tensor_size).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results (check just first and last few elements)
    for i in range(world_size):
        assert torch.allclose(output_lists[0][i][:10].cpu(), torch.ones(10) * i)
        assert torch.allclose(output_lists[0][i][-10:].cpu(), torch.ones(10) * i)
        assert torch.allclose(output_lists[1][i][:10].cpu(), torch.ones(10) * (i + 1))
        assert torch.allclose(output_lists[1][i][-10:].cpu(), torch.ones(10) * (i + 1))


def run_empty_tensor_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with empty tensors."""
    # Create one empty tensor and one normal tensor
    empty_tensor = torch.tensor([], dtype=torch.float32).to("neuron")
    empty_tensor_2 = torch.tensor([], dtype=torch.float32).to("neuron")

    input_tensors = [empty_tensor, empty_tensor_2]

    # Create output lists
    output_lists = [
        [torch.zeros(0).to("neuron") for _ in range(world_size)],
        [torch.zeros(0).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_async_operation_test(rank, world_size, kwargs):
    """Test asynchronous all_gather_coalesced operation."""
    # Create input tensors
    tensor1 = torch.ones(10) * rank
    tensor2 = torch.ones(10) * (rank + 1)

    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists
    output_lists = [
        [torch.zeros(10).to("neuron") for _ in range(world_size)],
        [torch.zeros(10).to("neuron") for _ in range(world_size)],
    ]

    # Perform async coalesced allgather
    work = dist.all_gather_coalesced(output_lists, input_tensors, async_op=True)
    assert work is not None
    work.wait()

    # Verify results after waiting
    for i in range(world_size):
        assert torch.allclose(output_lists[0][i].cpu(), torch.ones(10) * i)
        assert torch.allclose(output_lists[1][i].cpu(), torch.ones(10) * (i + 1))


def run_process_group_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with custom process groups."""
    if world_size < 2:
        return

    # Create a process group with ranks [0, 1]
    group = dist.new_group([0, 1])

    # Create input tensors - both with SAME SHAPE
    shape = [5]
    tensor1 = torch.ones(shape) * rank
    tensor2 = torch.ones(shape) * (rank + 10)  # Same shape as tensor1

    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    if rank <= 1:  # Only ranks 0 and 1 participate
        # Create output lists for the smaller group (size 2)
        output_lists = [
            [torch.zeros(shape).to("neuron") for _ in range(2)],  # 2 ranks in group
            [torch.zeros(shape).to("neuron") for _ in range(2)],
        ]

        # Perform coalesced allgather with specific group
        dist.all_gather_coalesced(output_lists, input_tensors, group=group)

        # Verify results
        expected1 = [torch.ones(shape) * i for i in range(2)]
        expected2 = [torch.ones(shape) * (i + 10) for i in range(2)]

        for i in range(2):
            assert torch.allclose(output_lists[0][i].cpu(), expected1[i])
            assert torch.allclose(output_lists[1][i].cpu(), expected2[i])


def run_mismatched_sizes_error(rank, world_size, kwargs):
    """Test error handling with mismatched input/output lengths."""
    tensor1 = torch.ones(5) * rank
    tensor2 = torch.ones(5) * rank

    # Create 2 input tensors but only 1 output list
    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]
    output_lists = [
        [torch.zeros(5).to("neuron") for _ in range(world_size)]
        # Missing second output list!
    ]

    # Should raise ValueError due to mismatched lengths
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_wrong_output_size_error(rank, world_size, kwargs):
    """Test error handling with incorrect output list size."""
    tensor1 = torch.ones(5) * rank
    input_tensors = [tensor1.to("neuron")]

    # Create output list with wrong size (world_size-1 instead of world_size)
    output_lists = [[torch.zeros(5).to("neuron") for _ in range(world_size - 1)]]

    # Should raise ValueError due to incorrect output list size
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_extreme_size_difference_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with tensors of very different sizes."""
    # One tiny tensor (just 2 elements)
    small_tensor = torch.tensor([rank, rank + 1], dtype=torch.float32).to("neuron")

    # One large tensor (100K elements)
    large_tensor = torch.ones(100000, dtype=torch.float32).to("neuron") * rank

    input_tensors = [small_tensor, large_tensor]

    # Create output lists
    output_lists = [
        [torch.zeros(2, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(100000, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        # Check small tensor
        assert torch.allclose(
            output_lists[0][i].cpu(), torch.tensor([i, i + 1], dtype=torch.float32)
        )

        # Check large tensor (just sample first/last few elements)
        assert torch.allclose(
            output_lists[1][i][:10].cpu(), torch.ones(10, dtype=torch.float32) * i
        )
        assert torch.allclose(
            output_lists[1][i][-10:].cpu(), torch.ones(10, dtype=torch.float32) * i
        )


def run_complex_nested_shapes_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with complex nested tensor shapes."""
    # Create tensors with different complex shapes
    tensor1 = torch.full((3, 4, 2), rank, dtype=torch.float32).to("neuron")
    tensor2 = torch.full((2, 1, 3, 2), rank + 1, dtype=torch.float32).to("neuron")
    tensor3 = torch.full((5, 5), rank + 2, dtype=torch.float32).to("neuron")

    input_tensors = [tensor1, tensor2, tensor3]

    # Create output lists with matching shapes
    output_lists = [
        [torch.zeros(3, 4, 2, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(2, 1, 3, 2, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(5, 5, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        assert torch.all(output_lists[0][i] == i)
        assert torch.all(output_lists[1][i] == i + 1)
        assert torch.all(output_lists[2][i] == i + 2)


def run_dtype_mismatch_error(rank, world_size, kwargs):
    """Test error handling with dtype mismatch between input and output."""
    tensor = torch.ones(10, dtype=torch.float32) * rank
    input_tensors = [tensor.to("neuron")]

    # Create output with wrong dtype
    output_lists = [[torch.zeros(10, dtype=torch.int32).to("neuron") for _ in range(world_size)]]
    # Should raise error due to dtype mismatch
    dist.all_gather_coalesced(output_lists, input_tensors)


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Create input tensors with rank-specific values
    tensor1 = torch.ones(tensor_size, dtype=torch.float32) * rank
    tensor2 = torch.ones(tensor_size, dtype=torch.float32) * (rank + 10)
    input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

    # Create output lists
    output_lists = [
        [torch.zeros(tensor_size, dtype=torch.float32).to("neuron") for _ in range(world_size)],
        [torch.zeros(tensor_size, dtype=torch.float32).to("neuron") for _ in range(world_size)],
    ]

    # Perform coalesced allgather
    dist.all_gather_coalesced(output_lists, input_tensors)

    # Verify results
    for i in range(world_size):
        assert torch.allclose(output_lists[0][i].cpu(), torch.ones(tensor_size) * i)
        assert torch.allclose(output_lists[1][i].cpu(), torch.ones(tensor_size) * (i + 10))


def run_bucketing_memory_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        tensor1 = torch.ones(tensor_size, dtype=torch.float32)
        tensor2 = torch.ones(tensor_size, dtype=torch.float32) * 2
        input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

        output_lists = [
            [torch.zeros(tensor_size).to("neuron") for _ in range(world_size)],
            [torch.zeros(tensor_size).to("neuron") for _ in range(world_size)],
        ]

        dist.all_gather_coalesced(output_lists, input_tensors)

        for j in range(world_size):
            assert torch.allclose(output_lists[0][j].cpu(), torch.ones(tensor_size))
            assert torch.allclose(output_lists[1][j].cpu(), torch.ones(tensor_size) * 2)

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak < 1073741824, f"Memory {peak} exceeds 1GB"


def run_bucketing_memory_flat_2d_tensor_test(rank, world_size, kwargs):
    """Test all_gather_coalesced with bucketing to ensure memory stays under 1GB."""
    core_id = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    dev_id = int(core_id) // 8
    core_id = int(core_id) - (dev_id * 8)
    os.system(
        f"echo 0 | sudo tee /sys/devices/virtual/neuron_device/neuron{dev_id}/"
        f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
    )
    for i in range(3):
        tensor_size = 31250000 + i * 31250000
        tensor1 = torch.ones(1, tensor_size, dtype=torch.float32)
        tensor2 = torch.ones(1, tensor_size, dtype=torch.float32) * 2
        input_tensors = [tensor1.to("neuron"), tensor2.to("neuron")]

        output_lists = [
            [torch.zeros(1, tensor_size).to("neuron") for _ in range(world_size)],
            [torch.zeros(1, tensor_size).to("neuron") for _ in range(world_size)],
        ]

        dist.all_gather_coalesced(output_lists, input_tensors)

        for j in range(world_size):
            assert torch.allclose(output_lists[0][j].cpu(), torch.ones(1, tensor_size))
            assert torch.allclose(output_lists[1][j].cpu(), torch.ones(1, tensor_size) * 2)

        peak = int(
            os.popen(
                f"cat /sys/devices/virtual/neuron_device/neuron{dev_id}/"
                f"neuron_core{core_id}/stats/memory_usage/device_mem/model_shared_scratchpad/peak"
            )
            .read()
            .strip()
        )
        assert peak < 1073741824, f"Memory {peak} exceeds 1GB"


class TestAllGatherCoalesced(BaseCollectiveOpTest):
    """Test cases for torch.distributed.all_gather_coalesced."""

    def test_basic_functionality(self):
        """Test basic all_gather_coalesced with same-shaped tensors."""
        self.distributed_tester.run_test(run_basic_coalesced_test)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_supported_dtypes(self, dtype):
        """Test all_gather_coalesced with different supported dtypes."""
        self.distributed_tester.run_test(run_supported_dtypes_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape):
        """Test all_gather_coalesced with tensors of different shapes."""
        self.distributed_tester.run_test(run_different_shapes_test, shape=shape)

    @pytest.mark.parametrize(
        "dtype_combo",
        [
            {"dtype1": torch.float32, "dtype2": torch.int64},
            {"dtype1": torch.float16, "dtype2": torch.float32},
        ],
    )
    @assert_raises(RuntimeError, match=r".*Invalid usage of tensors with different dtypes.*")
    def test_input_dtypes_mismatch_error(self, dtype_combo):
        """Test error handling when input tensors have different dtypes.

        Error thrown directly by torch
        """
        self.distributed_tester.run_test(
            run_input_dtypes_mismatch_error,
            dtype1=dtype_combo["dtype1"],
            dtype2=dtype_combo["dtype2"],
        )

    def test_different_input_shapes(self):
        """Test all_gather_coalesced with tensors of different shapes."""
        self.distributed_tester.run_test(run_different_input_shapes_test)

    @assert_raises(RuntimeError)
    def test_input_output_shape_mismatch(self):
        """Test that all_gather_coalesced correctly validates output tensor shapes."""
        self.distributed_tester.run_test(run_input_output_shape_mismatch_test)

    def test_special_values(self):
        """Test all_gather_coalesced with special values."""
        self.distributed_tester.run_test(run_special_values_test)

    def test_large_tensors(self):
        """Test all_gather_coalesced with large tensors."""
        self.distributed_tester.run_test(run_large_tensors_test)

    @assert_raises(RuntimeError)
    def test_empty_tensor(self):
        """Test all_gather_coalesced with empty tensors."""
        self.distributed_tester.run_test(run_empty_tensor_test)

    def test_async_operation(self):
        """Test asynchronous all_gather_coalesced operation."""
        self.distributed_tester.run_test(run_async_operation_test)

    def test_process_group(self):
        """Test all_gather_coalesced with process groups."""
        self.distributed_tester.run_test(run_process_group_test)

    @pytest.mark.xfail(reason="Our implementation will construct a correct output_list.")
    @assert_raises(RuntimeError)
    def test_mismatched_sizes_error(self):
        """Test error handling with mismatched input/output lengths."""
        self.distributed_tester.run_test(run_mismatched_sizes_error)

    @assert_raises(RuntimeError)
    def test_wrong_output_size_error(self):
        """Test error handling with incorrect output list size."""
        self.distributed_tester.run_test(run_wrong_output_size_error)

    def test_extreme_size_difference(self):
        """Test all_gather_coalesced with tensors of extremely different sizes."""
        self.distributed_tester.run_test(run_extreme_size_difference_test)

    def test_complex_nested_shapes(self):
        """Test all_gather_coalesced with complex nested tensor shapes."""
        self.distributed_tester.run_test(run_complex_nested_shapes_test)

    @pytest.mark.xfail(reason="Our implementation will construct a correct output_list.")
    @assert_raises(RuntimeError)
    def test_dtype_mismatch(self):
        """Test error handling with dtype mismatch between input and output."""
        self.distributed_tester.run_test(run_dtype_mismatch_error)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test all_gather_coalesced with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory(self):
        """Test all_gather_coalesced with bucketing for memory efficiency."""
        self.distributed_tester.run_test(run_bucketing_memory_test)

    @pytest.mark.xfail(reason="Memory assertions require further investigation, thus xfailing.")
    def test_bucketing_memory_flat_2d_tensor_test(self):
        """Test all_gather_into_tensor with bucketing for memory efficiency with flat 2d tensor"""
        self.distributed_tester.run_test(run_bucketing_memory_flat_2d_tensor_test)


class TestAllGatherCoalescedWorldSize4(BaseCollectiveOpTest):
    """Test all_gather_coalesced with world_size=4."""

    @property
    def world_size(self) -> int:
        return 4

    def test_larger_world(self):
        """Test all_gather_coalesced with a larger world size."""
        self.distributed_tester.run_test(run_basic_coalesced_test)

    def test_partial_groups(self):
        """Test all_gather_coalesced with partial groups in larger world."""
        self.distributed_tester.run_test(run_process_group_test)
