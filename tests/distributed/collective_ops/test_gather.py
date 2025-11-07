import os

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest


def verify_gather_list(gather_list, expected_list, inf_test=False):
    for output, expected in zip(gather_list, expected_list, strict=True):
        if inf_test:
            assert torch.isinf(output).all()
        else:
            assert torch.allclose(output.cpu(), expected)


def run_different_dtype_test(rank, world_size, kwargs):
    tensor = torch.ones(10, dtype=kwargs["dtype"]) * (rank + 1)
    tensor_neuron = tensor.to("neuron")

    root_rank = 0
    # Root rank provides pre-allocated output tensors, non-root ranks don't receive output
    output_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor_neuron, output_list, dst=root_rank)

    if rank == root_rank:
        # Verify each gathered tensor matches expected
        expected_list = [torch.ones(10, dtype=kwargs["dtype"]) * (i + 1) for i in range(world_size)]
        verify_gather_list(output_list, expected_list)


def run_different_shape_test_gather(rank, world_size, kwargs):
    tensor = torch.ones(kwargs["shape"]) * (rank + 1)
    tensor_neuron = tensor.to("neuron")

    root_rank = 0
    # Root rank creates output list, non-root ranks don't
    gather_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor_neuron, gather_list)

    if rank == root_rank:
        # Create expected output and verify results
        expected_list = [torch.ones(kwargs["shape"]) * (i + 1) for i in range(world_size)]
        verify_gather_list(gather_list, expected_list)


def run_group_dst_gather(rank, world_size, kwargs):
    shape = [2, 2]
    tensor = torch.ones(shape) * (rank + 1)
    tensor_neuron = tensor.to("neuron")

    root_rank = kwargs["group_dst"]
    # Root rank creates output list, non-root ranks don't
    gather_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor_neuron, gather_list, group_dst=root_rank)

    if rank == root_rank:
        # Create expected output and verify results
        expected_list = [torch.ones(shape) * (i + 1) for i in range(world_size)]
        verify_gather_list(gather_list, expected_list)


def run_empty_tensor_test(rank, world_size, kwargs):
    empty_tensor = torch.tensor([], dtype=torch.float32)
    empty_tensor_neuron = empty_tensor.to("neuron")

    root_rank = 0
    gather_list = (
        [torch.zeros_like(empty_tensor_neuron) for _ in range(world_size)]
        if rank == root_rank
        else None
    )
    dist.gather(empty_tensor_neuron, gather_list, dst=root_rank)

    if rank == root_rank:
        # Verify the gather_list has correct length
        assert (
            len(gather_list) == world_size
        ), f"Expected gather_list length {world_size}, got {len(gather_list)}"

        # Verify each gathered tensor is empty and has correct properties
        for i, gathered_tensor in enumerate(gather_list):
            assert gathered_tensor.numel() == 0, f"Tensor {i} should be empty"
            assert torch.equal(
                gathered_tensor.cpu(), empty_tensor
            ), f"Tensor {i} doesn't match original empty tensor"


def run_large_tensor_test(rank, world_size, kwargs):
    tensor = torch.ones(1000000)  # 1M elements
    tensor_neuron = tensor.to("neuron")

    root_rank = 0
    gather_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor_neuron, gather_list, dst=root_rank)

    if rank == root_rank:
        # Expected: each tensor in gather_list should be ones
        expected = [tensor.cpu() for _ in range(world_size)]
        # Convert gathered tensors back to CPU and verify
        verify_gather_list(gather_list, expected)


def run_with_inf_inputs_test(rank, world_size, kwargs):
    tensor = torch.tensor([torch.inf]).clone().to("neuron")
    expected = [tensor.cpu() for _ in range(world_size)]

    root_rank = 0
    gather_list = (
        [torch.zeros_like(tensor) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor, gather_list, dst=root_rank)

    if rank == root_rank:
        # Verify each gathered tensor contains inf and matches the original
        verify_gather_list(gather_list, expected, inf_test=True)


def run_async_operations(rank, world_size, kwargs):
    tensor = torch.full((10,), rank, dtype=torch.float32)
    expected = [torch.full((10,), i, dtype=torch.float32) for i in range(world_size)]
    tensor_neuron = tensor.to("neuron")

    root_rank = 0
    gather_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    # Test with async operations
    work = dist.gather(tensor_neuron, gather_list, dst=root_rank, async_op=True)
    assert work is not None
    work.wait()

    if rank == root_rank:
        verify_gather_list(gather_list, expected)


def run_group_argument_test(rank, world_size, kwargs):
    group = dist.new_group([0, 1])
    tensor = torch.ones(10)
    expected = [tensor.cpu() for _ in range(dist.get_world_size(group))]
    tensor_neuron = tensor.to("neuron")

    root_rank = 0
    # size 2 for group [0, 1]
    gather_list = [torch.zeros_like(tensor_neuron) for _ in range(2)] if rank == root_rank else None
    dist.gather(tensor_neuron, gather_list, dst=root_rank, group=group)

    if rank == root_rank:
        # Verify gathered tensors
        verify_gather_list(gather_list, expected)


def run_different_device_test(rank, world_size, kwargs):
    tensor = torch.ones(10)

    root_rank = 0
    # Create output list to store gathered tensors
    gather_list = (
        [torch.zeros_like(tensor) for _ in range(world_size)] if rank == root_rank else None
    )
    # Perform gather operation
    dist.gather(tensor, gather_list, dst=root_rank)


def run_different_dtypes_mismatch_error(rank, world_size, kwargs):
    tensor = torch.randn(10, dtype=torch.float32).to("neuron")

    root_rank = 0
    # Create gather list with different dtype tensors
    gather_list = (
        [torch.zeros(10, dtype=torch.float16).to("neuron") for _ in range(world_size)]
        if rank == root_rank
        else None
    )
    dist.gather(tensor, gather_list, dst=root_rank)


def run_different_dtypes_mismatch_in_gather_list_error(rank, world_size, kwargs):
    tensor = torch.randn(3, dtype=torch.float32).to("neuron")

    root_rank = 0
    # Create gather list with different dtype tensors
    gather_list = (
        [
            torch.zeros(3, dtype=torch.float16).to("neuron"),
            torch.zeros(3, dtype=torch.bfloat16).to("neuron"),
            torch.zeros(3, dtype=torch.float32).to("neuron"),
        ]
        if rank == root_rank
        else None
    )
    dist.gather(tensor, gather_list, dst=root_rank)


def run_gather_with_uneven_tensors_test(rank, world_size, kwargs):
    input_tensor = torch.tensor([float(rank) for _ in range(rank + 1)]).to("neuron")

    root_rank = 0
    output_tensor_list = (
        [torch.zeros_like(input_tensor).to("neuron") for _ in range(world_size)]
        if rank == root_rank
        else None
    )
    # Perform gather
    dist.gather(input_tensor, output_tensor_list, dst=root_rank)


def run_different_root_ranks_test(rank, world_size, kwargs):
    """Test gather with different root ranks (not just rank 0)"""
    tensor = torch.ones(10) * (rank + 1)
    tensor_neuron = tensor.to("neuron")

    # Test with root_rank = 1 (if world_size > 1)
    root_rank = 1 if world_size > 1 else 0
    gather_list = (
        [torch.zeros_like(tensor_neuron) for _ in range(world_size)] if rank == root_rank else None
    )
    dist.gather(tensor_neuron, gather_list, dst=root_rank)

    if rank == root_rank:
        # Verify each gathered tensor matches expected
        expected_list = [torch.ones(10) * (i + 1) for i in range(world_size)]
        verify_gather_list(gather_list, expected_list)


def run_invalid_root_rank_test(rank, world_size, kwargs):
    """Test gather with invalid root rank"""
    tensor = torch.ones(10).to("neuron")

    invalid_root = world_size + 1  # Invalid root rank
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)] if rank == 0 else None
    dist.gather(tensor, gather_list, dst=invalid_root)


class TestGather(BaseCollectiveOpTest):
    """Test cases for torch.distributed.gather."""

    @property
    def world_size(self) -> int:
        return 2

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
        ],
    )
    def test_different_dtypes(self, dtype):
        """Test gather with different data types."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes_gather(self, shape):
        """Test gather with different tensor shapes."""
        self.distributed_tester.run_test(run_different_shape_test_gather, shape=shape)

    @pytest.mark.parametrize("group_dst", [0, 1])
    def test_group_dst_gather(self, group_dst):
        """Test gather with different tensor shapes."""
        self.distributed_tester.run_test(run_group_dst_gather, group_dst=group_dst)

    def test_edge_cases(self):
        """Test gather with edge cases like large tensors."""
        self.distributed_tester.run_test(run_large_tensor_test)

    def test_with_inf_inputs(self):
        """Test gather with tensors containing Inf values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_group_argument(self):
        """Test gather with group argument."""
        self.distributed_tester.run_test(run_group_argument_test)

    def test_async_operation(self):
        """Test asynchronous gather operation."""
        self.distributed_tester.run_test(run_async_operations)

    def test_group_operation(self):
        """Test gather with specific process groups."""
        self.distributed_tester.run_test(run_group_argument_test)

    def test_different_devices(self):
        """Test gather with tensors on different devices."""
        with pytest.raises(RuntimeError, match=r".*Operation all_gather is not implemented.*"):
            self.distributed_tester.run_test(run_different_device_test)

    def test_empty_tensor(self):
        """Test gather with empty tensor."""
        with pytest.raises(RuntimeError, match=r".*Operation all_gather is not implemented.*"):
            self.distributed_tester.run_test(run_empty_tensor_test)

    def test_dtype_mismatch_error(self):
        """Test gather error handling with dtype mismatch."""
        with pytest.raises(
            RuntimeError,
            match=r".*Invalid usage of tensors with different dtypesFound "
            "torch.float32 and  torch.float16.*",
        ):
            self.distributed_tester.run_test(run_different_dtypes_mismatch_error)

    def test_dtype_mismatch_in_gather_list_error(self):
        """Test gather error handling with dtype mismatch."""
        with pytest.raises(
            RuntimeError, match=r".*Invalid usage of tensors with different dtypesFound .*"
        ):
            self.distributed_tester.run_test(run_different_dtypes_mismatch_in_gather_list_error)

    def test_different_root_ranks(self):
        """Test gather with different root ranks."""
        self.distributed_tester.run_test(run_different_root_ranks_test)

    def test_invalid_root_rank(self):
        """Test gather with invalid root rank."""
        with pytest.raises(RuntimeError, match=r".*Invalid rootRank.*"):
            self.distributed_tester.run_test(run_invalid_root_rank_test)

    def test_with_uneven_sized_tensors(self):
        """Test gather with uneven sized tensors."""
        with pytest.raises(RuntimeError, match=r".*.*"):
            self.distributed_tester.run_test(run_gather_with_uneven_tensors_test)


def run_gather_partial_group_test(rank, world_size, kwargs):
    partial_group = kwargs["partial_group"]
    expected_values = kwargs["expected_values"]
    # Use provided root_rank or default to first rank in group
    root_rank = kwargs.get("root_rank", partial_group[0])

    # Create process group
    group = dist.new_group(partial_group)

    # Create input tensor based on rank
    input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], dtype=torch.float32)
    input_tensor_neuron = input_tensor.to("neuron")

    if rank in partial_group:
        # Root rank in the group: prepare output list, non-root rank in the group: just send
        gather_list = (
            [torch.zeros_like(input_tensor_neuron) for _ in range(len(partial_group))]
            if rank == root_rank
            else None
        )
        dist.gather(input_tensor_neuron, gather_list, dst=root_rank, group=group)
    # Ranks not in the group don't participate

    # Verify results (only on root rank)
    if rank in partial_group and rank == root_rank:
        verify_gather_list(gather_list, expected_values)


class TestGatherWorldSize4(BaseCollectiveOpTest):
    @property
    def world_size(self) -> int:
        return 4

    # Full world_size tests (reusing existing test functions)
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
        ],
    )
    def test_different_dtypes_world_size_4(self, dtype):
        """Test gather with different data types on 4 ranks."""
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes_gather_world_size_4(self, shape):
        """Test gather with different tensor shapes on 4 ranks."""
        self.distributed_tester.run_test(run_different_shape_test_gather, shape=shape)

    def test_edge_cases_world_size_4(self):
        """Test gather with edge cases like large tensors on 4 ranks."""
        self.distributed_tester.run_test(run_large_tensor_test)

    @pytest.mark.parametrize(
        "group_config",
        [
            pytest.param(
                {
                    "partial_group": [1, 2],
                    "expected_values": [
                        torch.tensor([11, 12], dtype=torch.float32),  # rank 1's values
                        torch.tensor([21, 22], dtype=torch.float32),  # rank 2's values
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
                    torch.tensor([1, 2], dtype=torch.float32),  # rank 0's values
                    torch.tensor([11, 12], dtype=torch.float32),  # rank 1's values
                ],
            },
            {
                "partial_group": [2, 3],
                "expected_values": [
                    torch.tensor([21, 22], dtype=torch.float32),  # rank 2's values
                    torch.tensor([31, 32], dtype=torch.float32),  # rank 3's values
                ],
            },
        ],
    )
    def test_gather_partial_group_test(self, group_config):
        partial_group = group_config["partial_group"]
        expected_values = group_config["expected_values"]
        self.distributed_tester.run_test(
            run_gather_partial_group_test,
            partial_group=partial_group,
            expected_values=expected_values,
        )

    @pytest.mark.parametrize(
        "group_config",
        [
            {
                "partial_group": [0, 1],
                "root_rank": 1,  # Non-first rank as root
                "expected_values": [
                    torch.tensor([1, 2], dtype=torch.float32),  # rank 0's values
                    torch.tensor([11, 12], dtype=torch.float32),  # rank 1's values
                ],
            },
            {
                "partial_group": [2, 3],
                "root_rank": 3,  # Non-first rank as root
                "expected_values": [
                    torch.tensor([21, 22], dtype=torch.float32),  # rank 2's values
                    torch.tensor([31, 32], dtype=torch.float32),  # rank 3's values
                ],
            },
        ],
    )
    def test_gather_partial_group_different_roots(self, group_config):
        """Test gather with different root ranks within partial groups"""
        partial_group = group_config["partial_group"]
        root_rank = group_config["root_rank"]
        expected_values = group_config["expected_values"]
        self.distributed_tester.run_test(
            run_gather_partial_group_test,
            partial_group=partial_group,
            root_rank=root_rank,
            expected_values=expected_values,
        )
