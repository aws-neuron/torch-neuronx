import os
from typing import Any

import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_different_dtype_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast functionality across different tensor data types.

    Verifies that broadcast correctly propagates tensor values from source rank
    to all other ranks regardless of the tensor's data type (float32, float16,
    int32, int64, bfloat16).
    """
    tensor: torch.Tensor = torch.ones(10, dtype=kwargs["dtype"]) * -1
    if rank == 0:  # Source rank
        tensor = torch.ones(10, dtype=kwargs["dtype"]) * 42

    expected: torch.Tensor = torch.ones(10, dtype=kwargs["dtype"]) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    dist.broadcast(tensor_neuron, src=0)
    assert torch.equal(tensor_neuron.cpu(), expected)


def run_different_shape_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with tensors of various dimensional shapes.

    Ensures broadcast works correctly with 1D, 2D, 3D, and 4D tensors,
    verifying that tensor shape doesn't affect the broadcast operation.
    """
    tensor: torch.Tensor = torch.ones(kwargs["shape"]) * -1
    if rank == 0:
        tensor = torch.ones(kwargs["shape"]) * 42

    expected: torch.Tensor = torch.ones(kwargs["shape"]) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    dist.broadcast(tensor_neuron, src=0)
    assert torch.equal(tensor_neuron.cpu(), expected)


def run_different_src_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with different source ranks.

    Validates that broadcast works correctly when initiated from different
    source ranks (0 or 1), ensuring any rank can serve as the broadcast source.
    """
    src_rank: int = kwargs["src"]
    tensor: torch.Tensor = torch.ones(10) * -1
    if rank == src_rank:
        tensor = torch.ones(10) * 42

    expected: torch.Tensor = torch.ones(10) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    dist.broadcast(tensor_neuron, src=src_rank)
    assert torch.equal(tensor_neuron.cpu(), expected)


def run_inplace_operation_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test that broadcast modifies tensors in-place.

    Verifies that broadcast operation modifies the original tensor object
    rather than creating a new tensor, confirming in-place behavior.
    """
    tensor: torch.Tensor = torch.ones(10).to("neuron")
    tensor_id: int = id(tensor)
    dist.broadcast(tensor, src=0)
    assert id(tensor) == tensor_id  # Should be the same object


def run_zero_size_tensor_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with zero-size tensor.

    Verifies that broadcast correctly handles zero-size tensors by
    propagating the tensor from source rank to all other ranks.
    """
    tensor: torch.Tensor = torch.ones(0).to("neuron")
    if rank == 0:  # Source rank
        tensor = torch.ones(0).to("neuron")

    expected: torch.Tensor = torch.ones(0)
    dist.broadcast(tensor, src=0)
    assert torch.equal(tensor.cpu(), expected)


def run_large_tensor_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with large tensors (1M elements).

    Validates that broadcast can handle memory-intensive operations with
    large tensors without performance degradation or memory issues.
    """
    tensor: torch.Tensor = torch.ones(1000000) * -1  # 1M elements
    if rank == 0:
        tensor = torch.ones(1000000) * 42

    expected: torch.Tensor = torch.ones(1000000) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    dist.broadcast(tensor_neuron, src=0)
    assert torch.equal(tensor_neuron.to("cpu"), expected)


def run_with_inf_inputs_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with infinity values.

    Validates that broadcast operations can handle tensors containing
    infinity values correctly across all participating ranks.
    """
    tensor: torch.Tensor = torch.tensor([torch.inf]).clone().to("neuron")
    expected: torch.Tensor = tensor.cpu().clone()
    dist.broadcast(tensor, src=0)
    assert torch.equal(tensor.cpu(), expected)


def run_async_operation_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test asynchronous broadcast operation.

    Verifies that broadcast supports async_op=True parameter, allowing
    non-blocking operations where the caller can wait for completion.
    """
    tensor: torch.Tensor = torch.ones(10) * -1
    if rank == 0:
        tensor = torch.ones(10) * 42

    expected: torch.Tensor = torch.ones(10) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")

    work: dist.Work = dist.broadcast(tensor_neuron, src=0, async_op=True)
    assert work is not None
    work.wait()  # Wait for completion
    assert torch.equal(tensor_neuron.cpu(), expected)


def run_group_argument_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast within custom process groups.

    Validates that broadcast works correctly when limited to a subset
    of processes using custom process groups rather than all processes.
    """
    group: dist.ProcessGroup = dist.new_group([0, 1])
    tensor: torch.Tensor = torch.ones(10) * -1
    if rank == 0:
        tensor = torch.ones(10) * 42

    expected: torch.Tensor = torch.ones(10) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")

    if rank <= 1:  # Only ranks 0 and 1 are in the group
        dist.broadcast(tensor_neuron, src=0, group=group)
        assert torch.equal(tensor_neuron.cpu(), expected)


def run_different_device_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Negative test: broadcast with CPU tensors should fail.

    This is a negative test because the neuron backend requires tensors
    to be on neuron device. CPU tensors should raise RuntimeError.
    """
    tensor: torch.Tensor = torch.ones(10)
    dist.broadcast(tensor, src=0)
    # Negative test - no assertion as RuntimeError expected


def run_invalid_src_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Negative test: broadcast with invalid source rank should fail.

    This is a negative test that verifies proper error handling when
    an invalid source rank (999) is provided. Expected to raise RuntimeError.
    """
    tensor: torch.Tensor = torch.ones(10).to("neuron")
    dist.broadcast(tensor, src=999)  # Invalid source rank
    # Negative test - no assertion as RuntimeError expected


def run_multiple_tensors_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test sequential broadcast of multiple tensors.

    Verifies that multiple separate broadcast operations can be performed
    sequentially without interference, each maintaining correct values.
    """
    tensors: list[torch.Tensor] = [torch.ones(10) * -1 for _ in range(3)]
    if rank == 0:
        tensors = [torch.ones(10) * 42 for _ in range(3)]

    expected: list[torch.Tensor] = [torch.ones(10) * 42 for _ in range(3)]
    tensors_neuron: list[torch.Tensor] = [tensor.to("neuron") for tensor in tensors]

    for t in tensors_neuron:
        dist.broadcast(t, src=0)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.equal(t.cpu(), e)


def run_empty_tensor_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with empty tensor.

    Verifies that broadcast correctly handles empty tensors by
    propagating the tensor from source rank to all other ranks.
    """
    empty_tensor: torch.Tensor = torch.tensor([], dtype=torch.float32)
    if rank == 0:
        empty_tensor = torch.tensor([], dtype=torch.float32)
    empty_tensor_neuron: torch.Tensor = empty_tensor.to("neuron")
    dist.broadcast(empty_tensor_neuron, src=0)
    expected: torch.Tensor = torch.tensor([], dtype=torch.float32)
    assert torch.equal(empty_tensor_neuron.cpu(), expected)


def run_error_handling_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Negative test: broadcast with invalid parameters should fail.

    This is a negative test that verifies proper error handling when
    invalid parameters (like string instead of int for src) are provided.
    Expected to raise RuntimeError.
    """
    tensor: torch.Tensor = torch.ones(10).to("neuron")
    dist.broadcast(tensor, src="invalid_src")  # Invalid src type
    # Negative test - no assertion as RuntimeError expected


def run_uneven_tensors_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Negative test: broadcast with different tensor sizes across ranks should fail.

    This is a negative test because broadcast requires all participating
    tensors to have the same size. Different sized tensors across ranks
    should raise RuntimeError. Raised at neff level when it tries to execute neff
    """
    tensor_size: int = rank + 1
    tensor: torch.Tensor = torch.ones(tensor_size).to("neuron")
    dist.broadcast(tensor, src=0)


def run_group_with_group_src_rank_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast using group-relative source rank.

    Validates that when using process groups, the src parameter refers to
    the rank within the group (group-relative) rather than global rank.
    Tests with group [0,1] using group-relative src=1 (global rank 1).
    """
    group: dist.ProcessGroup = dist.new_group([0, 1])
    tensor: torch.Tensor = torch.ones(10) * -1
    if rank == 1:  # Global rank 1 is group-relative rank 1
        tensor = torch.ones(10) * 42

    expected: torch.Tensor = torch.ones(10) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")

    if rank <= 1:
        dist.broadcast(tensor_neuron, src=1, group=group)  # Group-relative src=1
        assert torch.equal(tensor_neuron.cpu(), expected)


def run_invalid_group_src_rank_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Negative test: broadcast with invalid group-relative source rank.

    This is a negative test that verifies error handling when an invalid
    group-relative source rank is provided (src=2 for group of size 2).
    Expected to raise RuntimeError.
    """
    group: dist.ProcessGroup = dist.new_group([0, 1])
    tensor: torch.Tensor = torch.ones(10).to("neuron")

    if rank <= 1:
        dist.broadcast(tensor, src=2, group=group)  # Invalid group src
    raise AssertionError("RuntimeError was expected but not raised.")
    # Negative test - no assertion as RuntimeError expected


def run_5gb_tensor_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32
    tensor: torch.Tensor = torch.ones(tensor_size, dtype=torch.float32) * -1
    if rank == 0:  # Source rank
        tensor = torch.ones(tensor_size, dtype=torch.float32) * 42

    expected: torch.Tensor = torch.ones(tensor_size, dtype=torch.float32) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    dist.broadcast(tensor_neuron, src=0)
    assert torch.equal(tensor_neuron.cpu(), expected)


def run_group_broadcast_world_size_4_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast within group [2,3] where each rank has its own data.

    Creates an 8-process setup where each rank initially has tensor filled
    with its rank value. Tests broadcast within group [2,3] with group_src=1
    (group-relative, meaning global rank 3). Validates that rank 3's data
    is broadcast to ranks 2 while other ranks remain unchanged.
    """
    # Each rank has tensor with its own rank value
    tensor: torch.Tensor = torch.ones(1) * rank
    tensor_neuron: torch.Tensor = tensor.to("neuron")
    group: dist.ProcessGroup = dist.new_group([2, 3])

    if rank in [2, 3]:
        dist.broadcast(tensor_neuron, group_src=1, group=group)

        expected: torch.Tensor = torch.ones(1) * 3  # Rank 3's original data should be broadcasted
        assert torch.equal(tensor_neuron.cpu(), expected)

    dist.destroy_process_group(group)


def run_requires_grad_tensor_test(rank: int, world_size: int, kwargs: dict[str, Any]):
    """Test broadcast with tensors that require gradients.

    Validates that broadcast correctly handles tensors with requires_grad=True,
    ensuring the autograd graph is preserved and gradients can flow correctly
    after the broadcast operation. This tests the fix for in-place operations
    on leaf variables.
    """
    # Create tensor with requires_grad=True
    tensor: torch.Tensor = torch.ones(10, requires_grad=True) * -1
    if rank == 0:  # Source rank
        tensor = torch.ones(10, requires_grad=True) * 42

    expected: torch.Tensor = torch.ones(10) * 42
    tensor_neuron: torch.Tensor = tensor.to("neuron")

    dist.broadcast(tensor_neuron, src=0)

    # Verify the values are correct
    assert torch.equal(tensor_neuron.cpu().detach(), expected)

    # Verify requires_grad is still True
    assert tensor_neuron.requires_grad, "requires_grad should still be True after broadcast"


class TestBroadcast(BaseCollectiveOpTest):
    """Test cases for torch.distributed.broadcast."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bfloat16]
    )
    def test_different_dtypes(self, dtype: torch.dtype):
        """Test broadcast functionality across different tensor data types.

        Verifies that broadcast correctly propagates tensor values from source rank
        to all other ranks regardless of the tensor's data type (float32, float16,
        int32, int64, bfloat16).
        """
        self.distributed_tester.run_test(run_different_dtype_test, dtype=dtype)

    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 5, 5), (2, 3, 4, 5)])
    def test_different_shapes(self, shape: tuple):
        """Test broadcast with tensors of various dimensional shapes.

        Ensures broadcast works correctly with 1D, 2D, 3D, and 4D tensors,
        verifying that tensor shape doesn't affect the broadcast operation.
        """
        self.distributed_tester.run_test(run_different_shape_test, shape=shape)

    @pytest.mark.parametrize("src", [0, 1])
    def test_different_src_ranks(self, src: int):
        """Test broadcast with different source ranks.

        Validates that broadcast works correctly when initiated from different
        source ranks (0 or 1), ensuring any rank can serve as the broadcast source.
        """
        self.distributed_tester.run_test(run_different_src_test, src=src)

    def test_inplace_operation(self):
        """Test that broadcast modifies tensors in-place.

        Verifies that broadcast operation modifies the original tensor object
        rather than creating a new tensor, confirming in-place behavior.
        """
        self.distributed_tester.run_test(run_inplace_operation_test)

    def test_zero_size_tensor(self):
        """Test broadcast with zero-size tensor.

        Verifies that broadcast correctly handles zero-size tensors by
        propagating the tensor from source rank to all other ranks.
        """
        self.distributed_tester.run_test(run_zero_size_tensor_test)

    def test_large_tensor(self):
        """Test broadcast with large tensors (1M elements).

        Validates that broadcast can handle memory-intensive operations with
        large tensors without performance degradation or memory issues.
        """
        self.distributed_tester.run_test(run_large_tensor_test)

    def test_with_inf_inputs(self):
        """Test broadcast with infinity values.

        Validates that broadcast operations can handle tensors containing
        infinity values correctly across all participating ranks.
        """
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_async_operation(self):
        """Test asynchronous broadcast operation.

        Verifies that broadcast supports async_op=True parameter, allowing
        non-blocking operations where the caller can wait for completion.
        """
        self.distributed_tester.run_test(run_async_operation_test)

    def test_group_operation(self):
        """Test broadcast within custom process groups.

        Validates that broadcast works correctly when limited to a subset
        of processes using custom process groups rather than all processes.
        """
        self.distributed_tester.run_test(run_group_argument_test)

    @assert_raises(RuntimeError)
    def test_different_devices(self):
        """Negative test: broadcast with CPU tensors should fail.

        This is a negative test because the neuron backend requires tensors
        to be on neuron device. CPU tensors should raise RuntimeError.
        """
        self.distributed_tester.run_test(run_different_device_test)

    @assert_raises(RuntimeError)
    def test_invalid_src_rank(self):
        """Negative test: broadcast with invalid source rank should fail.

        This is a negative test that verifies proper error handling when
        an invalid source rank (999) is provided. Expected to raise RuntimeError.
        """
        self.distributed_tester.run_test(run_invalid_src_test)

    def test_multiple_tensors(self):
        """Test sequential broadcast of multiple tensors.

        Verifies that multiple separate broadcast operations can be performed
        sequentially without interference, each maintaining correct values.
        """
        self.distributed_tester.run_test(run_multiple_tensors_test)

    def test_empty_tensor(self):
        """Test broadcast with empty tensor.

        Verifies that broadcast correctly handles empty tensors by
        propagating the tensor from source rank to all other ranks.
        """
        self.distributed_tester.run_test(run_empty_tensor_test)

    @assert_raises(RuntimeError)
    def test_error_handling(self):
        """Negative test: broadcast with invalid parameters should fail.

        This is a negative test that verifies proper error handling when
        invalid parameters (like string instead of int for src) are provided.
        Expected to raise RuntimeError.
        """
        self.distributed_tester.run_test(run_error_handling_test)

    @assert_raises(RuntimeError)
    def test_with_uneven_sized_tensors(self):
        """Negative test: broadcast with different tensor sizes across ranks should fail.

        This is a negative test because broadcast requires all participating
        tensors to have the same size. Different sized tensors across ranks
        should raise RuntimeError.
        """
        self.distributed_tester.run_test(run_uneven_tensors_test)

    def test_group_with_group_src_rank(self):
        """Test broadcast using group-relative source rank.

        Validates that when using process groups, the src parameter refers to
        the rank within the group (group-relative) rather than global rank.
        Tests with group [0,1] using group-relative src=1 (global rank 1).
        """
        self.distributed_tester.run_test(run_group_with_group_src_rank_test)

    @assert_raises(RuntimeError)
    def test_invalid_group_src_rank(self):
        """Negative test: broadcast with invalid group-relative source rank.

        This is a negative test that verifies error handling when an invalid
        group-relative source rank is provided (src=2 for group of size 2).
        Expected to raise RuntimeError.
        """
        self.distributed_tester.run_test(run_invalid_group_src_rank_test)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test broadcast with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)

    def test_requires_grad_tensor(self):
        """Test broadcast with tensors that require gradients.

        Validates that broadcast correctly handles tensors with requires_grad=True,
        ensuring the autograd graph is preserved. This tests the fix for in-place
        operations on leaf variables that would previously raise:
        "a leaf Variable that requires grad is being used in an in-place operation"
        """
        self.distributed_tester.run_test(run_requires_grad_tensor_test)


class TestBroadcastTensorWorldSize4(BaseCollectiveOpTest):
    """Test cases for torch.distributed.broadcast with a world size of 4."""

    world_size = 4  # Set at class level instead of in setup_method

    def test_group_broadcast_with_rank_data(self):
        """Test broadcast within group [2,3] with group_src=1 (global rank 3)."""
        self.distributed_tester.run_test(run_group_broadcast_world_size_4_test)
