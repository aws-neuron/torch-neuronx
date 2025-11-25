import pytest
import torch
import torch.distributed as dist

from tests.utils.neuron_test_utils import assert_raises

from .base_collective_op import BaseCollectiveOpTest


def run_basic_coalesced_test(rank, world_size, kwargs):
    """Test basic all_reduce_coalesced functionality with multiple tensors"""
    # Each rank contributes different values to make verification clear
    tensors = [
        torch.full(
            (2, 2), rank + 1, dtype=torch.float32
        ),  # rank 0: [[1,1],[1,1]], rank 1: [[2,2],[2,2]]
        torch.full((3,), rank + 2, dtype=torch.float32),  # rank 0: [2,2,2], rank 1: [3,3,3]
    ]

    # Expected after SUM reduction across 2 ranks:
    # tensor 0: [[1,1],[1,1]] + [[2,2],[2,2]] = [[3,3],[3,3]]
    # tensor 1: [2,2,2] + [3,3,3] = [5,5,5]
    expected = [
        torch.full((2, 2), 3.0),  # 1 + 2 = 3
        torch.full((3,), 5.0),  # 2 + 3 = 5
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


def run_avg_coalesced_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with AVG operation"""
    tensors = [
        torch.full((2,), rank * 2 + 1, dtype=torch.float32),  # rank 0: [1,1], rank 1: [3,3]
        torch.tensor([rank + 1], dtype=torch.float32),  # rank 0: [1], rank 1: [2]
        torch.full((3,), rank * 2 + 2, dtype=torch.float32),  # rank 0: [2,2,2], rank 1: [4,4,4]
    ]

    # Expected after AVG reduction across 2 ranks:
    # tensor 0: ([1,1] + [3,3]) / 2 = [2,2]
    # tensor 1: ([1] + [2]) / 2 = [1.5]
    # tensor 2: ([2,2,2] + [4,4,4]) / 2 = [3,3,3]
    expected = [torch.full((2,), 2.0), torch.tensor([1.5]), torch.full((3,), 3.0)]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.AVG)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


def run_min_coalesced_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with MIN operation"""
    tensors = [
        torch.full((2,), rank * 3 + 1, dtype=torch.float32),  # rank 0: [1,1], rank 1: [4,4]
        torch.tensor([rank * 2 + 2], dtype=torch.float32),  # rank 0: [2], rank 1: [4]
        torch.full((3,), rank * 2 + 3, dtype=torch.float32),  # Third tensor
    ]

    # Expected after MIN reduction across 2 ranks:
    # tensor 0: min([1,1], [4,4]) = [1,1]
    # tensor 1: min([2], [4]) = [2]
    expected = [
        torch.full((2,), 1.0),
        torch.tensor([2.0]),
        torch.full((3,), 3.0),
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.MIN)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


def run_max_coalesced_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with MAX operation"""
    tensors = [
        torch.full((2,), rank * 3 + 1, dtype=torch.float32),  # rank 0: [1,1], rank 1: [4,4]
        torch.tensor([rank * 2 + 2], dtype=torch.float32),  # rank 0: [2], rank 1: [4]
    ]

    # Expected after MAX reduction across 2 ranks:
    # tensor 0: max([1,1], [4,4]) = [4,4]
    # tensor 1: max([2], [4]) = [4]
    expected = [
        torch.full((2,), 4.0),
        torch.tensor([4.0]),
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.MAX)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


def run_max_coalesced_3_tensors_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with MAX operation using 3 tensors"""
    tensors = [
        torch.full((2,), rank * 2 + 1, dtype=torch.float32),  # rank 0: [1,1], rank 1: [3,3]
        torch.tensor([rank + 2], dtype=torch.float32),  # rank 0: [2], rank 1: [3]
        torch.full((3,), rank * 4 + 1, dtype=torch.float32),  # rank 0: [1,1,1], rank 1: [5,5,5]
    ]

    # Expected after MAX reduction across 2 ranks:
    # tensor 0: max([1,1], [3,3]) = [3,3]
    # tensor 1: max([2], [3]) = [3]
    # tensor 2: max([1,1,1], [5,5,5]) = [5,5,5]
    expected = [
        torch.full((2,), 3.0),  # max(1, 3) = 3
        torch.tensor([3.0]),  # max(2, 3) = 3
        torch.full((3,), 5.0),  # max(1, 5) = 5
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.MAX)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


@assert_raises(RuntimeError)
def run_mismatched_empty_tensor_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with mismatched empty tensors across ranks

    Rank 0 has empty tensor, Rank 1 has valid tensor at same position.
    This should fail because ranks have different numbers of non-empty tensors
    after filtering.
    """
    if rank == 0:
        tensors = [
            torch.full((2, 2), 1.0, dtype=torch.float32),  # Non-empty
            torch.empty(0, dtype=torch.float32),  # Empty - only on rank 0!
            torch.full((3,), 2.0, dtype=torch.float32),  # Non-empty
        ]
    else:  # rank == 1
        tensors = [
            torch.full((2, 2), 2.0, dtype=torch.float32),  # Non-empty
            torch.full((1,), 3.0, dtype=torch.float32),  # Non-empty - rank 1 has data here!
            torch.full((3,), 3.0, dtype=torch.float32),  # Non-empty
        ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]

    # This should fail - ranks have mismatched tensor counts after filtering
    # Rank 0: filters to 2 non-empty tensors
    # Rank 1: filters to 3 non-empty tensors
    # Expect some error from XLA/collective layer
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)


def run_empty_tensor_mixed_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with mix of empty and non-empty tensors"""
    tensors = [
        torch.full((2, 2), rank + 1, dtype=torch.float32),  # Non-empty
        torch.empty(0, dtype=torch.float32),  # Empty tensor
        torch.full((3,), rank + 2, dtype=torch.float32),  # Non-empty
    ]

    expected = [
        torch.full((2, 2), 3.0),  # 1 + 2 = 3
        torch.empty(0, dtype=torch.float32),  # Stays empty
        torch.full((3,), 5.0),  # 2 + 3 = 5
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)

    for t, e in zip(tensors_neuron, expected, strict=False):
        if t.numel() == 0:
            assert t.numel() == e.numel(), "Empty tensor should remain empty"
        else:
            assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


def run_all_empty_tensors_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with all empty tensors"""
    tensors = [
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, 3, dtype=torch.float32),
        torch.empty(2, 0, dtype=torch.float32),
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)

    # All should remain empty
    for t in tensors_neuron:
        assert t.numel() == 0, "Empty tensor should remain empty"


def run_single_empty_tensor_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with single empty tensor"""
    tensors = [torch.empty(0, 5, dtype=torch.float32)]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)

    assert tensors_neuron[0].numel() == 0, "Empty tensor should remain empty"
    assert tensors_neuron[0].shape == torch.Size([0, 5]), "Shape should be preserved"


def run_empty_tensor_different_ops_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with empty tensors and different operations"""
    reduce_op = kwargs["reduce_op"]

    tensors = [
        torch.full((2,), rank + 1, dtype=torch.float32),
        torch.empty(0, dtype=torch.float32),  # Empty in middle
        torch.full((3,), rank + 2, dtype=torch.float32),
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=reduce_op)

    # Check that empty tensor is still empty
    assert tensors_neuron[1].numel() == 0, "Empty tensor should remain empty"

    # Check that non-empty tensors were reduced
    assert tensors_neuron[0].numel() > 0, "Non-empty tensor should remain non-empty"
    assert tensors_neuron[2].numel() > 0, "Non-empty tensor should remain non-empty"


def run_5gb_tensor_test(rank, world_size, kwargs):
    """Test all_reduce_coalesced with 5GB tensor (~1.25 billion float32 elements)."""
    tensor_size = 1250000000  # ~5GB for float32

    # Each rank contributes different values
    tensors = [
        torch.full((tensor_size,), rank + 1, dtype=torch.float32),
        torch.full((tensor_size,), rank + 2, dtype=torch.float32),
    ]

    # Expected after SUM reduction across 2 ranks:
    expected = [
        torch.full((tensor_size,), 3.0),  # 1 + 2 = 3
        torch.full((tensor_size,), 5.0),  # 2 + 3 = 5
    ]

    tensors_neuron = [tensor.to("neuron") for tensor in tensors]
    dist.all_reduce_coalesced(tensors_neuron, op=dist.ReduceOp.SUM)

    for t, e in zip(tensors_neuron, expected, strict=False):
        assert torch.allclose(t.cpu(), e), f"Expected {e}, got {t.cpu()}"


class TestAllReduceCoalesced(BaseCollectiveOpTest):
    """Test class for all_reduce_coalesced collective operation"""

    def test_basic_coalesced(self):
        """Test basic all_reduce_coalesced functionality with multiple tensors"""
        self.distributed_tester.run_test(run_basic_coalesced_test)

    def test_avg_coalesced(self):
        """Test all_reduce_coalesced with AVG operation"""
        self.distributed_tester.run_test(run_avg_coalesced_test)

    def test_min_coalesced(self):
        """Test all_reduce_coalesced with MIN operation"""
        self.distributed_tester.run_test(run_min_coalesced_test)

    @pytest.mark.xfail(
        reason="Not supported max reduce op for all-reduce-coalesced with max when "
        "2 inputs. HLO IR looks correct, but output isn't. Same issue with min and average."
        "Potentially compiler issue"
    )
    def test_max_coalesced(self):
        """Test all_reduce_coalesced with MAX operation"""
        self.distributed_tester.run_test(run_max_coalesced_test)

    def test_max_coalesced_3_tensors(self):
        """Test all_reduce_coalesced with MAX operation using 3 tensors"""
        self.distributed_tester.run_test(run_max_coalesced_3_tensors_test)

    def test_empty_tensor_mixed(self):
        """Test all_reduce_coalesced with mix of empty and non-empty tensors"""
        self.distributed_tester.run_test(run_empty_tensor_mixed_test)

    def test_all_empty_tensors(self):
        """Test all_reduce_coalesced with all empty tensors"""
        self.distributed_tester.run_test(run_all_empty_tensors_test)

    def test_mismatched_empty_tensor(self):
        """Test all_reduce_coalesced fails when ranks have mismatched empty tensors"""
        self.distributed_tester.run_test(run_mismatched_empty_tensor_test)

    def test_single_empty_tensor(self):
        """Test all_reduce_coalesced with single empty tensor"""
        self.distributed_tester.run_test(run_single_empty_tensor_test)

    @pytest.mark.parametrize(
        "reduce_op",
        [
            dist.ReduceOp.SUM,
            dist.ReduceOp.AVG,
            dist.ReduceOp.MIN,
            dist.ReduceOp.MAX,
        ],
    )
    def test_empty_tensor_different_ops(self, reduce_op):
        """Test all_reduce_coalesced with empty tensors across different operations"""
        self.distributed_tester.run_test(run_empty_tensor_different_ops_test, reduce_op=reduce_op)

    @pytest.mark.xfail(reason="5GB tensor test expected to fail")
    def test_5gb_tensor(self):
        """Test all_reduce_coalesced with 5GB tensor."""
        self.distributed_tester.run_test(run_5gb_tensor_test)
