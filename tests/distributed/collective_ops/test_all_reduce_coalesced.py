import pytest
import torch
import torch.distributed as dist

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
