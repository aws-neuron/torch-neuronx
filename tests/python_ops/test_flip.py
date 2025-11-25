import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestFlip:
    def test_flip_basic(self):
        """Test basic flip operation along a single dimension"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3, 4], device=device)
            x_cpu = torch.tensor([1, 2, 3, 4])

            result_neuron = torch.flip(x_neuron, [0])
            result_cpu = torch.flip(x_cpu, [0])

            assert result_neuron.device.type == "neuron"

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::flip")

    def test_flip_2d_single_dim(self):
        """Test flipping a 2D tensor along one dimension"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
            x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]])

            # Flip along dimension 0 (rows)
            result_neuron_dim0 = torch.flip(x_neuron, [0])
            result_cpu_dim0 = torch.flip(x_cpu, [0])
            torch.testing.assert_close(result_neuron_dim0.cpu(), result_cpu_dim0)

            # Flip along dimension 1 (columns)
            result_neuron_dim1 = torch.flip(x_neuron, [1])
            result_cpu_dim1 = torch.flip(x_cpu, [1])
            torch.testing.assert_close(result_neuron_dim1.cpu(), result_cpu_dim1)

            assert_op_runs_on_neuron("aten::flip")

    def test_flip_2d_multiple_dims(self):
        """Test flipping a 2D tensor along multiple dimensions"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
            x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]])

            result_neuron = torch.flip(x_neuron, [0, 1])
            result_cpu = torch.flip(x_cpu, [0, 1])

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::flip")

    # Test all possible dimension combinations
    @pytest.mark.parametrize("dims", [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
    def test_flip_3d_tensor(self, dims):
        """Test flipping a 3D tensor along various dimension combinations"""
        device = "neuron"
        with track_neuron_ops():
            # Initialize 3D tensor
            x_neuron = torch.arange(24).reshape(2, 3, 4).to(device)
            x_cpu = torch.arange(24).reshape(2, 3, 4)

            result_neuron = torch.flip(x_neuron, dims)
            result_cpu = torch.flip(x_cpu, dims)
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            assert_op_runs_on_neuron("aten::flip")

    def test_flip_empty_dims(self):
        """Test flipping with empty dims list (should return a copy)"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3, 4], device=device)
            x_cpu = torch.tensor([1, 2, 3, 4])

            result_neuron = torch.flip(x_neuron, [])
            result_cpu = torch.flip(x_cpu, [])

            torch.testing.assert_close(result_neuron.cpu(), x_cpu)
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::flip")

    def test_flip_negative_dims(self):
        """Test flipping with negative dimension indices"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
            x_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]])

            # Flip using negative indices (-1 refers to the last dimension)
            result_neuron = torch.flip(x_neuron, [-1])
            result_cpu = torch.flip(x_cpu, [-1])

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            torch.testing.assert_close(result_neuron.cpu(), torch.flip(x_cpu, [1]))

            assert_op_runs_on_neuron("aten::flip")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
    )
    def test_flip_different_dtypes(self, dtype):
        """Test flipping tensors with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randint(0, 10, (2, 3), device=device).to(dtype)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.flip(x_neuron, [0, 1])
            result_cpu = torch.flip(x_cpu, [0, 1])

            # Check dtype preservation
            assert result_neuron.dtype == dtype

            # Compare results
            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            assert_op_runs_on_neuron("aten::flip")

    # Negative tests
    @assert_raises(
        IndexError,
        match=r"Dimension out of range \(expected to be in range of \[.*?\], but got .*?\)",
    )
    def test_flip_invalid_dims_index_error(self):
        """Test flipping with invalid index dimensions"""
        device = "neuron"
        x = torch.tensor([1, 2, 3, 4], device=device)

        torch.flip(x, [1])

        torch.flip(x, [0, 1])

    @assert_raises(
        TypeError,
        match=(
            r"flip\(\): argument 'dims' \(position \d+\) must be tuple of ints,"
            r" but found element of type \w+ at pos \d+"
        ),
    )
    def test_flip_invalid_dims_type_error(self):
        """Test flipping with invalid type dimensions"""
        device = "neuron"
        x = torch.tensor([1, 2, 3, 4], device=device)
        torch.flip(x, [0.5])

    @assert_raises(
        RuntimeError,
        match=r"duplicate value in the list of dims",
        match_cpu=r"dim \d+ appears multiple times in the list of dims)",
    )
    def test_flip_repeated_dims(self):
        """Test flipping with repeated dimensions"""
        device = "neuron"

        with track_neuron_ops():
            x = torch.tensor([[1, 2], [3, 4]], device=device)

            torch.flip(x, [1, 1])

    def test_flip_requires_grad(self):
        """Test flipping tensors with requires_grad=True"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)
            x_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

            result_neuron = torch.flip(x_neuron, [0, 1])
            result_cpu = torch.flip(x_cpu, [0, 1])

            assert result_neuron.requires_grad
            assert result_cpu.requires_grad

            assert_op_runs_on_neuron("aten::flip")
