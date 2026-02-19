"""Test that full operation is properly registered with PyTorch dispatcher."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_does_not_run,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestFull:
    def test_full_basic(self):
        """Test basic full functionality."""
        with track_neuron_ops():
            result = torch.full((2, 3), 5.0, device="neuron")
            assert result.device.type == "neuron"
            assert result.shape == torch.Size([2, 3])
            assert torch.all(result == 5.0)
            assert_op_runs_on_neuron("aten::full")

    @pytest.mark.parametrize(
        "shape, fill_value",
        [
            # Scalar tensor
            ((), 42.0),
            # 1D tensor case
            ((5,), 3.14),
            # 2D tensor case
            ((3, 4), -2.5),
            # 3D tensor case
            ((2, 3, 4), 0.0),
            # 4D tensor case
            ((2, 1, 4, 5), 1.0),
        ],
        ids=["scalar", "1d_tensor", "2d_tensor", "3d_tensor", "4d_tensor"],
    )
    def test_full_different_shapes(self, shape, fill_value):
        """Test full on tensors with different shapes."""
        expected_cpu = torch.full(shape, fill_value)
        with track_neuron_ops():
            result = torch.full(shape, fill_value, device="neuron")
            assert result.shape == torch.Size(shape)
            assert torch.all(result == fill_value)
            assert torch.all(result.cpu() == expected_cpu)
            assert_op_runs_on_neuron("aten::full")

    def test_full_with_backward(self):
        """Test full with backward pass."""
        with track_neuron_ops():
            # Create tensor that requires grad
            result = torch.full((2, 3), 4.0, device="neuron", requires_grad=True)
            loss = result.sum()
            loss.backward()

            # result is a leaf tensor, so it should have grad_fn=None but requires_grad=True
            assert result.grad_fn is None  # Leaf tensors don't have grad_fn
            assert result.requires_grad is True
            # loss should have grad_fn since it's result of sum operation
            assert loss.grad_fn is not None
            # Gradient should be all ones for sum operation
            assert torch.all(result.grad == 1.0)

    @pytest.mark.parametrize(
        "dtype, fill_value",
        [
            (torch.float64, 3.14),
            (torch.float32, 3.14),
            (torch.float16, 2.5),
            (torch.bfloat16, 1.5),
            (torch.int8, 127),
            (torch.int16, 32767),
            (torch.int32, 42),
            (torch.int64, 100),
            (torch.uint8, 255),
            (torch.bool, False),
        ],
        ids=[
            "float64",
            "float32",
            "float16",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        ],
    )
    def test_full_dtypes(self, dtype, fill_value):
        """Test full with different dtypes."""
        with track_neuron_ops():
            result = torch.full((2, 3), fill_value, dtype=dtype, device="neuron")
            assert result.dtype == dtype
            assert result.shape == torch.Size([2, 3])
            assert torch.all(result == fill_value)
            assert_op_runs_on_neuron("aten::full")

    @pytest.mark.parametrize(
        "fill_value",
        [
            1e-6,  # Very small positive value
            1e6,  # Very large value
            1e-10,  # Extremely small positive value
            1e10,  # Extremely large value
        ],
        ids=["small_positive", "large_positive", "extremely_small", "extremely_large"],
    )
    def test_full_edge_case_values(self, fill_value):
        """Test full with edge case values."""
        with track_neuron_ops():
            result = torch.full((2, 2), fill_value, device="neuron")
            assert torch.all(result == fill_value)
            assert_op_runs_on_neuron("aten::full")

    def test_full_with_out_parameter(self):
        """Test full with out parameter."""
        with track_neuron_ops():
            # Create output tensor
            out = torch.empty((2, 3), device="neuron")
            result = torch.full((2, 3), 5.0, device="neuron", out=out)

            # Result should be the same as out tensor
            assert result is out
            assert torch.all(result == 5.0)
            assert_op_runs_on_neuron("aten::full.out")

    @pytest.mark.parametrize(
        "fill_value",
        [
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
        ],
        ids=["inf", "-inf", "zero", "-zero"],
    )
    def test_full_special_float_values(self, fill_value):
        """Test full with special float values."""
        with track_neuron_ops():
            result = torch.full((2, 2), fill_value, device="neuron")
            if fill_value == float("inf"):
                assert torch.all(torch.isinf(result))
                assert torch.all(result > 0)
            elif fill_value == float("-inf"):
                assert torch.all(torch.isinf(result))
                assert torch.all(result < 0)
            else:
                assert torch.all(result == fill_value)
            assert_op_runs_on_neuron("aten::full")

    def test_full_empty_tensor(self):
        """Test full on tensor with empty dimensions matches CPU behavior."""
        # Check what PyTorch does with empty tensors
        shape = (2, 0, 3)
        fill_value = 5.0
        expected_cpu = torch.full(shape, fill_value)

        # Now test on neuron
        with track_neuron_ops():
            result = torch.full(shape, fill_value, device="neuron")
            assert result.shape == expected_cpu.shape
            assert result.numel() == 0
            assert_op_runs_on_neuron("aten::full")

    def test_full_layout_strided(self):
        """Test full with strided layout (default)."""
        with track_neuron_ops():
            result = torch.full((3, 4), 2.0, device="neuron", layout=torch.strided)
            assert result.shape == torch.Size([3, 4])
            assert torch.all(result == 2.0)
            assert result.layout == torch.strided
            assert_op_runs_on_neuron("aten::full")
