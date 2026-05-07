"""Test that zeros operation is properly registered with PyTorch dispatcher."""

import logging
import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestZerosRegistration:
    """Test zeros operation registration and functionality."""

    @pytest.mark.parametrize(
        "size",
        [
            0,
            1,
            (0,),
            (1,),
            (2, 2),
            (3, 3, 3),
            (2, 0, 2),
        ],
    )
    def test_zeros_of_different_sizes(self, size):
        """Test that zeros operation is properly registered with PyTorch dispatcher."""
        with track_neuron_ops():
            tensor = torch.zeros(size, device="neuron")
            assert_op_runs_on_neuron("aten::zeros")
            assert tensor.device.type == "neuron"
            if isinstance(size, int):
                size = (size,)
            assert tensor.shape == torch.Size(size)
            assert tensor.dtype == torch.get_default_dtype()
            assert (tensor == 0).all()

    def test_zeros_scalar_value(self):
        with track_neuron_ops():
            x = torch.zeros((), device="neuron:0")
            assert x.device.type == "neuron"
            assert x.dim() == 0
            assert x.to("cpu").item() == 0
            assert_op_runs_on_neuron("aten::zeros")

    @pytest.mark.parametrize(
        "size,expected_exception",
        [
            (-1, RuntimeError),
            (1.1, TypeError),
            (-1.1, TypeError),
            (None, TypeError),
        ],
    )
    @assert_raises((RuntimeError, TypeError))
    def test_zeros_of_invalid_sizes(self, size, expected_exception):
        """Test that zeros operation properly raises exceptions for invalid sizes."""
        _ = torch.zeros(size, device="neuron")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
            None,
        ],
    )
    def test_zeros_of_different_dtypes(self, dtype):
        """Test that zeros operation is properly registered with PyTorch dispatcher."""
        with track_neuron_ops():
            tensor = torch.zeros((1, 2, 3), dtype=dtype, device="neuron")
            assert_op_runs_on_neuron("aten::zeros")
            assert tensor.device.type == "neuron"
            if dtype is None:
                dtype = torch.get_default_dtype()
            assert tensor.dtype == dtype
            assert tensor.shape == torch.Size((1, 2, 3))
            assert (tensor == 0).all()

    def test_zeros_with_backward(self):
        """Test zeros with backward pass."""
        with track_neuron_ops():
            # Create tensor that requires grad
            result = torch.zeros((2, 3), device="neuron", requires_grad=True)
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
        "size",
        [
            0,
            1,
            (0,),
            (1,),
            (2, 2),
            (3, 3, 3),
            (2, 0, 2),
        ],
    )
    def test_zeros_out_of_different_sizes(self, size):
        """Test that zeros operation is properly registered with PyTorch dispatcher."""
        with track_neuron_ops():
            out = torch.empty(size, device="neuron")
            tensor = torch.zeros(size, device="neuron", out=out)
            op_name = "zeros" if os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1" else "zero_"
            assert_op_runs_on_neuron(f"aten::{op_name}")
            assert tensor is out
            assert tensor.device.type == "neuron"
            if isinstance(size, int):
                size = (size,)
            assert tensor.shape == torch.Size(size)
            assert (tensor == 0).all()
