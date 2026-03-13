"""Test that eye operation is properly registered with PyTorch dispatcher."""

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
from torch_neuronx.utils import use_mlir_aten_ops

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.skipif(not use_mlir_aten_ops(), reason="Eye operation requires MLIR ATEN ops")
class TestEyeRegistration:
    """Test eye operation registration and functionality."""

    @pytest.mark.parametrize(
        "n",
        [1, 2, 3, 5],
    )
    def test_eye_square_matrices(self, n):
        """Test eye operation for square matrices."""
        with track_neuron_ops():
            tensor = torch.eye(n, device="neuron")
            assert_op_runs_on_neuron("aten::eye")
            assert tensor.device.type == "neuron"
            assert tensor.shape == torch.Size((n, n))
            assert tensor.dtype == torch.get_default_dtype()
            # Check diagonal elements are 1 and off-diagonal are 0
            cpu_tensor = tensor.cpu()
            expected = torch.eye(n)
            assert torch.allclose(cpu_tensor, expected)

    @pytest.mark.parametrize(
        "n,m",
        [(1, 2), (2, 1), (3, 5), (5, 3)],
    )
    def test_eye_rectangular_matrices(self, n, m):
        """Test eye operation for rectangular matrices."""
        with track_neuron_ops():
            tensor = torch.eye(n, m, device="neuron")
            assert_op_runs_on_neuron("aten::eye")
            assert tensor.device.type == "neuron"
            assert tensor.shape == torch.Size((n, m))
            assert tensor.dtype == torch.get_default_dtype()
            # Check diagonal elements are 1 and off-diagonal are 0
            cpu_tensor = tensor.cpu()
            expected = torch.eye(n, m)
            assert torch.allclose(cpu_tensor, expected)

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
        ],
    )
    def test_eye_different_dtypes(self, dtype):
        """Test eye operation with different data types."""
        with track_neuron_ops():
            tensor = torch.eye(3, dtype=dtype, device="neuron")
            assert_op_runs_on_neuron("aten::eye")
            assert tensor.device.type == "neuron"
            assert tensor.dtype == dtype
            assert tensor.shape == torch.Size((3, 3))
            # Check diagonal elements are 1 and off-diagonal are 0
            cpu_tensor = tensor.cpu()
            expected = torch.eye(3, dtype=dtype)
            assert torch.allclose(cpu_tensor, expected)

    @pytest.mark.parametrize(
        "n,expected_exception",
        [
            (-1, RuntimeError),
            (1.1, TypeError),
            (None, TypeError),
        ],
    )
    @assert_raises((RuntimeError, TypeError))
    def test_eye_invalid_sizes(self, n, expected_exception):
        """Test that eye operation properly raises exceptions for invalid sizes."""
        _ = torch.eye(n, device="neuron")

    def test_eye_with_out_parameter(self):
        """Test eye operation with out parameter."""
        with track_neuron_ops():
            out = torch.empty((3, 3), device="neuron")
            tensor = torch.eye(3, device="neuron", out=out)
            op_name = "eye" if os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1" else "eye_"
            assert_op_runs_on_neuron(f"aten::{op_name}")
            assert tensor is out
            assert tensor.device.type == "neuron"
            assert tensor.shape == torch.Size((3, 3))
            # Check diagonal elements are 1 and off-diagonal are 0
            cpu_tensor = tensor.cpu()
            expected = torch.eye(3)
            assert torch.allclose(cpu_tensor, expected)
