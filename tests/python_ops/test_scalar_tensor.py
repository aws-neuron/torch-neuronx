"""Tests for scalar_tensor operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestScalarTensor:
    """Test suite for scalar_tensor operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    def test_scalar_tensor_runs_on_neuron(self, device):
        """Test that scalar_tensor runs on Neuron"""
        with track_neuron_ops():
            result = torch.scalar_tensor(3.14, device=device)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize("scalar_value", [3.14, 42, 0, -5.5])
    def test_scalar_tensor_basic_values(self, device, scalar_value):
        """Test basic scalar_tensor with various values."""
        with track_neuron_ops():
            neuron_result = torch.scalar_tensor(scalar_value, device=device)
            cpu_result = torch.scalar_tensor(scalar_value)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == torch.Size([])  # 0-dimensional tensor
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize(
        "value,dtype",
        [
            (3.14, torch.float32),
            (42, torch.int32),
            (True, torch.bool),
            (2.5, torch.float16),
            (100, torch.int64),
        ],
    )
    def test_scalar_tensor_different_dtypes(self, device, value, dtype):
        """Test scalar_tensor with different data types."""
        with track_neuron_ops():
            result = torch.scalar_tensor(value, dtype=dtype, device=device)
            assert result.device.type == "neuron"
            assert result.dtype == dtype
            assert result.shape == torch.Size([])
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize("value", [True, False])
    def test_scalar_tensor_bool_values(self, device, value):
        """Test scalar_tensor with boolean values."""
        with track_neuron_ops():
            result = torch.scalar_tensor(value, device=device)
            assert result.device.type == "neuron"
            assert result.item() == value
            assert result.dtype == torch.float32
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize("value", [1e10, -1e10, 1e-10, -1e-10])
    def test_scalar_tensor_large_values(self, device, value):
        """Test scalar_tensor with large values."""
        with track_neuron_ops():
            result = torch.scalar_tensor(value, device=device)
            assert result.device.type == "neuron"
            assert abs(result.item() - value) < 1e-15
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_scalar_tensor_special_float_values(self, device, value):
        """Test scalar_tensor with special float values."""
        with track_neuron_ops():
            result = torch.scalar_tensor(value, device=device)
            assert result.device.type == "neuron"

            if torch.isnan(torch.tensor(value)):
                assert torch.isnan(result)
            else:
                assert result.item() == value
            assert_op_runs_on_neuron("aten::scalar_tensor")

    def test_scalar_tensor_with_requires_grad(self, device):
        """Test scalar_tensor with requires_grad=True."""
        with track_neuron_ops():
            result = torch.scalar_tensor(2.5, device=device, requires_grad=True)

            assert result.device.type == "neuron"
            assert result.requires_grad
            assert result.item() == 2.5
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize("value", [0, 1, -1, 3.14, -2.71, True, False])
    def test_scalar_tensor_cpu_comparison(self, device, value):
        """Test that neuron scalar_tensor matches CPU behavior."""
        cpu_result = torch.scalar_tensor(value)
        with track_neuron_ops():
            neuron_result = torch.scalar_tensor(value, device=device)

            assert neuron_result.shape == cpu_result.shape
            assert neuron_result.dtype == cpu_result.dtype

            # Compare values
            if torch.isnan(cpu_result):
                assert torch.isnan(neuron_result)
            else:
                torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::scalar_tensor")

    @pytest.mark.parametrize(
        "value,expected_dtype",
        [
            (42, torch.float32),  # Python int -> torch.int64
            (3.14, torch.float32),  # Python float -> torch.float32
            (True, torch.float32),  # Python bool -> torch.bool
        ],
    )
    def test_scalar_tensor_dtype_inference(self, device, value, expected_dtype):
        """Test that scalar_tensor correctly infers dtype."""
        with track_neuron_ops():
            result = torch.scalar_tensor(value, device=device)
            assert result.dtype == expected_dtype
            assert_op_runs_on_neuron("aten::scalar_tensor")

    def test_scalar_tensor_pin_memory(self, device):
        """Test scalar_tensor with pin_memory (should be ignored for neuron device)."""
        # pin_memory should be ignored for non-GPU devices
        with track_neuron_ops():
            result = torch.scalar_tensor(1.0, device=device, pin_memory=True)
            assert result.device.type == "neuron"
            assert result.item() == 1.0
            assert_op_runs_on_neuron("aten::scalar_tensor")
