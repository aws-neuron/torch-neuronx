"""Tests for _local_scalar_dense operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLocalScalarDense:
    """Test suite for _local_scalar_dense operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("scalar_value", [3.14, 42, 0, -5.5, 1e-10, 1e10])
    def test_local_scalar_dense_basic_values(self, device, scalar_value):
        """Test _local_scalar_dense with various values."""
        with track_neuron_ops():
            neuron_tensor = torch.scalar_tensor(scalar_value, device=device)
            cpu_tensor = torch.scalar_tensor(scalar_value)

            neuron_result = neuron_tensor.item()
            cpu_result = cpu_tensor.item()

            torch.testing.assert_close(neuron_result, cpu_result)
            assert_op_runs_on_neuron("aten::_local_scalar_dense")

    @pytest.mark.parametrize(
        "value,dtype",
        [
            (3.14, torch.float32),
            (42, torch.int32),
            (123, torch.int64),
            (True, torch.bool),
            (False, torch.bool),
            (2.5, torch.float64),
            (255, torch.uint8),
            (-128, torch.int8),
            (32767, torch.int16),
        ],
    )
    def test_local_scalar_dense_different_dtypes(self, device, value, dtype):
        """Test _local_scalar_dense with different data types."""
        with track_neuron_ops():
            tensor = torch.tensor(value, dtype=dtype, device=device)
            result = tensor.item()

            # Verify result type and value
            if dtype == torch.bool:
                assert result == bool(value)
                assert isinstance(result, bool)
            elif dtype in [torch.int32, torch.int64, torch.uint8, torch.int8, torch.int16]:
                assert result == int(value)
                assert isinstance(result, int)
            else:
                assert abs(result - float(value)) < 1e-6
                assert isinstance(result, float)

            assert_op_runs_on_neuron("aten::_local_scalar_dense")

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_local_scalar_dense_special_float_values(self, device, value):
        """Test _local_scalar_dense with special float values."""
        with track_neuron_ops():
            tensor = torch.tensor(value, device=device)
            result = tensor.item()
            assert result == value
            assert_op_runs_on_neuron("aten::_local_scalar_dense")

    def test_local_scalar_dense_nan_value(self, device):
        """Test _local_scalar_dense with NaN value."""
        with track_neuron_ops():
            tensor = torch.tensor(float("nan"), device=device)
            result = tensor.item()
            assert torch.isnan(torch.tensor(result))
            assert_op_runs_on_neuron("aten::_local_scalar_dense")

    @assert_raises(RuntimeError, match="a Tensor with 3 elements cannot be converted to Scalar")
    def test_local_scalar_dense_error_non_scalar(self, device):
        """Test that _local_scalar_dense raises error for non-scalar tensors."""
        # Create a non-scalar tensor
        tensor = torch.tensor([1, 2, 3], device=device)

        tensor.item()

    def test_local_scalar_dense_multi_dimension_tensor(self, device):
        """Test that _local_scalar_dense for multi-dimension tensors."""
        with track_neuron_ops():
            tensor = torch.tensor([[1]], device=device)
            tensor.item()
            assert_op_runs_on_neuron("aten::_local_scalar_dense")
