"""Tests for sign operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSign:
    """Test suite for sign operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_sign_basic_shapes(self, device, shape):
        """Test sign with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.sign(x)
            cpu_result = torch.sign(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sign")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_sign_different_dtypes(self, device, dtype):
        """Test sign with different data types."""
        with track_neuron_ops():
            x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)
            neuron_result = torch.sign(x)
            cpu_result = torch.sign(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sign")

    def test_sign_single_element(self, device):
        """Test sign with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([5.0], device=device)
            neuron_result = torch.sign(x)
            cpu_result = torch.sign(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sign")

    @pytest.mark.xfail(reason="Neuron returns nan for nan input while CPU return 0 for nan")
    def test_sign_special_values(self, device):
        """Test sign with special values (positive, negative, zero, inf, -inf, nan)."""
        with track_neuron_ops():
            x = torch.tensor(
                [-3.14, -1.0, 0.0, 1.0, 3.14, float("inf"), float("-inf"), float("nan")],
                device=device,
            )
            neuron_result = torch.sign(x)
            cpu_result = torch.sign(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::sign")
