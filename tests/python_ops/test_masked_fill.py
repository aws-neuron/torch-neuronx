"""Tests for masked_fill operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedFill:
    """Test suite for masked_fill operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_masked_fill_basic_shapes(self, device, shape):
        """Test masked_fill with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randint(0, 10, shape, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            value = 42.1
            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ],
    )
    @pytest.mark.parametrize("value", [42.1, -42.1, 0, 4, -4])
    def test_masked_fill_different_dtypes(self, device, dtype, value):
        """Test masked_fill with various tensor shapes."""
        shape = (3, 4)
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, shape, dtype=dtype, device=device)
            else:
                x = torch.randint(0, 10, shape, dtype=dtype, device=device)
            mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill")

    @pytest.mark.parametrize(
        "value,equal_nan",
        [
            (float("inf"), False),
            (float("-inf"), False),
            (float("nan"), True),
        ],
    )
    def test_masked_fill_special_values(self, device, value, equal_nan):
        """Test masked_fill with special values (inf, -inf, nan)."""
        with track_neuron_ops():
            x = torch.randn(3, device=device)
            mask = torch.tensor([True, False, True], device=device)

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=equal_nan)
            assert_op_runs_on_neuron("aten::masked_fill")

    @pytest.mark.parametrize(
        "tensor_shape,mask_shape",
        [
            ((4, 3), (1, 3)),  # Broadcast first dimension
            ((4, 3), (4, 1)),  # Broadcast second dimension
            ((2, 3, 4), (1, 3, 4)),  # Broadcast first dimension
            ((2, 3, 4), (2, 1, 4)),  # Broadcast middle dimension
            ((2, 3, 4), (2, 3, 1)),  # Broadcast last dimension
            ((2, 3, 4), (1, 1, 4)),  # Broadcast multiple dimensions
            ((2, 3, 4), (3, 4)),  # Broadcast with fewer dimensions
            ((2, 3, 4), (4,)),  # Broadcast with single dimension
        ],
    )
    def test_masked_fill_broadcast(self, device, tensor_shape, mask_shape):
        """Test masked_fill with broadcasting masks."""
        with track_neuron_ops():
            x = torch.randn(tensor_shape, device=device)
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device=device)
            value = 42.1

            neuron_result = x.masked_fill(mask, value)
            cpu_result = x.cpu().masked_fill(mask.cpu(), value)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_fill")
