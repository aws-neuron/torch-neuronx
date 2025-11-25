"""Tests for masked_select operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaskedSelect:
    """Test suite for masked_select operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_masked_select_basic_shapes(self, device, shape):
        """Test masked_select with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            mask = torch.randint(0, 2, shape, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_all_false_mask(self, device):
        """Test masked_select with all False mask."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.zeros(3, 4, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            # Op will be short-circuited with all false mask
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_masked_select_all_true_mask(self, device):
        """Test masked_select with all True mask."""
        with track_neuron_ops():
            x = torch.randn(2, 3, device=device)
            mask = torch.ones(2, 3, device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            pytest.param(torch.float64, marks=pytest.mark.xfail(reason="float64 not supported")),
            torch.float16,
            torch.bfloat16,
            torch.int32,
            pytest.param(torch.int64, marks=pytest.mark.xfail(reason="int64 not supported")),
            torch.bool,
        ],
    )
    def test_masked_select_different_dtypes(self, device, dtype):
        """Test masked_select with different data types."""
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, (3, 4), device=device).bool()
            elif dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                x = torch.randn(3, 4, dtype=dtype, device=device)
            else:
                x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_broadcast_mask(self, device):
        """Test masked_select with broadcastable mask."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.tensor([True, False, True, False], device=device)
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_broadcast_input(self, device):
        """Test masked_select with broadcastable input tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_single_element(self, device):
        """Test masked_select with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            mask = torch.tensor([True], device=device)
            neuron_result = torch.masked_select(x, mask)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")

    def test_masked_select_with_out(self, device):
        """Test masked_select with out parameter."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            mask = torch.randint(0, 2, (3, 4), device=device).bool()
            # Get expected shape first
            expected_shape = torch.masked_select(x.cpu(), mask.cpu()).shape
            out = torch.empty(expected_shape, dtype=x.dtype, device=device)
            result = torch.masked_select(x, mask, out=out)
            cpu_result = torch.masked_select(x.cpu(), mask.cpu())

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::masked_select")
