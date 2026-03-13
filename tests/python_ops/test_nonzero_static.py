"""Tests for nonzero_static operation."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonzeroStatic:
    """Test suite for nonzero_static operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_nonzero_static_basic_shapes(self, device, shape):
        """Test nonzero_static with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randint(0, 2, shape, device=device)
            size = torch.sum(x).item()
            neuron_result = torch.nonzero_static(x, size=size)
            cpu_result = torch.nonzero_static(x.cpu(), size=size)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero_static")

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    def test_nonzero_static_all_zeros(self, device):
        """Test nonzero_static with all zeros tensor."""
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            neuron_result = torch.nonzero_static(x, size=0)
            cpu_result = torch.nonzero_static(x.cpu(), size=0)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::_to_copy")
            assert_op_did_not_run_on_neuron("aten::nonzero_static")

    def test_nonzero_static_all_ones(self, device):
        """Test nonzero_static with all ones tensor."""
        with track_neuron_ops():
            x = torch.ones(2, 3, device=device)
            size = x.numel()
            neuron_result = torch.nonzero_static(x, size=size)
            cpu_result = torch.nonzero_static(x.cpu(), size=size)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero_static")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
    def test_nonzero_static_different_dtypes(self, device, dtype):
        """Test nonzero_static with different data types."""
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.randint(0, 2, (3, 4), device=device).bool()
            else:
                x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
            size = torch.sum(x, dtype=torch.int).item()
            neuron_result = torch.nonzero_static(x, size=size)
            cpu_result = torch.nonzero_static(x.cpu(), size=size)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero_static")

    def test_nonzero_static_oversized(self, device):
        """Test nonzero_static with size larger than actual nonzeros."""
        with track_neuron_ops():
            x = torch.tensor([[1, 0], [0, 1]], device=device)
            size = 5  # Larger than actual nonzeros (2)
            neuron_result = torch.nonzero_static(x, size=size)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape[0] == size
            cpu_result = torch.nonzero_static(x.cpu(), size=size)
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero_static")

    def test_nonzero_static_with_out(self, device):
        """Test nonzero_static with out parameter."""
        with track_neuron_ops():
            x = torch.randint(0, 2, (3, 4), device=device)
            size = torch.sum(x).item()
            out = torch.empty((size, x.ndim), dtype=torch.long, device=device)
            result = torch.nonzero_static(x, size=size, out=out)
            cpu_result = torch.nonzero_static(x.cpu(), size=size)

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::nonzero_static")

    def test_nonzero_static_with_fill_value_scalar(self, device):
        """Test nonzero_static with scalar fill_value."""
        with track_neuron_ops():
            x = torch.tensor([[1, 0], [0, 1]], device=device)
            size = 5
            fill_value = -5
            neuron_result = torch.nonzero_static(x, size=size, fill_value=fill_value)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == (size, x.ndim)
            # Check that padding values are filled with fill_value
            assert torch.all(neuron_result[2:] == fill_value)
            assert_op_runs_on_neuron("aten::nonzero_static")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
    def test_nonzero_with_out_wrong_dtype(self, device, dtype):
        """Test nonzero with out parameter of wrong dtype."""
        with pytest.raises(RuntimeError):
            x = torch.randint(0, 2, (3, 4), device=device)
            size = 5
            out = torch.empty((size, 2), dtype=dtype, device=device)
            torch.nonzero_static(x, size=size, out=out)
