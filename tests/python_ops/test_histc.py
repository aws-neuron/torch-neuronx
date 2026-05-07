"""Tests for histc operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestHistc:
    """Test suite for histc operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(50,), (30, 40), (20, 30, 40)])
    def test_histc_basic_shapes(self, device, shape):
        """Test histc with various tensor shapes."""
        torch.manual_seed(123)
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.histc(x, bins=10)
            cpu_result = torch.histc(x.cpu(), bins=10)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    @pytest.mark.parametrize("bins", [5, 10, 20, 50])
    def test_histc_different_bins(self, device, bins):
        """Test histc with different number of bins."""
        with track_neuron_ops():
            x = torch.randn(100, device=device)
            neuron_result = torch.histc(x, bins=bins)
            cpu_result = torch.histc(x.cpu(), bins=bins)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == (bins,)
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            # Small value to trigger OOB when tensor value is greater than range
            (-4.0, -2.0),
            (0, 0),
            (100, 200),
        ],
    )
    def test_histc_with_range(self, device, min_val, max_val):
        """Test histc with specified min and max range."""
        with track_neuron_ops():
            x = torch.randn(100, device=device)
            neuron_result = torch.histc(x, bins=10, min=min_val, max=max_val)
            cpu_result = torch.histc(x.cpu(), bins=10, min=min_val, max=max_val)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_histc_different_dtypes(self, device, dtype):
        """Test histc with different data types.

        Note that CPU does not support Int and Long for this op"""
        with track_neuron_ops():
            x = torch.randint(0, 10, (30, 40), dtype=dtype, device=device)
            neuron_result = torch.histc(x, bins=10)
            cpu_result = torch.histc(x.cpu(), bins=10)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    def test_histc_out(self, device):
        """Test histc with different number of bins."""
        with track_neuron_ops():
            x = torch.randn(100, device=device)
            out = torch.zeros(10, device=device)
            neuron_result = torch.histc(x, bins=10, out=out)
            cpu_result = torch.histc(x.cpu(), bins=10, out=out.cpu())

            assert neuron_result is out
            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    @assert_raises(
        RuntimeError,
        match="torch.histogram: input tensor and hist tensor should have the same dtype, "
        "but got input float and hist int",
    )
    def test_histc_out_inconsistent_dtype_cpu(self, device):
        """Test histc with inconsistent dtype on CPU device to ensure same behavior."""
        x = torch.randn(100, device=device)
        out = torch.zeros(10, device=device, dtype=torch.int32)
        torch.histc(x.cpu(), bins=10, out=out.cpu())

    def test_histc_single_element(self, device):
        """Test histc with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            neuron_result = torch.histc(x, bins=5)
            cpu_result = torch.histc(x.cpu(), bins=5)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")

    def test_histc_empty_tensor(self, device):
        """Test histc with empty tensor."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.histc(x, bins=5)
            cpu_result = torch.histc(x.cpu(), bins=5)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")
