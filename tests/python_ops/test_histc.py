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

    def test_histc_auto_range_stale_neff_regression(self, device):
        """Two auto-range histc calls with the same input shape but different data
        ranges must not share a cached NEFF.

        min=max=0 means "derive the range from the data", so the bin edges are
        baked into the NEFF from x.min()/x.max(). The raw dispatch args are
        identical for both calls (same-shape float32, bins=10, min=0, max=0), so
        the lowering must key on the preprocessed range; otherwise the second
        call buckets its data against the first call's edges.
        """
        with track_neuron_ops():
            # First call: data spans [2.0, 7.0] -> edges derived from that range.
            x_a = torch.linspace(2.0, 7.0, 100, device=device)
            result_a = torch.histc(x_a, bins=10)
            torch.testing.assert_close(result_a.cpu(), torch.histc(x_a.cpu(), bins=10))

            # Second call: same shape/dtype, data spans [1.0, 9.0] -> different
            # edges. Stale edges would put the counts in the wrong buckets.
            x_b = torch.linspace(1.0, 9.0, 100, device=device)
            result_b = torch.histc(x_b, bins=10)
            torch.testing.assert_close(result_b.cpu(), torch.histc(x_b.cpu(), bins=10))
            assert_op_runs_on_neuron("aten::histc")

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_histc_different_dtypes(self, device, dtype):
        """Test histc with different data types.

        Note that CPU does not support Int and Long for this op"""
        with track_neuron_ops():
            x = torch.randint(0, 10, (30, 40), dtype=dtype).to(device)
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

    @pytest.mark.parametrize("min_max", [5, 0, -3, 100])
    def test_histc_equal_min_max_auto_ranges(self, device, min_max):
        """histc with explicit min==max auto-ranges to [input.min(), input.max()]."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2, 1], device=device)
            neuron_result = torch.histc(x, bins=4, min=min_max, max=min_max)
            cpu_result = torch.histc(x.cpu(), bins=4, min=min_max, max=min_max)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::histc")
