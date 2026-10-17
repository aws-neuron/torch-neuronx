"""Test isnan operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIsnan:
    """Test isnan operation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_isnan_basic(self, dtype):
        """Test basic isnan functionality."""
        x_cpu = torch.tensor([1.0, float("nan"), 3.0, float("nan"), 0.0], dtype=dtype)
        expected = torch.isnan(x_cpu)

        x_neuron = x_cpu.to("neuron")

        with track_neuron_ops():
            result = torch.isnan(x_neuron)
            assert_op_runs_on_neuron("aten::isnan")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("shape", [(4,), (4, 8), (2, 3, 4)])
    def test_isnan_shapes(self, shape):
        """Test isnan with various shapes."""
        x_cpu = torch.randn(shape)
        x_cpu.view(-1)[0] = float("nan")
        expected = torch.isnan(x_cpu)

        x_neuron = x_cpu.to("neuron")

        with track_neuron_ops():
            result = torch.isnan(x_neuron)
            assert_op_runs_on_neuron("aten::isnan")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_isnan_out_variant(self, dtype):
        """Test aten::isnan.out writes correct result to pre-allocated tensor."""
        x_cpu = torch.tensor([1.0, float("nan"), 3.0, float("nan")], dtype=dtype)
        expected = torch.isnan(x_cpu)

        x_neuron = x_cpu.to("neuron")
        out = torch.empty(4, dtype=torch.bool, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.isnan.out(x_neuron, out=out)
            assert_op_runs_on_neuron("aten::isnan")

        assert out.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out.cpu(), expected)
