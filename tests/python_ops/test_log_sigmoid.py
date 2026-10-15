"""Tests for aten::log_sigmoid_forward on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SHAPES = [(8,), (4, 8), (2, 3, 4), (2, 3, 4, 5)]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLogSigmoidForward:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_basic(self, shape, dtype):
        """Output and buffer match CPU reference."""
        x = torch.randn(*shape, dtype=dtype)
        x_neuron = x.to("neuron")

        with track_neuron_ops():
            out, buf = torch.ops.aten.log_sigmoid_forward(x_neuron)
            assert_op_runs_on_neuron("aten::log_sigmoid_forward")

        expected_out, expected_buf = torch.ops.aten.log_sigmoid_forward(x)
        torch.testing.assert_close(out.cpu(), expected_out, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(buf.cpu(), expected_buf, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_nn_module(self, shape, dtype):
        """torch.nn.LogSigmoid module produces same result as aten op."""
        x = torch.randn(*shape, dtype=dtype)
        x_neuron = x.to("neuron")

        with track_neuron_ops():
            out = torch.nn.LogSigmoid()(x_neuron)
            assert_op_runs_on_neuron("aten::log_sigmoid_forward")

        torch.testing.assert_close(out.cpu(), torch.nn.LogSigmoid()(x), rtol=1e-3, atol=1e-2)

    def test_output_range(self):
        """log_sigmoid output must be in (-inf, 0] and buffer in (0, 1]."""
        x = torch.randn(100).to("neuron")
        with track_neuron_ops():
            out, buf = torch.ops.aten.log_sigmoid_forward(x)
            assert_op_runs_on_neuron("aten::log_sigmoid_forward")
        assert (out.cpu() <= 0).all()
        assert (buf.cpu() > 0).all()
        assert (buf.cpu() <= 1).all()

    def test_large_positive(self):
        """Numerically stable for large positive inputs (no -inf)."""
        x = torch.tensor([100.0]).to("neuron")
        with track_neuron_ops():
            out, buf = torch.ops.aten.log_sigmoid_forward(x)
            assert_op_runs_on_neuron("aten::log_sigmoid_forward")
        assert torch.isfinite(out.cpu()).all()
        assert torch.isfinite(buf.cpu()).all()

    def test_large_negative(self):
        """Numerically stable for large negative inputs (no overflow)."""
        x = torch.tensor([-100.0]).to("neuron")
        with track_neuron_ops():
            out, buf = torch.ops.aten.log_sigmoid_forward(x)
            assert_op_runs_on_neuron("aten::log_sigmoid_forward")
        assert torch.isfinite(out.cpu()).all()
        assert torch.isfinite(buf.cpu()).all()

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_log_sigmoid_forward_output_variant(self, shape):
        """Test aten::log_sigmoid_forward.output dispatches to Neuron."""
        torch.manual_seed(42)
        x = torch.randn(shape, device="neuron")
        output = torch.empty(shape, device="neuron")
        buffer = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.log_sigmoid_forward.output(x, output=output, buffer=buffer)
        assert_op_runs_on_neuron("aten::log_sigmoid_forward")

        expected_out, expected_buf = torch.ops.aten.log_sigmoid_forward(x)
        torch.testing.assert_close(result[0].cpu(), expected_out.cpu())
        torch.testing.assert_close(output.cpu(), expected_out.cpu())
