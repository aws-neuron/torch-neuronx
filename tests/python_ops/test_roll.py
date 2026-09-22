import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestRoll:
    """Test cases for aten::roll (on-device lowering, replaces the CPU fallback)."""

    @pytest.mark.parametrize(
        "shape,shifts,dims",
        [
            ((4, 8), 1, 1),  # single dim, positive shift
            ((4, 8), -3, 0),  # single dim, negative shift
            ((8,), 3, 0),  # 1D
            ((2, 3, 4), 2, 2),  # 3D
            ((64, 7, 7, 96), (-3, -3), (1, 2)),  # swin shifted-window (fwd)
            ((64, 7, 7, 96), (3, 3), (1, 2)),  # swin reverse roll (bwd shifts)
            ((3, 5), 4, ()),  # flatten roll (empty dims)
            ((4, 4), 5, 0),  # shift > dim size (wraps)
        ],
    )
    def test_roll_forward(self, shape, shifts, dims):
        """roll runs on Neuron and matches CPU torch.roll."""
        with track_neuron_ops():
            x_cpu = torch.randn(shape)
            x_neuron = x_cpu.detach().clone().to("neuron")

            ref = torch.roll(x_cpu, shifts, dims)
            out = torch.roll(x_neuron, shifts, dims)

            torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)
            assert_op_runs_on_neuron("aten::roll")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_roll_dtypes(self, dtype):
        """roll preserves values across dtypes on Neuron."""
        with track_neuron_ops():
            x_cpu = torch.randn(2, 7, 7, 96, dtype=dtype)
            x_neuron = x_cpu.detach().clone().to("neuron")

            ref = torch.roll(x_cpu, (-2, -2), (1, 2))
            out = torch.roll(x_neuron, (-2, -2), (1, 2))

            torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
            assert_op_runs_on_neuron("aten::roll")

    def test_roll_backward(self):
        """Gradient of roll (autograd expresses it as roll with negated shifts) runs
        on Neuron and matches CPU."""
        with track_neuron_ops():
            x_cpu = torch.randn(64, 7, 7, 96, requires_grad=True)
            x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

            y_cpu = torch.roll(x_cpu, (-3, -3), (1, 2))
            y_neuron = torch.roll(x_neuron, (-3, -3), (1, 2))

            g = torch.randn_like(y_cpu)
            y_cpu.backward(g)
            y_neuron.backward(g.to("neuron"))

            torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::roll")

    def test_roll_is_not_cpu_fallback(self):
        """Explicitly assert roll no longer offloads to CPU (the bug this fixes)."""
        with track_neuron_ops():
            x = torch.randn(8, 14, 14, 96).to("neuron")
            _ = torch.roll(x, (-3, -3), (1, 2))
            torch_neuronx.synchronize()
            assert "aten::roll" not in torch_neuronx.get_fallback_ops()
            assert not any(
                "neuron_to_cpu" in o for o in torch_neuronx.get_executed_ops()
            ), "roll round-tripped through CPU"
