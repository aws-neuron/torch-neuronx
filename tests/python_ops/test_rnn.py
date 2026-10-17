"""Tests for the fused LSTM cell (``aten::_thnn_fused_lstm_cell``) on Neuron.

The fused cell has no CPU kernel, so once torch-neuronx makes it available for
PrivateUse1, ``aten::lstm`` selects the fused path on the neuron device. These
tests verify the eager lowering (via the shared decomposition) runs the fused
cell natively on Neuron — forward and backward — instead of hitting the CPU
fallback (which lacks the op).
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestFusedLSTMCell:
    def setup_method(self):
        torch.manual_seed(42)

    def test_lstm_forward_matches_cpu(self):
        """nn.LSTM forward (eval) runs the fused cell on Neuron and matches CPU."""
        lstm = torch.nn.LSTM(4, 5, num_layers=2).eval()
        x = torch.randn(3, 2, 4)
        y_cpu, _ = lstm(x)

        with track_neuron_ops():
            y_neuron, _ = lstm.to("neuron")(x.to("neuron"))
            assert_op_runs_on_neuron("aten::_thnn_fused_lstm_cell")

        torch.testing.assert_close(y_neuron.cpu(), y_cpu)

    def test_lstm_backward_matches_cpu(self):
        """nn.LSTM backward runs the fused cell backward on Neuron and matches CPU."""
        lstm_cpu = torch.nn.LSTM(4, 5).train()
        x = torch.randn(3, 2, 4, requires_grad=True)
        out_cpu, _ = lstm_cpu(x)
        out_cpu.sum().backward()
        grad_cpu = x.grad.clone()

        lstm_neuron = torch.nn.LSTM(4, 5).train()
        lstm_neuron.load_state_dict(lstm_cpu.state_dict())
        lstm_neuron = lstm_neuron.to("neuron")
        x_neuron = x.detach().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            out_neuron, _ = lstm_neuron(x_neuron)
            out_neuron.sum().backward()
            assert_op_runs_on_neuron("aten::_thnn_fused_lstm_cell")

        torch.testing.assert_close(x_neuron.grad.cpu(), grad_cpu)
