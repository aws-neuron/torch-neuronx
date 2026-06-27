"""Test kaiser_window .out variant operations.

The functional kaiser_window variants (aten::kaiser_window, .periodic, .beta)
work via CompositeImplicitAutograd decomposition into primitive ops that already
have neuron kernels. Only the .out variants need explicit registration because
the .out dispatch path bypasses composite decomposition. These tests verify the
.out registration produces correct results via CPU fallback.

Note: torch.i0 is not supported in Torch-MLIR so the lowering falls back to
CPU. Relaxed tolerance (atol=1e-4) is used because precision accumulates
through the Bessel function computation in the CPU fallback path.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestKaiserWindow:
    """Test kaiser_window .out variant operations."""

    @pytest.mark.parametrize("window_length", [5, 10, 64])
    def test_kaiser_window_out(self, window_length):
        """Test aten::kaiser_window.out produces correct result."""
        expected = torch.kaiser_window(window_length)

        out = torch.empty(window_length, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.kaiser_window.out(window_length, out=out)

        assert out.data_ptr() == result.data_ptr()
        # Relaxed tolerance: CPU fallback path accumulates precision error via i0
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_kaiser_window_periodic_out(self, periodic):
        """Test aten::kaiser_window.periodic_out produces correct result."""
        expected = torch.kaiser_window(10, periodic=periodic)

        out = torch.empty(10, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.kaiser_window.periodic_out(10, periodic, out=out)

        assert out.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("beta", [6.0, 12.0, 14.0])
    def test_kaiser_window_beta_out(self, beta):
        """Test aten::kaiser_window.beta_out produces correct result."""
        expected = torch.kaiser_window(10, periodic=True, beta=beta)

        out = torch.empty(10, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.kaiser_window.beta_out(10, True, beta, out=out)

        assert out.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
