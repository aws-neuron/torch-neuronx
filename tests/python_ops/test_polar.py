"""Tests for polar (complex construction) operation.

torch.polar(abs, angle) builds a complex tensor: abs * (cos(angle) + i*sin(angle)).
The inputs are real; the output is complex. Neuron has no complex polar, so the
shared decomposition rewrites polar into real cos/sin math plus one
view_as_complex. This suite exercises the eager (make_fx) path.

Angles stay in (-pi, pi) to keep fp32 cos/sin accurate.
"""

import math

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPolar:
    """Test suite for aten::polar, aten::polar.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_polar_basic_shapes(self, device, shape):
        """Test polar with various tensor shapes."""
        with track_neuron_ops():
            torch.manual_seed(0)
            abs_ = torch.rand(shape, device=device)
            angle = (torch.rand(shape, device=device) * 2 - 1) * math.pi
            neuron_result = torch.polar(abs_, angle)
            cpu_result = torch.polar(abs_.cpu(), angle.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == torch.complex64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5)
            assert_op_runs_on_neuron("aten::polar")

    def test_polar_float32_gives_complex64(self, device):
        """polar on float32 inputs runs on Neuron and gives complex64.

        Neuron has no float64. torch.polar on float64 falls back to CPU and gives
        complex128, so this test covers the Neuron-native float32 path only.
        """
        with track_neuron_ops():
            abs_ = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32, device=device)
            angle = torch.tensor([0.0, 0.5, -0.5, 1.0], dtype=torch.float32, device=device)
            neuron_result = torch.polar(abs_, angle)
            cpu_result = torch.polar(abs_.cpu(), angle.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == torch.complex64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5)
            assert_op_runs_on_neuron("aten::polar")

    def test_polar_zero_angle_is_real(self, device):
        """polar(abs, 0) == abs + 0i for every abs."""
        with track_neuron_ops():
            abs_ = torch.tensor([0.0, 0.5, 1.0, 100.0], device=device)
            angle = torch.zeros(4, device=device)
            neuron_result = torch.polar(abs_, angle)
            expected = torch.complex(abs_.cpu(), torch.zeros(4))
            torch.testing.assert_close(neuron_result.cpu(), expected, rtol=1e-4, atol=1e-5)
            assert_op_runs_on_neuron("aten::polar")

    def test_polar_zero_abs_is_zero(self, device):
        """polar(0, angle) == 0 for every angle."""
        with track_neuron_ops():
            abs_ = torch.zeros(4, device=device)
            angle = torch.tensor([-1.0, 0.0, 1.0, 3.0], device=device)
            neuron_result = torch.polar(abs_, angle)
            torch.testing.assert_close(
                neuron_result.cpu(), torch.zeros(4, dtype=torch.complex64), atol=1e-6, rtol=0
            )
            assert_op_runs_on_neuron("aten::polar")

    def test_polar_out_variant(self, device):
        """Test polar .out variant writes into a pre-allocated complex tensor.

        The out= tensor is complex64. The Neuron copy-back path does not support a
        complex out= buffer, so polar.out routes through CPU fallback. The result
        must still be correct and must alias the supplied out tensor.
        """
        abs_ = torch.tensor([0.5, 1.0, 2.0], device=device)
        angle = torch.tensor([0.5, -0.5, 1.0], device=device)
        out = torch.empty(3, dtype=torch.complex64, device=device)
        result = torch.polar(abs_, angle, out=out)

        expected = torch.polar(abs_.cpu(), angle.cpu())
        assert out.data_ptr() == result.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-5)

    def test_polar_compile(self, device):
        """polar works under torch.compile(backend="neuron").

        The decomposition lives in the shared table (decompositions.py), so the
        compile path lowers polar the same as eager — this guards that parity.
        """
        abs_ = torch.tensor([0.5, 1.0, 2.0], device=device)
        angle = torch.tensor([0.5, -0.5, 1.0], device=device)
        compiled = torch.compile(torch.polar, backend="neuron")
        neuron_result = compiled(abs_, angle)
        cpu_result = torch.polar(abs_.cpu(), angle.cpu())
        torch.testing.assert_close(neuron_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5)

    def test_polar_empty_tensor(self, device):
        """Test polar with an empty tensor (numel=0)."""
        with track_neuron_ops():
            abs_ = torch.tensor([], device=device)
            angle = torch.tensor([], device=device)
            neuron_result = torch.polar(abs_, angle)
            assert neuron_result.numel() == 0
            assert neuron_result.dtype == torch.complex64
            assert_op_runs_on_neuron("aten::polar")

    def test_polar_single_element(self, device):
        """Test polar with a single-element tensor."""
        with track_neuron_ops():
            abs_ = torch.tensor([2.0], device=device)
            angle = torch.tensor([1.0], device=device)
            neuron_result = torch.polar(abs_, angle)
            cpu_result = torch.polar(abs_.cpu(), angle.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, rtol=1e-4, atol=1e-5)
            assert_op_runs_on_neuron("aten::polar")
