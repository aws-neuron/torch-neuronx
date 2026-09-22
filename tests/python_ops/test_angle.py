"""Tests for angle (phase angle) operation.

torch.angle on a real input is 0 where x >= 0 (including +0.0, -0.0, +inf) and
pi where x < 0 (including -inf); nan maps to nan. Integer/bool inputs promote to
the default float dtype. Complex inputs are not supported on Neuron.
"""

import math

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAngle:
    """Test suite for aten::angle, aten::angle.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_angle_basic_shapes(self, device, shape):
        """Test angle with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.angle(x)
            cpu_result = torch.angle(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::angle")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
    def test_angle_float_dtypes(self, device, dtype):
        """Test angle with floating-point data types (0 for +, pi for -)."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dtype, device=device)
            neuron_result = torch.angle(x)
            cpu_result = torch.angle(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_positive_is_zero(self, device):
        """angle(x) == 0 for every non-negative x."""
        with track_neuron_ops():
            x = torch.tensor([0.0, 1e-6, 0.5, 1.0, 100.0, 1e10], device=device)
            neuron_result = torch.angle(x)
            assert torch.equal(neuron_result.cpu(), torch.zeros(6))
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_negative_is_pi(self, device):
        """angle(x) == pi for every negative x."""
        with track_neuron_ops():
            x = torch.tensor([-1e-6, -0.5, -1.0, -100.0, -1e10], device=device)
            neuron_result = torch.angle(x)
            torch.testing.assert_close(
                neuron_result.cpu(), torch.full((5,), math.pi, dtype=torch.float32)
            )
            assert_op_runs_on_neuron("aten::angle")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint8, torch.bool])
    def test_angle_integer_promotes_to_float(self, device, dtype):
        """Integer and bool inputs promote to the default float dtype."""
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.tensor([True, False], dtype=dtype, device=device)
            elif dtype == torch.uint8:
                x = torch.tensor([0, 1, 255], dtype=dtype, device=device)
            else:
                x = torch.tensor([3, -3, 0], dtype=dtype, device=device)
            neuron_result = torch.angle(x)
            cpu_result = torch.angle(x.cpu())

            assert neuron_result.dtype == torch.get_default_dtype()
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_out_variant(self, device):
        """Test angle .out variant writes to a pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0], device=device)
            out = torch.empty_like(x)
            result = torch.angle(x, out=out)

            expected = torch.angle(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_out_dtype_mismatch_raises(self, device):
        """A non-float out= dtype for a float op must raise RuntimeError."""
        x = torch.tensor([0.0, 1.0], device=device)
        out = torch.empty(2, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.angle(x, out=out)

    def test_angle_special_values(self, device):
        """Test angle with +inf, -inf, nan, and signed zero.

        angle(+inf)=0, angle(-inf)=pi, angle(nan)=nan, angle(-0.0)=0.
        """
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0, -0.0], device=device)
            neuron_result = torch.angle(x)
            cpu_result = torch.angle(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_compile(self, device):
        """angle works under torch.compile(backend="neuron").

        The decomposition lives in the shared table (decompositions.py), so the
        compile path lowers angle the same as eager — this guards that parity.
        """
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=device)
        compiled = torch.compile(torch.angle, backend="neuron")
        neuron_result = compiled(x)
        cpu_result = torch.angle(x.cpu())
        torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_angle_empty_tensor(self, device):
        """Test angle with an empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.angle(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::angle")

    def test_angle_single_element(self, device):
        """Test angle with a single-element tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0], device=device)
            neuron_result = torch.angle(x)
            cpu_result = torch.angle(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::angle")
