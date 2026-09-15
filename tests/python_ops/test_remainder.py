"""Tests for remainder operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestRemainder:
    """Test suite for remainder operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_remainder_basic_shapes(self, device, shape):
        """Test remainder with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device) * 10
            y = torch.randn(shape, device=device) * 5
            neuron_result = torch.remainder(x, y)
            cpu_result = torch.remainder(x.cpu(), y.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::remainder")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            pytest.param(torch.float16, marks=pytest.mark.xfail(reason="float16 not supported")),
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ],
    )
    def test_remainder_different_dtypes(self, device, dtype):
        """Test remainder with different data types."""
        with track_neuron_ops():
            x = torch.randint(1, 20, (3, 4), dtype=dtype, device=device)
            y = torch.randint(1, 10, (3, 4), dtype=dtype, device=device)
            neuron_result = torch.remainder(x, y)
            cpu_result = torch.remainder(x.cpu(), y.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::remainder")

    def test_remainder_scalar_divisor(self, device):
        """Test remainder with scalar divisor."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device) * 10
            divisor = 3.0
            neuron_result = torch.remainder(x, divisor)
            cpu_result = torch.remainder(x.cpu(), divisor)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::remainder")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("shape", [(), (3, 4)])
    def test_remainder_integer_scalar_divisor(self, device, dtype, shape):
        """remainder(int tensor, python-int) must keep the input's shape and dtype.

        Regression: remainder.Scalar was not registered for Neuron, so it fell
        back to PyTorch's decomposition that wraps the Python int as a *strong*
        int64 tensor. That promoted the result to int64 and packed a 0-dim
        output as int32[2], leaking shape [2] instead of () (and int64 instead
        of int32). Registering remainder.Scalar routes the raw scalar through
        PreprocessArgs (-> int32), matching CPU weak-scalar semantics.
        """
        with track_neuron_ops():
            x = torch.randint(1, 20, shape, dtype=dtype, device=device)
            neuron_result = torch.remainder(x, 3)
            cpu_result = torch.remainder(x.cpu(), 3)

            assert neuron_result.device.type == "neuron"
            assert tuple(neuron_result.shape) == tuple(cpu_result.shape)
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::remainder")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_remainder_float_boundary(self, device, dtype):
        """remainder on floats must match CPU at integer boundaries.

        fp16/bf16 compute the whole x - floor(x/y)*y in fp32 and downcast once, so the
        boundary correction applies to every float dtype.

        Regression: remainder is x - floor(x/y)*y, and flooring an fp32-rounded quotient
        at an integer boundary (k - eps -> k-1) shifted the remainder by ~y. Sweep every
        (num, den) pair over a grid of fractional values so boundary-adjacent quotients
        are covered, and compare to CPU.
        """
        vals = torch.arange(-16, 16, 0.5)
        num, den = torch.cartesian_prod(vals, vals).unbind(1)
        keep = den != 0
        num, den = num[keep].to(dtype), den[keep].to(dtype)
        with track_neuron_ops():
            expected = torch.remainder(num, den)
            result = torch.remainder(num.to(device), den.to(device))
            torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::remainder")

    def test_remainder_single_element(self, device):
        """Test remainder with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([7.0], device=device)
            y = torch.tensor([3.0], device=device)
            neuron_result = torch.remainder(x, y)
            cpu_result = torch.remainder(x.cpu(), y.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::remainder")

    @pytest.mark.xfail(
        reason="Incorrect result in Neuron for special values:\n"
        "inf % 2.0: Neuron=inf, CPU=nan\n"
        "-inf % 2.0: Neuron=-inf, CPU=nan\n"
        "2.0 % 0.1: Neuron=0.0, CPU=1.0e-01\n"
        "2.0 % 0.0: Neuron=2.0, CPU=nan\n"
        "3.0 % -0.2: Neuron=0.0, CPU=-5.9605e-08"
    )
    def test_remainder_special_values(self, device):
        """Test remainder with special values (inf, -inf, nan)."""
        with track_neuron_ops():
            x = torch.tensor(
                [float("inf"), float("-inf"), float("nan"), 2.0, 2.0, 3.0, 4.0], device=device
            )
            y = torch.tensor([2.0, 2.0, 2.0, 0.1, 0.0, -0.2, -2.0], device=device)
            neuron_result = torch.remainder(x, y)
            cpu_result = torch.remainder(x.cpu(), y.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::remainder")
