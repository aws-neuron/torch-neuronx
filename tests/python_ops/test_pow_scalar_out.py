"""Unit tests for the scalar-base pow overloads (``scalar ** tensor``).

Covers ``aten::pow.Scalar`` (functional) and ``aten::pow.Scalar_out`` (the
preallocated-output variant). Distinct from the tensor-base ``pow.Tensor_Scalar``
family: here the base is a Python scalar and the exponent is a tensor. CPU is the
reference oracle and every device call asserts the op ran on Neuron (not a silent
CPU fallback).
"""

import math

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)

SHAPES = [(), (1,), (5,), (2, 3), (2, 3, 4), (0,)]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPowScalarOut:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_functional_shapes(self, shape):
        """scalar ** tensor over a range of shapes, incl. scalar and 0-element."""
        with track_neuron_ops():
            if shape == ():
                exp_cpu = torch.tensor(2.0)
            else:
                exp_cpu = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            expected = torch.pow(3.0, exp_cpu)
            result = torch.pow(3.0, exp_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
    def test_float_dtypes(self, dtype):
        """Float exponent dtypes, including float64 (Neuron computes in float32)."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
            expected = torch.pow(2.0, exp_cpu)
            result = torch.pow(2.0, exp_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_int_dtypes(self, dtype):
        """Integer base and integer exponent stays integer; int64 exercises the
        int32-compute-then-widen path (neuronx-cc has no 64-bit power)."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0, 1, 2, 3], dtype=dtype)
            expected = torch.pow(2, exp_cpu)
            assert expected.dtype == dtype
            result = torch.pow(2, exp_cpu.to("neuron"))
            assert result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    @pytest.mark.parametrize("base", [-2.0, -2.5])
    def test_negative_base(self, base):
        """Negative scalar base: integer-valued float exponents alternate sign,
        fractional exponents yield ``nan`` (matching CPU). This is the classic
        transcendental-pow divergence point — a naive ``exp(exp*log(base))`` would
        return ``nan`` for every exponent of a negative base."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0.0, 2.0, 3.0, 0.5, -1.0, -2.0])
            expected = torch.pow(base, exp_cpu)
            result = torch.pow(base, exp_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_negative_base_int_exponent(self, dtype):
        """Negative scalar base with integer-dtype exponents: sign alternates with
        exponent parity; also exercises the int64 widen path with a negative base."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0, 1, 2, 3], dtype=dtype)
            expected = torch.pow(-2.0, exp_cpu)
            result = torch.pow(-2.0, exp_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    def test_out_variant(self):
        """pow.Scalar_out writes the value AND leaves the exponent (self) untouched."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0.0, 1.0, 2.0, 3.0])
            exp_dev = exp_cpu.to("neuron")
            exp_before = exp_dev.clone()
            out = torch.empty(4, device="neuron")
            expected = torch.pow(3.0, exp_cpu)
            ret = torch.pow(3.0, exp_dev, out=out)
            torch.testing.assert_close(out.cpu(), expected)
            torch.testing.assert_close(ret.cpu(), expected)
            # exponent argument must not be mutated by the out= call
            assert torch.equal(exp_dev.cpu(), exp_before.cpu())
            assert_op_runs_on_neuron("aten::pow.Scalar_out")

    def test_out_variant_noncontiguous(self):
        """out= into a non-contiguous view."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0.0, 1.0, 2.0, 3.0])
            out = torch.empty(4, 2, device="neuron")[:, 0]
            assert not out.is_contiguous()
            expected = torch.pow(2.0, exp_cpu)
            torch.pow(2.0, exp_cpu.to("neuron"), out=out)
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar_out")

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_out_variant_int(self, dtype):
        """out= with integer dtypes, incl. the int64 widen path."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0, 1, 2, 3], dtype=dtype)
            out = torch.empty(4, dtype=dtype, device="neuron")
            expected = torch.pow(2, exp_cpu)
            torch.pow(2, exp_cpu.to("neuron"), out=out)
            assert out.dtype == dtype
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Scalar_out")

    def test_autograd_backward_on_neuron(self):
        """Differentiable wrt the exponent: gradient matches CPU and the backward
        runs on Neuron with no CPU fallback."""
        exp_cpu = torch.tensor([0.5, 1.0, 2.0, 3.0], requires_grad=True)
        torch.pow(2.0, exp_cpu).sum().backward()

        exp_dev = torch.tensor([0.5, 1.0, 2.0, 3.0], device="neuron", requires_grad=True)
        y = torch.pow(2.0, exp_dev)
        torch.neuron.synchronize()
        with track_neuron_ops():
            y.sum().backward()
            torch.neuron.synchronize()
            fallback = torch_neuronx.get_fallback_ops()
            assert fallback == [], f"backward fell back to CPU: {fallback}"
        torch.testing.assert_close(exp_dev.grad.cpu(), exp_cpu.grad)

    def test_backward_op_runs_on_neuron(self):
        """The declared backward op (pow.Tensor_Tensor) dispatches on Neuron."""
        with track_neuron_ops():
            base = torch.tensor([2.0, 3.0, 4.0], device="neuron")
            expo = torch.tensor([2.0, 2.0, 3.0], device="neuron")
            expected = torch.pow(base.cpu(), expo.cpu())
            result = torch.pow(base, expo)
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::pow.Tensor_Tensor")

    def test_inf_nan(self):
        """inf/nan exponents match CPU semantics."""
        with track_neuron_ops():
            exp_cpu = torch.tensor([0.0, 1.0, float("inf"), float("-inf"), float("nan")])
            expected = torch.pow(2.0, exp_cpu)
            result = torch.pow(2.0, exp_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
            assert_op_runs_on_neuron("aten::pow.Scalar")

    @pytest.mark.xfail(reason="complex dtypes not supported on Neuron")
    def test_complex_unsupported(self):
        exp_cpu = torch.tensor([1.0, 2.0], dtype=torch.complex64)
        with track_neuron_ops():
            result = torch.pow(2.0, exp_cpu.to("neuron"))
            torch.neuron.synchronize()
            torch.testing.assert_close(result.cpu(), torch.pow(2.0, exp_cpu))
