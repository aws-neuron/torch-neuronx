"""Tests for acosh (inverse hyperbolic cosine) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAcosh:
    """Test suite for aten::acosh, aten::acosh.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_acosh_basic_shapes(self, device, shape):
        """Test acosh with various tensor shapes."""
        with track_neuron_ops():
            # acosh domain is [1, inf)
            x = torch.rand(shape, device=device) * 10 + 1.0  # [1, 11]
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acosh")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_acosh_float_dtypes(self, device, dtype):
        """Test acosh with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 1.5, 2.0, 5.0, 10.0], dtype=dtype, device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_out_variant(self, device):
        """Test acosh .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 5.0, 10.0], device=device)
            out = torch.empty_like(x)
            result = torch.acosh(x, out=out)

            expected = torch.acosh(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::acosh")

    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    )
    def test_acosh_integer_input_promotes_out(self, device, dtype):
        """acosh promotes integer input to float; a float32 out= must be accepted.

        Regression for the out= validate rejecting a correctly-typed (promoted)
        out tensor when the input is integer, which surfaced as test_meta
        "Original OpInfo is broken" failures.
        """
        with track_neuron_ops():
            x = torch.ones(4, dtype=dtype, device=device)
            out = torch.empty(4, dtype=torch.get_default_dtype(), device=device)
            torch.acosh(x, out=out)

            expected = torch.acosh(x.cpu())
            assert out.dtype == torch.get_default_dtype()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_out_dtype_mismatch_raises(self, device):
        """A genuinely wrong out= dtype must still raise (guard not over-relaxed)."""
        x = torch.tensor([2.0], dtype=torch.float32, device=device)
        out = torch.empty(1, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError, match="dtype"):
            torch.acosh(x, out=out)

    def test_acosh_empty_tensor(self, device):
        """Test acosh with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            assert neuron_result.shape == cpu_result.shape
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_single_element(self, device):
        """Test acosh with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([2.0], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_boundary_value(self, device):
        """Test acosh at domain boundary (x=1 → acosh=0)."""
        with track_neuron_ops():
            x = torch.tensor([1.0, 1.0001, 1.001, 1.01], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_out_of_domain(self, device):
        """Test acosh with values < 1 — should produce NaN."""
        with track_neuron_ops():
            x = torch.tensor([0.0, 0.5, -1.0, 0.99], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            # Both should be NaN for out-of-domain inputs
            assert torch.isnan(neuron_result.cpu()).all()
            assert torch.isnan(cpu_result).all()
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_large_values(self, device):
        """Test acosh with large values."""
        with track_neuron_ops():
            x = torch.tensor([100.0, 1000.0, 1e6, 1e10], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_special_values(self, device):
        """Test acosh with inf and nan."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("nan")], device=device)
            neuron_result = torch.acosh(x)
            cpu_result = torch.acosh(x.cpu())

            # acosh(inf) = inf, acosh(nan) = nan
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_no_recompilation(self, device):
        """Test that calling acosh twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.rand(64, 64, device=device) + 1.0
            torch.acosh(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.acosh(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::acosh")

    def test_acosh_backward_correctness(self, device):
        """Test acosh backward pass matches CPU."""
        x_cpu = torch.tensor([1.5, 2.0, 5.0], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.acosh(x_cpu)
        out_neuron = torch.acosh(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_acosh_autograd_chain(self, device):
        """Test that acosh does not break autograd chain."""
        with track_neuron_ops():
            x = torch.tensor([2.0, 3.0], device=device, requires_grad=True)
            y = torch.acosh(x)
            assert y.grad_fn is not None

            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
