"""Tests for acos (inverse cosine) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAcos:
    """Test suite for aten::acos, aten::acos.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_acos_basic_shapes(self, device, shape):
        """Test acos with various tensor shapes."""
        with track_neuron_ops():
            # acos domain is [-1, 1]
            x = torch.rand(shape, device=device) * 2 - 1
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acos")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_acos_float_dtypes(self, device, dtype):
        """Test acos with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_out_variant(self, device):
        """Test acos .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-0.5, 0.0, 0.5, 1.0], device=device)
            out = torch.empty_like(x)
            result = torch.acos(x, out=out)

            expected = torch.acos(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::acos")

    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
    )
    def test_acos_integer_input_promotes_out(self, device, dtype):
        """acos promotes integer input to float; a float32 out= must be accepted.

        Regression for the out= validate rejecting a correctly-typed (promoted)
        out tensor when the input is integer, which surfaced as test_meta
        "Original OpInfo is broken" failures.
        """
        with track_neuron_ops():
            x = torch.ones(4, dtype=dtype, device=device)
            out = torch.empty(4, dtype=torch.get_default_dtype(), device=device)
            torch.acos(x, out=out)

            expected = torch.acos(x.cpu())
            assert out.dtype == torch.get_default_dtype()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_out_dtype_mismatch_raises(self, device):
        """A genuinely wrong out= dtype must still raise (guard not over-relaxed)."""
        x = torch.tensor([0.5], dtype=torch.float32, device=device)
        out = torch.empty(1, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError, match="dtype"):
            torch.acos(x, out=out)

    def test_acos_empty_tensor(self, device):
        """Test acos with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            assert neuron_result.shape == cpu_result.shape
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_single_element(self, device):
        """Test acos with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([0.5], device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_boundary_values(self, device):
        """Test acos at domain boundaries [-1, 1]."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, -0.999, 0.0, 0.999, 1.0], device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_out_of_domain(self, device):
        """Test acos with values outside [-1, 1] — should produce NaN."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -1.5, 1.5, 2.0], device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            # Both should be NaN for out-of-domain inputs
            assert torch.isnan(neuron_result.cpu()).all()
            assert torch.isnan(cpu_result).all()
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_special_values(self, device):
        """Test acos with inf, -inf, nan — all produce NaN (out of domain)."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.acos(x)
            cpu_result = torch.acos(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_no_recompilation(self, device):
        """Test that calling acos twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.rand(64, 64, device=device) * 2 - 1
            torch.acos(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.acos(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::acos")

    def test_acos_backward_correctness(self, device):
        """Test acos backward pass matches CPU."""
        x_cpu = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.acos(x_cpu)
        out_neuron = torch.acos(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_acos_autograd_chain(self, device):
        """Test that acos does not break autograd chain."""
        with track_neuron_ops():
            x = torch.tensor([0.5, -0.5], device=device, requires_grad=True)
            y = torch.acos(x)
            assert y.grad_fn is not None

            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
