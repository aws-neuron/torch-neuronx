"""Tests for asinh (inverse hyperbolic sine) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAsinh:
    """Test suite for aten::asinh, aten::asinh.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_asinh_basic_shapes(self, device, shape):
        """Test asinh with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device) * 5
            neuron_result = torch.asinh(x)
            cpu_result = torch.asinh(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asinh")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_asinh_float_dtypes(self, device, dtype):
        """Test asinh with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
            neuron_result = torch.asinh(x)
            cpu_result = torch.asinh(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_out_variant(self, device):
        """Test asinh .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 1.0, 5.0], device=device)
            out = torch.empty_like(x)
            result = torch.asinh(x, out=out)

            expected = torch.asinh(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_empty_tensor(self, device):
        """Test asinh with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.asinh(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_single_element(self, device):
        """Test asinh with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            neuron_result = torch.asinh(x)
            cpu_result = torch.asinh(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_zero(self, device):
        """Test asinh(0) = 0."""
        with track_neuron_ops():
            x = torch.tensor([0.0], device=device)
            neuron_result = torch.asinh(x)
            assert neuron_result.cpu().item() == 0.0
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_symmetry(self, device):
        """Test asinh(-x) = -asinh(x) (odd function)."""
        with track_neuron_ops():
            x = torch.tensor([0.5, 1.0, 2.0, 5.0], device=device)
            pos = torch.asinh(x)
            neg = torch.asinh(-x)
            torch.testing.assert_close(neg.cpu(), -pos.cpu())
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_special_values(self, device):
        """Test asinh with inf, -inf, nan."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.asinh(x)
            cpu_result = torch.asinh(x.cpu())
            # asinh(inf)=inf, asinh(-inf)=-inf, asinh(nan)=nan
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_no_recompilation(self, device):
        """Test that calling asinh twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            torch.asinh(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.asinh(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::asinh")

    def test_asinh_backward_correctness(self, device):
        """Test asinh backward pass matches CPU."""
        x_cpu = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.asinh(x_cpu)
        out_neuron = torch.asinh(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_asinh_autograd_chain(self, device):
        """Test that asinh does not break autograd chain."""
        with track_neuron_ops():
            x = torch.randn(4, device=device, requires_grad=True)
            y = torch.asinh(x)
            assert y.grad_fn is not None
            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
