"""Tests for atan (inverse tangent) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAtan:
    """Test suite for aten::atan, aten::atan.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_atan_basic_shapes(self, device, shape):
        """Test atan with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device) * 10
            neuron_result = torch.atan(x)
            cpu_result = torch.atan(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::atan")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_atan_float_dtypes(self, device, dtype):
        """Test atan with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=dtype, device=device)
            neuron_result = torch.atan(x)
            cpu_result = torch.atan(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_out_variant(self, device):
        """Test atan .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 1.0, 5.0], device=device)
            out = torch.empty_like(x)
            result = torch.atan(x, out=out)

            expected = torch.atan(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_empty_tensor(self, device):
        """Test atan with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.atan(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_single_element(self, device):
        """Test atan with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            neuron_result = torch.atan(x)
            cpu_result = torch.atan(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_known_values(self, device):
        """Test atan with known mathematical values."""
        import math

        with track_neuron_ops():
            # atan(0)=0, atan(1)=pi/4, atan(-1)=-pi/4
            x = torch.tensor([0.0, 1.0, -1.0], device=device)
            neuron_result = torch.atan(x)
            expected = torch.tensor([0.0, math.pi / 4, -math.pi / 4])
            torch.testing.assert_close(neuron_result.cpu(), expected)
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_special_values(self, device):
        """Test atan with inf, -inf, nan."""
        import math

        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.atan(x)
            cpu_result = torch.atan(x.cpu())
            # atan(inf)=pi/2, atan(-inf)=-pi/2, atan(nan)=nan
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_no_recompilation(self, device):
        """Test that calling atan twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            torch.atan(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.atan(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::atan")

    def test_atan_backward_correctness(self, device):
        """Test atan backward pass matches CPU."""
        x_cpu = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.atan(x_cpu)
        out_neuron = torch.atan(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_atan_autograd_chain(self, device):
        """Test that atan does not break autograd chain."""
        with track_neuron_ops():
            x = torch.randn(4, device=device, requires_grad=True)
            y = torch.atan(x)
            assert y.grad_fn is not None
            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
