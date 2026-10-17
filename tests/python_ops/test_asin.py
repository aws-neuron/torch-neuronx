"""Tests for asin (inverse sine) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAsin:
    """Test suite for aten::asin, aten::asin.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_asin_basic_shapes(self, device, shape):
        """Test asin with various tensor shapes."""
        with track_neuron_ops():
            # asin domain is [-1, 1]
            x = torch.rand(shape, device=device) * 2 - 1
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asin")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_asin_float_dtypes(self, device, dtype):
        """Test asin with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=device)
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_out_variant(self, device):
        """Test asin .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-0.5, 0.0, 0.5, 1.0], device=device)
            out = torch.empty_like(x)
            result = torch.asin(x, out=out)

            expected = torch.asin(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_empty_tensor(self, device):
        """Test asin with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.asin(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_single_element(self, device):
        """Test asin with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([0.5], device=device)
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_boundary_values(self, device):
        """Test asin at domain boundaries [-1, 1]."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, -0.999, 0.0, 0.999, 1.0], device=device)
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_out_of_domain(self, device):
        """Test asin with values outside [-1, 1] — should produce NaN."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -1.5, 1.5, 2.0], device=device)
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())
            assert torch.isnan(neuron_result.cpu()).all()
            assert torch.isnan(cpu_result).all()
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_special_values(self, device):
        """Test asin with inf, -inf, nan — all produce NaN (out of domain)."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.asin(x)
            cpu_result = torch.asin(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_no_recompilation(self, device):
        """Test that calling asin twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.rand(64, 64, device=device) * 2 - 1
            torch.asin(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.asin(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::asin")

    def test_asin_backward_correctness(self, device):
        """Test asin backward pass matches CPU."""
        x_cpu = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.asin(x_cpu)
        out_neuron = torch.asin(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_asin_autograd_chain(self, device):
        """Test that asin does not break autograd chain."""
        with track_neuron_ops():
            x = torch.tensor([0.5, -0.5], device=device, requires_grad=True)
            y = torch.asin(x)
            assert y.grad_fn is not None
            z = y.sum()
            z.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
