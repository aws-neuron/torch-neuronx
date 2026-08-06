"""Tests for cosh (hyperbolic cosine) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCosh:
    """Test suite for aten::cosh, aten::cosh.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_cosh_basic_shapes(self, device, shape):
        """Test cosh with various tensor shapes."""
        with track_neuron_ops():
            # cosh grows exponentially — keep inputs in [-2, 2] for reasonable output range
            x = torch.rand(shape, device=device) * 4 - 2
            neuron_result = torch.cosh(x)
            cpu_result = torch.cosh(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::cosh")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cosh_float_dtypes(self, device, dtype):
        """Test cosh with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
            neuron_result = torch.cosh(x)
            cpu_result = torch.cosh(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_out_variant(self, device):
        """Test cosh .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
            out = torch.empty_like(x)
            result = torch.cosh(x, out=out)

            expected = torch.cosh(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_out_dtype_mismatch_raises(self, device):
        """Test an out= tensor with the wrong dtype raises, matching CPU.

        The Neuron dispatch path skips PyTorch's structured-kernel unsafe-cast
        guard, so the lowering validates this itself.
        """
        x = torch.tensor([0.0, 1.0], device=device)
        out = torch.empty(2, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.cosh(x, out=out)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
    def test_cosh_integer_promotes_to_float(self, device, dtype):
        """Test integer and bool inputs promote to the default float dtype."""
        with track_neuron_ops():
            x = torch.ones(4, dtype=dtype, device=device)
            neuron_result = torch.cosh(x)
            cpu_result = torch.cosh(x.cpu())

            assert neuron_result.dtype == torch.get_default_dtype()
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_empty_tensor(self, device):
        """Test cosh with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.cosh(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_single_element(self, device):
        """Test cosh with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([1.0], device=device)
            neuron_result = torch.cosh(x)
            cpu_result = torch.cosh(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_zero(self, device):
        """Test cosh(0) = 1."""
        with track_neuron_ops():
            x = torch.tensor([0.0], device=device)
            neuron_result = torch.cosh(x)
            assert abs(neuron_result.cpu().item() - 1.0) < 1e-6
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_symmetry(self, device):
        """Test cosh(-x) = cosh(x) (even function)."""
        with track_neuron_ops():
            x = torch.tensor([0.5, 1.0, 2.0, 3.0], device=device)
            pos = torch.cosh(x)
            neg = torch.cosh(-x)
            torch.testing.assert_close(pos.cpu(), neg.cpu())
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_special_values(self, device):
        """Test cosh with inf, -inf, nan."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.cosh(x)
            cpu_result = torch.cosh(x.cpu())
            # cosh(inf)=inf, cosh(-inf)=inf, cosh(nan)=nan
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_no_recompilation(self, device):
        """Test that calling cosh twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            torch.cosh(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.cosh(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::cosh")

    def test_cosh_backward_correctness(self, device):
        """Test cosh backward pass matches CPU."""
        x_cpu = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.cosh(x_cpu)
        out_neuron = torch.cosh(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_cosh_autograd_chain(self, device):
        """Test that cosh does not break autograd chain."""
        with track_neuron_ops():
            x = torch.randn(4, device=device, requires_grad=True)
            y = torch.cosh(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
