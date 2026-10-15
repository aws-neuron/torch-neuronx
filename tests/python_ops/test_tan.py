"""Tests for tan (tangent) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestTan:
    """Test suite for aten::tan, aten::tan.out operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (100, 100)])
    def test_tan_basic_shapes(self, device, shape):
        """Test tan with various tensor shapes."""
        with track_neuron_ops():
            # Keep inputs away from poles (odd multiples of pi/2)
            x = torch.rand(shape, device=device) * 2 - 1  # [-1, 1]
            neuron_result = torch.tan(x)
            cpu_result = torch.tan(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::tan")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_tan_float_dtypes(self, device, dtype):
        """Test tan with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=device)
            neuron_result = torch.tan(x)
            cpu_result = torch.tan(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_out_variant(self, device):
        """Test tan .out variant writes to pre-allocated tensor."""
        with track_neuron_ops():
            x = torch.tensor([-0.5, 0.0, 0.5, 1.0], device=device)
            out = torch.empty_like(x)
            result = torch.tan(x, out=out)

            expected = torch.tan(x.cpu())
            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_out_dtype_mismatch_raises(self, device):
        """Test an out= tensor with the wrong dtype raises, matching CPU.

        The Neuron dispatch path skips PyTorch's structured-kernel unsafe-cast
        guard, so the lowering validates this itself.
        """
        x = torch.tensor([0.0, 1.0], device=device)
        out = torch.empty(2, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.tan(x, out=out)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
    def test_tan_integer_promotes_to_float(self, device, dtype):
        """Test integer and bool inputs promote to the default float dtype."""
        with track_neuron_ops():
            x = torch.ones(4, dtype=dtype, device=device)
            neuron_result = torch.tan(x)
            cpu_result = torch.tan(x.cpu())

            assert neuron_result.dtype == torch.get_default_dtype()
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_empty_tensor(self, device):
        """Test tan with empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.tan(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_single_element(self, device):
        """Test tan with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([0.5], device=device)
            neuron_result = torch.tan(x)
            cpu_result = torch.tan(x.cpu())
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_zero(self, device):
        """Test tan(0) = 0."""
        with track_neuron_ops():
            x = torch.tensor([0.0], device=device)
            neuron_result = torch.tan(x)
            assert abs(neuron_result.cpu().item()) < 1e-7
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_odd_function(self, device):
        """Test tan(-x) = -tan(x) (odd function)."""
        with track_neuron_ops():
            x = torch.tensor([0.3, 0.7, 1.0], device=device)
            pos = torch.tan(x)
            neg = torch.tan(-x)
            torch.testing.assert_close(neg.cpu(), -pos.cpu())
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_special_values(self, device):
        """Test tan with inf, -inf, nan — all produce NaN."""
        with track_neuron_ops():
            x = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            neuron_result = torch.tan(x)
            cpu_result = torch.tan(x.cpu())
            # tan(inf)=nan, tan(-inf)=nan, tan(nan)=nan
            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_no_recompilation(self, device):
        """Test that calling tan twice with same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.rand(64, 64, device=device) * 2 - 1
            torch.tan(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.tan(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::tan")

    def test_tan_backward_correctness(self, device):
        """Test tan backward pass matches CPU."""
        x_cpu = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        out_cpu = torch.tan(x_cpu)
        out_neuron = torch.tan(x_neuron)

        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_tan_autograd_chain(self, device):
        """Test that tan does not break autograd chain."""
        with track_neuron_ops():
            x = torch.tensor([0.3, -0.3], device=device, requires_grad=True)
            y = torch.tan(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
