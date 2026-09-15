"""Tests for expm1 (exp(x) - 1) operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestExpm1:
    """Test suite for aten::expm1 and aten::expm1.out."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize(
        "shape",
        [
            (),
            (5,),
            (3, 4),
            (2, 3, 4),
            (2, 3, 4, 5),
            (7, 19),
            (1, 128),
            (256, 256),
            (1024, 1024),
        ],
    )
    def test_expm1_shapes(self, device, shape):
        """Test expm1 across 0-d, 1D-4D, non-square, odd, and large (1M+) shapes."""
        with track_neuron_ops():
            # keep inputs in [-2, 2] so the output range stays reasonable
            x = torch.rand(shape, device=device) * 4 - 2
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::expm1")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_expm1_float_dtypes(self, device, dtype):
        """Test expm1 with floating-point data types."""
        with track_neuron_ops():
            x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=dtype, device=device)
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::expm1")

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool]
    )
    def test_expm1_integer_promotes_to_float(self, device, dtype):
        """Test integer and bool inputs promote to the default float dtype.

        The hardware exp is inaccurate, so a naive exp(x) - 1 misses the fp32
        tolerance and every promoted integer case fails. The lowering uses the
        range-reduced exp instead, so these must match the CPU oracle exactly.
        """
        with track_neuron_ops():
            x = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            assert neuron_result.dtype == torch.get_default_dtype()
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_float64_downcast(self, device):
        """Test float64 input keeps its dtype (computed at float32 precision).

        Neuron has no native float64, so compare at a relaxed tolerance.
        """
        with track_neuron_ops():
            x = torch.linspace(-3, 3, 30, dtype=torch.float64, device=device)
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            assert neuron_result.dtype == torch.float64
            torch.testing.assert_close(
                neuron_result.cpu().float(), cpu_result.float(), rtol=1e-4, atol=1e-4
            )
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_near_zero_accuracy(self, device):
        """Test the cancellation region near zero, where exp(x) - 1 loses digits.

        This is the regime the Taylor-series branch exists for. Compare against
        the CPU oracle at the default fp32 tolerance.
        """
        with track_neuron_ops():
            x = torch.cat(
                [
                    torch.linspace(-0.5, 0.5, 401),
                    torch.tensor([-1e-3, -1e-5, -1e-7, 1e-7, 1e-5, 1e-3, 0.34, 0.36]),
                ]
            ).to(device)
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_large_magnitude(self, device):
        """Test large positive and negative inputs, including overflow to inf."""
        with track_neuron_ops():
            x = torch.tensor([-100.0, -50.0, -10.0, 10.0, 50.0, 88.0, 100.0], device=device)
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_special_values(self, device):
        """Test inf, -inf, nan, +/-0, and denormal-scale inputs.

        expm1(+inf)=+inf, expm1(-inf)=-1, expm1(nan)=nan, expm1(-0.0)=-0.0.
        """
        with track_neuron_ops():
            x = torch.tensor(
                [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1e38, 1e-38],
                dtype=torch.float32,
                device=device,
            )
            neuron_result = torch.expm1(x)
            cpu_result = torch.expm1(x.cpu())

            torch.testing.assert_close(neuron_result.cpu(), cpu_result, equal_nan=True)
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_empty_tensor(self, device):
        """Test expm1 with an empty tensor (numel=0)."""
        with track_neuron_ops():
            x = torch.tensor([], device=device)
            neuron_result = torch.expm1(x)
            assert neuron_result.numel() == 0
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_single_element(self, device):
        """Test expm1 with a single-element tensor."""
        with track_neuron_ops():
            x = torch.tensor([0.5], device=device)
            neuron_result = torch.expm1(x)
            torch.testing.assert_close(neuron_result.cpu(), torch.expm1(x.cpu()))
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_zero(self, device):
        """Test expm1(0) = 0."""
        with track_neuron_ops():
            x = torch.tensor([0.0], device=device)
            neuron_result = torch.expm1(x)
            assert abs(neuron_result.cpu().item()) < 1e-7
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_out_variant(self, device):
        """Test the .out variant writes the value and aliases the out tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 0.5, 2.0], device=device)
            out = torch.empty_like(x)
            result = torch.expm1(x, out=out)

            assert out.data_ptr() == result.data_ptr()
            torch.testing.assert_close(out.cpu(), torch.expm1(x.cpu()))
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_out_leaves_self_untouched(self, device):
        """Test the .out variant does not corrupt the input tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 0.5, 2.0], device=device)
            self_before = x.clone()
            out = torch.empty_like(x)
            torch.expm1(x, out=out)

            torch.testing.assert_close(x.cpu(), self_before.cpu())
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_out_noncontiguous(self, device):
        """Test the .out variant into a non-contiguous view."""
        with track_neuron_ops():
            x = torch.linspace(-2, 2, 8, device=device)
            base = torch.empty(8, 2, device=device)
            out = base[:, 0]
            assert not out.is_contiguous()
            torch.expm1(x, out=out)

            torch.testing.assert_close(out.cpu(), torch.expm1(x.cpu()))
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_out_dtype_mismatch_raises(self, device):
        """A non-float out= dtype for a float op must raise RuntimeError.

        CPU allows a float-to-float out= downcast but rejects a float-to-integer
        out=. The Neuron dispatch path skips PyTorch's structured-kernel guard,
        so the lowering validates this itself.
        """
        x = torch.tensor([0.0, 1.0], device=device)
        out = torch.empty(2, dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.expm1(x, out=out)

    def test_expm1_inplace_variant(self, device):
        """Test expm1_ writes the value in place and keeps the same tensor."""
        with track_neuron_ops():
            x = torch.tensor([-1.0, 0.0, 0.5, 2.0], device=device)
            expected = torch.expm1(x.cpu())
            ptr_before = x.data_ptr()
            result = x.expm1_()

            assert result.data_ptr() == ptr_before
            torch.testing.assert_close(x.cpu(), expected)
            assert_op_runs_on_neuron("aten::expm1_")

    def test_expm1_inplace_integer_raises(self, device):
        """Test in-place expm1_ on an integer tensor raises, matching CPU.

        expm1 promotes an integer to float, so the float result cannot be
        stored back into an integer tensor.
        """
        x = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            x.expm1_()

    def test_expm1_no_recompilation(self, device):
        """Test calling expm1 twice with the same shape does not recompile."""
        import time

        with track_neuron_ops():
            x = torch.randn(64, 64, device=device)
            torch.expm1(x)
            torch.neuron.synchronize()

            start = time.perf_counter()
            torch.expm1(x)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start

            assert elapsed < 0.1, f"Recompiled: {elapsed:.3f}s"
            assert_op_runs_on_neuron("aten::expm1")

    def test_expm1_backward_correctness(self, device):
        """Test the expm1 backward matches CPU and runs on Neuron.

        d/dx expm1(x) = exp(x). Backward is derived (no dedicated backward op),
        so assert the forward op runs on Neuron and compare gradients to CPU.
        """
        x_cpu = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0, 2.0], requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(device).requires_grad_(True)

        with track_neuron_ops():
            out_neuron = torch.expm1(x_neuron)
            assert_op_runs_on_neuron("aten::expm1")

        out_cpu = torch.expm1(x_cpu)
        grad = torch.ones_like(out_cpu)
        out_cpu.backward(grad)
        out_neuron.backward(grad.to(device))

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_expm1_autograd_chain(self, device):
        """Test expm1 does not break the autograd chain."""
        with track_neuron_ops():
            x = torch.randn(4, device=device, requires_grad=True)
            y = torch.expm1(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
