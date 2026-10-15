"""Tests for TryHandleEmptyTensors fill-value ops.

Ops like constant_pad_nd and linear can produce non-empty outputs from
empty inputs (e.g. padding expands a [0,3,3] to [2,5,5]).  The
FillValueConfig in TryHandleEmptyTensors must use the correct fill
value — not zeros — for these cases.
"""

import pytest
import torch


class TestEmptyInputFillValue:
    def test_constant_pad_nd_nonzero_fill(self):
        """Padding an empty tensor with value=1.0 should produce all 1s."""
        x = torch.randn(0, 3, 3, device="neuron")
        result = torch.constant_pad_nd(x, (1, 1, 1, 1, 1, 1), 1.0)
        ref = torch.constant_pad_nd(x.cpu(), (1, 1, 1, 1, 1, 1), 1.0)
        assert result.shape == ref.shape
        torch.testing.assert_close(result.cpu(), ref, rtol=0, atol=0)

    def test_constant_pad_nd_zero_fill(self):
        """Padding with value=0 (default) — was accidentally correct before."""
        x = torch.randn(0, 3, 3, device="neuron")
        result = torch.constant_pad_nd(x, (1, 1, 1, 1, 1, 1), 0.0)
        assert result.cpu().eq(0).all()

    def test_linear_bias_with_zero_contraction(self):
        """Linear with zero contraction dim should return bias, not zeros."""
        torch.manual_seed(42)
        inp = torch.randn(3, 4, 0, device="neuron")
        w = torch.randn(5, 0, device="neuron")
        b = torch.randn(5, device="neuron")
        result = torch.nn.functional.linear(inp, w, b)
        ref = torch.nn.functional.linear(inp.cpu(), w.cpu(), b.cpu())
        assert result.shape == ref.shape
        torch.testing.assert_close(result.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_linear_no_bias_with_zero_contraction(self):
        """Linear with no bias and zero contraction dim should be all zeros."""
        inp = torch.randn(2, 0, device="neuron")
        w = torch.randn(3, 0, device="neuron")
        result = torch.nn.functional.linear(inp, w, None)
        assert result.shape == torch.Size([2, 3])
        assert result.cpu().eq(0).all()


class TestEmptyScalarOutOnDevice:
    """Empty full reduction with a scalar out= must stay on-device.

    The empty-input identity is fabricated on host and written into the
    caller's out= buffer with a single host->device copy. A device-side 0-d
    copy_ would round-trip through the host (Neuron has no on-device scalar
    copy), surfacing as neuron::copy::neuron_to_cpu and mislabeling the op as a
    CPU fallback. These tests assert the write stays on-device and is correct.
    """

    @pytest.mark.parametrize(
        "run, ref",
        [
            (lambda e, o: torch.ops.aten.prod.out(e, out=o), torch.prod),
            (
                lambda e, o: torch.ops.aten.sum.IntList_out(e, dim=[], out=o),
                lambda e: torch.sum(e, dim=[]),
            ),
            (
                lambda e, o: torch.ops.aten.mean.out(e, dim=[], out=o),
                lambda e: torch.mean(e, dim=[]),
            ),
        ],
        ids=["prod.out", "sum.IntList_out", "mean.out"],
    )
    def test_empty_scalar_out_stays_on_device(self, run, ref):
        import torch_neuronx

        e = torch.empty(0, device="neuron")
        out = torch.empty((), device="neuron")

        torch_neuronx.synchronize()
        torch_neuronx.clear_op_tracking()
        run(e, out)
        torch_neuronx.synchronize()

        executed = set(torch_neuronx.get_executed_ops())
        assert (
            "neuron::copy::neuron_to_cpu" not in executed
        ), f"empty scalar out= round-tripped through host: {sorted(executed)}"
        torch.testing.assert_close(out.cpu(), ref(e.cpu()), rtol=0, atol=0, equal_nan=True)
