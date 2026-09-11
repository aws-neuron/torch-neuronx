"""Tests for TryHandleEmptyTensors fill-value ops.

Ops like constant_pad_nd and linear can produce non-empty outputs from
empty inputs (e.g. padding expands a [0,3,3] to [2,5,5]).  The
FillValueConfig in TryHandleEmptyTensors must use the correct fill
value — not zeros — for these cases.
"""

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
