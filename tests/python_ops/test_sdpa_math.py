"""Tests for the shared SDPA Math2 formulas (CPU only, no Neuron device)."""

import pytest
import torch

from torch_neuronx._sdpa_math import compute_attention, compute_attention_backward

SEQ_LEN = 512
NUM_HEADS = 4
HEAD_DIM = 128


def _causal_qkv(dtype, saturation):
    """Build causal QKV. A large saturation makes the softmax rows near one-hot."""
    torch.manual_seed(0)
    query = (torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM) * saturation).to(dtype)
    key = (torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM) * saturation).to(dtype)
    value = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM).to(dtype)
    return query, key, value, HEAD_DIM**-0.5


def _backward_probs(query, key, lse, scale):
    """Rebuild the probabilities the way compute_attention_backward does."""
    logits = (query.float() @ key.float().transpose(-2, -1)) * scale
    index = torch.arange(SEQ_LEN)
    logits = logits.masked_fill(index[:, None] < index[None, :], float("-inf"))
    return torch.exp(logits - lse.float().unsqueeze(-1))


@pytest.mark.parametrize("saturation", [8, 32, 64])
def test_saturated_softmax_keeps_probs_normalized(saturation):
    """The forward lse must normalize the scores that the backward recomputes.

    A bf16 score accumulation gives an lse that is too small, so a probability
    goes above 1 and the backward manufactures a gradient. See issue #1563.
    """
    query, key, value, scale = _causal_qkv(torch.bfloat16, saturation)
    _, lse = compute_attention(query, key, value, None, None, scale, True, 0.0)

    probs = _backward_probs(query, key, lse, scale)
    assert probs.max().item() <= 1.0 + 1e-3


@pytest.mark.parametrize("saturation", [8, 32, 64])
def test_saturated_softmax_backward_tracks_float32(saturation):
    """A bf16 backward must give the same gradient magnitude as an fp32 backward."""
    grad_scale = 1e-7
    amplitudes = {}
    for dtype in (torch.bfloat16, torch.float32):
        query, key, value, scale = _causal_qkv(dtype, saturation)
        out, lse = compute_attention(query, key, value, None, None, scale, True, 0.0)
        grad_out = torch.full_like(out, grad_scale)
        grad_q, grad_k, _, _ = compute_attention_backward(
            grad_out, query, key, value, None, None, out, lse, scale, True, 0.0
        )
        for name, grad in (("q", grad_q), ("k", grad_k)):
            assert torch.isfinite(grad.float()).all()
            amplitudes[(dtype, name)] = grad.float().abs().max().item() / grad_scale

    for name in ("q", "k"):
        bf16 = amplitudes[(torch.bfloat16, name)]
        fp32 = amplitudes[(torch.float32, name)]
        assert bf16 == pytest.approx(fp32, rel=0.25), f"grad_{name} {bf16} vs fp32 {fp32}"
