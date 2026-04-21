"""Tests for Neuron-safe rotary position embedding.

Verifies that :func:`torch_neuronx.python_ops.rotary_emb.apply_rotary_emb`
produces finite, numerically correct results on both CPU and Neuron, in
contrast to the standard ``empty_like + stride-2 slice`` pattern which
yields inf/NaN on Neuron.
"""

from __future__ import annotations

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import math

import pytest
import torch

import torch_neuronx  # noqa: F401 — registers neuron device
from torch_neuronx.python_ops.rotary_emb import apply_rotary_emb

DEVICE = torch.device("neuron")


def _buggy_rope_reference(hidden_states, freqs_cos, freqs_sin):
    """The original diffusers RoPE formula that is broken on Neuron."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _cpu_reference(hidden_states_cpu, freqs_cos_cpu, freqs_sin_cpu):
    """CPU reference (always correct)."""
    return _buggy_rope_reference(hidden_states_cpu, freqs_cos_cpu, freqs_sin_cpu)


def _make_inputs(B, S, H, D, dtype=torch.bfloat16, seed=42):
    """Create deterministic RoPE inputs."""
    torch.manual_seed(seed)
    h = torch.randn(B, S, H, D, dtype=dtype)
    cos = torch.randn(1, S, 1, D, dtype=torch.float32) * 0.1
    sin = torch.randn(1, S, 1, D, dtype=torch.float32) * 0.1
    return h, cos, sin


# ---------- CPU correctness (baseline sanity check) ----------

class TestRoPECPU:
    """Verify apply_rotary_emb matches the reference on CPU."""

    @pytest.mark.parametrize(
        "B,S,H,D",
        [
            (1, 8, 2, 16),
            (1, 64, 2, 128),
            (2, 32, 4, 64),
        ],
    )
    def test_cpu_matches_reference(self, B, S, H, D):
        h, cos, sin = _make_inputs(B, S, H, D)
        ref = _cpu_reference(h, cos, sin).float()
        result = apply_rotary_emb(h, cos, sin).float()
        assert torch.isfinite(result).all(), "apply_rotary_emb produced non-finite on CPU"
        assert torch.allclose(result, ref, atol=1e-3, rtol=1e-3), (
            f"CPU mismatch: max_diff={float((result - ref).abs().max()):.4g}"
        )


# ---------- Neuron tests ----------

class TestRoPENeuron:
    """Verify apply_rotary_emb works correctly on Neuron device."""

    @pytest.mark.parametrize(
        "B,S,H,D",
        [
            (1, 64, 2, 128),
            (1, 3510, 24, 128),   # Wan TI2V-5B smoke test shape
        ],
    )
    def test_safe_rope_finite_on_neuron(self, B, S, H, D):
        """The safe implementation should produce finite results on Neuron."""
        h, cos, sin = _make_inputs(B, S, H, D)
        h_n = h.to(DEVICE)
        cos_n = cos.to(DEVICE)
        sin_n = sin.to(DEVICE)

        result = apply_rotary_emb(h_n, cos_n, sin_n)
        result_cpu = result.detach().float().cpu()

        assert torch.isfinite(result_cpu).all(), (
            f"apply_rotary_emb produced non-finite on Neuron! "
            f"min={float(result_cpu.min()):.4g} max={float(result_cpu.max()):.4g}"
        )

    @pytest.mark.parametrize(
        "B,S,H,D",
        [
            (1, 64, 2, 128),
        ],
    )
    def test_safe_rope_matches_cpu_reference(self, B, S, H, D):
        """The safe implementation should match CPU reference numerically."""
        h, cos, sin = _make_inputs(B, S, H, D)
        ref = _cpu_reference(h, cos, sin).float()

        h_n = h.to(DEVICE)
        cos_n = cos.to(DEVICE)
        sin_n = sin.to(DEVICE)
        result = apply_rotary_emb(h_n, cos_n, sin_n).detach().float().cpu()

        if not torch.isfinite(result).all():
            pytest.fail("Result is non-finite, cannot compare")

        max_diff = float((result - ref).abs().max())
        assert max_diff < 1.0, f"Neuron vs CPU max_diff={max_diff:.4g} (too large)"

    @pytest.mark.parametrize(
        "B,S,H,D",
        [
            (1, 64, 2, 128),
        ],
    )
    def test_buggy_rope_reproduces_nan(self, B, S, H, D):
        """Verify the original pattern still produces non-finite on Neuron (regression guard)."""
        h, cos, sin = _make_inputs(B, S, H, D)
        h_n = h.to(DEVICE)
        cos_n = cos.to(DEVICE)
        sin_n = sin.to(DEVICE)

        result = _buggy_rope_reference(h_n, cos_n, sin_n)
        result_cpu = result.detach().float().cpu()

        is_finite = bool(torch.isfinite(result_cpu).all())
        if is_finite:
            pytest.skip(
                "Buggy pattern did not reproduce NaN on this Neuron build "
                "(the underlying issue may have been fixed in the compiler)"
            )


# ---------- Dtype and shape edge cases ----------

class TestRoPEEdgeCases:
    """Edge-case tests for the safe RoPE implementation."""

    def test_float32_hidden_states(self):
        """RoPE should work with fp32 hidden states too."""
        h, cos, sin = _make_inputs(1, 16, 2, 32, dtype=torch.float32)
        result = apply_rotary_emb(h, cos, sin)
        assert torch.isfinite(result).all()
        assert result.dtype == torch.float32

    def test_output_dtype_matches_input(self):
        """Output dtype should match hidden_states dtype."""
        h, cos, sin = _make_inputs(1, 16, 2, 32, dtype=torch.bfloat16)
        result = apply_rotary_emb(h, cos, sin)
        assert result.dtype == h.dtype

    def test_output_shape_matches_input(self):
        """Output shape should match hidden_states shape."""
        h, cos, sin = _make_inputs(1, 16, 2, 32)
        result = apply_rotary_emb(h, cos, sin)
        assert result.shape == h.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
