"""
Test for bf16/fp32 dtype interaction on Neuron.

When using model.to(torch.bfloat16) on Neuron, Conv3d (and some other ops)
reject fp32 input when weights are bf16, raising:
    RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same

This commonly occurs in diffusion/flow-matching training where noise scheduling
produces fp32 sigma values that, through broadcasting, upcast bf16 latents to
fp32 before passing to a bf16 conv layer.

Workaround: cast intermediates with .to(input_dtype) or use torch.autocast.
"""
import pytest
import torch
import torch.nn as nn


def _get_device():
    try:
        return torch.device("neuron")
    except Exception:
        pytest.skip("Neuron device not available")


class TestBF16DtypePromotion:
    """Test bf16/fp32 dtype interactions on Neuron device."""

    def test_conv3d_rejects_fp32_input_with_bf16_weights(self):
        """Conv3d with bf16 weights rejects fp32 input on Neuron."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        x_fp32 = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Input type.*bias type"):
            conv(x_fp32)

    def test_conv3d_accepts_bf16_input_with_bf16_weights(self):
        """Conv3d with bf16 weights accepts bf16 input."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        x_bf16 = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
        out = conv(x_bf16)
        assert out.dtype == torch.bfloat16

    def test_explicit_dtype_cast_fixes_conv3d(self):
        """Casting fp32 intermediate to bf16 before Conv3d works."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)

        x_bf16 = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
        scale = torch.tensor([0.5], device=device, dtype=torch.float32)
        scale = scale.view(1, 1, 1, 1, 1)
        mixed = x_bf16.float() * scale

        assert mixed.dtype == torch.float32
        fixed = mixed.to(torch.bfloat16)
        out = conv(fixed)
        assert out.dtype == torch.bfloat16

    def test_autocast_fixes_conv3d(self):
        """torch.autocast handles fp32 input to bf16 Conv3d."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(device)
        x_fp32 = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.float32)
        with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
            out = conv(x_fp32)
        assert out.dtype == torch.bfloat16

    def test_diffusion_noise_scheduling_pattern(self):
        """
        End-to-end: fp32 sigma + bf16 latents -> must cast before bf16 Conv3d.

        This is the pattern that breaks in diffusion/flow-matching training
        when migrating from XLA_USE_BF16 to model.to(torch.bfloat16).
        """
        device = _get_device()
        conv = nn.Conv3d(16, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)

        latents = torch.randn(1, 16, 1, 8, 8, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, 1000, (1,), device=device)
        sigma = (timestep.float() / 1000.0)
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)

        noisy = (1 - sigma) * latents + sigma * noise
        noisy_safe = noisy.to(latents.dtype)
        assert noisy_safe.dtype == torch.bfloat16

        out = conv(noisy_safe)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
