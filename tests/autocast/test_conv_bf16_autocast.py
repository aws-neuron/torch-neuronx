"""
Test for Conv bf16 weight / fp32 input autocast on Neuron.

Conv3d (and Conv2d/Conv1d) with bf16 weights should accept fp32 input by
auto-casting input to match weight dtype, matching CUDA behavior.

This pattern commonly occurs in diffusion/flow-matching training where
noise scheduling produces fp32 intermediates that are passed to bf16 models.
"""
import pytest
import torch
import torch.nn as nn


def _get_device():
    try:
        return torch.device("neuron")
    except Exception:
        pytest.skip("Neuron device not available")


class TestConvBF16Autocast:
    """Convolution should auto-cast fp32 input to match bf16 weights."""

    def test_conv3d_fp32_input_bf16_weight(self):
        """Conv3d with bf16 weights should accept fp32 input."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        x = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.float32)
        out = conv(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 8, 1, 2, 2)

    def test_conv2d_fp32_input_bf16_weight(self):
        """Conv2d with bf16 weights should accept fp32 input."""
        device = _get_device()
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        conv = conv.to(torch.bfloat16).to(device)
        x = torch.randn(1, 3, 8, 8, device=device, dtype=torch.float32)
        out = conv(x)
        assert out.dtype == torch.bfloat16

    def test_conv1d_fp32_input_bf16_weight(self):
        """Conv1d with bf16 weights should accept fp32 input."""
        device = _get_device()
        conv = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        conv = conv.to(torch.bfloat16).to(device)
        x = torch.randn(1, 4, 16, device=device, dtype=torch.float32)
        out = conv(x)
        assert out.dtype == torch.bfloat16

    def test_conv3d_bf16_input_bf16_weight(self):
        """Conv3d with matching dtypes still works."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        x = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
        out = conv(x)
        assert out.dtype == torch.bfloat16

    def test_conv3d_fp32_input_fp32_weight(self):
        """Conv3d with fp32 everything still works."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(device)
        x = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.float32)
        out = conv(x)
        assert out.dtype == torch.float32

    def test_conv3d_backward_mixed_dtype(self):
        """Backward pass works with mixed fp32/bf16 convolution."""
        device = _get_device()
        conv = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        x = torch.randn(1, 4, 1, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
        out = conv(x)
        loss = out.float().sum()
        loss.backward()
        assert conv.weight.grad is not None
        assert conv.weight.grad.dtype == torch.bfloat16

    def test_diffusion_training_pattern(self):
        """
        End-to-end test: fp32 sigma from noise scheduling, bf16 model.
        This is the pattern that breaks without the auto-cast fix.
        """
        device = _get_device()
        conv = nn.Conv3d(16, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        conv = conv.to(torch.bfloat16).to(device)
        optimizer = torch.optim.AdamW(conv.parameters(), lr=1e-4)

        latents = torch.randn(1, 16, 1, 8, 8, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(latents)
        sigma = torch.tensor([0.5], device=device, dtype=torch.float32)
        sigma = sigma.view(1, 1, 1, 1, 1)

        noisy = (1 - sigma) * latents + sigma * noise
        assert noisy.dtype == torch.float32

        out = conv(noisy)
        assert out.dtype == torch.bfloat16

        loss = out.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(torch.tensor(loss.item()))
