import pytest
import torch

import torch_neuronx


def test_contiguous_channels_last_on_neuron():
    """channels_last contiguous should produce a tensor with NHWC strides."""
    x = torch.randn(2, 3, 4, 5, device="neuron:0")
    y = x.contiguous(memory_format=torch.channels_last)
    assert y.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(
        y.to("cpu"), x.to("cpu").contiguous(memory_format=torch.channels_last)
    )


def test_contiguous_channels_last_3d_on_neuron():
    """channels_last_3d contiguous should produce a tensor with NDHWC strides."""
    x = torch.randn(2, 3, 4, 5, 6, device="neuron:0")
    y = x.contiguous(memory_format=torch.channels_last_3d)
    assert y.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(
        y.to("cpu"), x.to("cpu").contiguous(memory_format=torch.channels_last_3d)
    )


def test_contiguous_channels_last_already_channels_last():
    """If tensor is already channels_last, return as-is (no copy)."""
    x = torch.empty(2, 3, 4, 5, device="neuron:0", memory_format=torch.channels_last)
    y = x.contiguous(memory_format=torch.channels_last)
    assert y.data_ptr() == x.data_ptr()


def test_contiguous_accepts_preserve_format_on_neuron():
    x = torch.randn(3, 4, device="neuron:0").transpose(0, 1)
    assert not x.is_contiguous()
    # preserve_format should be accepted and behave like contiguous
    y = x.contiguous(memory_format=torch.preserve_format)
    assert y.is_contiguous()
    # Values preserved
    torch.testing.assert_close(y.to("cpu"), x.to("cpu").contiguous())


def test_cpu_contiguous_channels_last_unaffected():
    # CPU-only path should follow PyTorch behavior (no Neuron validation)
    x = torch.randn(2, 3, 4, 5, device="cpu")
    y = x.contiguous(memory_format=torch.channels_last)
    # Verify channels_last contiguity on CPU
    assert y.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(y, x)
