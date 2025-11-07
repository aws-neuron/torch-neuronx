import os

import pytest
import torch

import torch_neuron


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
    reason="Error message for Neuron execution only",
)
def test_contiguous_rejects_channels_last_on_neuron():
    x = torch.randn(2, 3, 4, 5, device="neuron:0")
    with pytest.raises(RuntimeError) as ei:
        # Use both Python API and dispatcher op form are equivalent here
        x.contiguous(memory_format=torch.channels_last)
    assert "Neuron tensors only support contiguous or preserve memory format" in str(ei.value)


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
