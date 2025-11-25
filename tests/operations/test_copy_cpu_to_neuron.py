import pytest
import torch

import torch_neuronx


def test_copy_cpu_to_neuron_success():
    """Test successful copy from CPU to Neuron"""
    cpu_tensor = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    # Copy should succeed
    neuron_tensor.copy_(cpu_tensor)

    # Verify by copying back and checking data. Destination CPU tensor must be contiguous.
    cpu_result = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_result)


def test_copy_cpu_to_neuron_non_contiguous_success():
    """Test successful copy from non-contiguous CPU tensor to Neuron"""
    cpu_tensor = torch.randn(4, 4).transpose(0, 1)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    assert not cpu_tensor.is_contiguous()

    # Copy should now succeed even if CPU tensor is non-contiguous
    neuron_tensor.copy_(cpu_tensor)

    # Verify by copying back and checking data. Destination CPU tensor must be contiguous.
    cpu_result = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_result)
