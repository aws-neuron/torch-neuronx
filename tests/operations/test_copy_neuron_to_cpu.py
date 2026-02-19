import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


def test_copy_neuron_to_cpu_basic():
    """Test basic Neuron to CPU copy"""
    # Create neuron tensor with known data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    # First copy data to neuron tensor
    neuron_tensor.copy_(cpu_source)

    # Now copy back to CPU
    cpu_result = torch.empty_like(cpu_source)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_source, cpu_result)


def test_copy_neuron_to_cpu_non_contiguous_dst():
    """Test that non-contiguous CPU destination works"""
    # Create a neuron tensor with data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")
    neuron_tensor.copy_(cpu_source)

    # Create non-contiguous CPU tensor with matching size
    cpu_base = torch.empty(4, 4)
    cpu_tensor = cpu_base.transpose(0, 1)

    assert not cpu_tensor.is_contiguous()

    cpu_tensor.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_source)
