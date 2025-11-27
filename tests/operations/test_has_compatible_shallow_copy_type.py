"""Tests for _has_compatible_shallow_copy_type operation"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises


class TestHasCompatibleShallowCopyType:
    """Test the _has_compatible_shallow_copy_type operation"""

    @assert_raises(AssertionError)
    def test_same_device_compatible(self):
        """Test that same device tensors are compatible"""
        # CPU to CPU
        x_cpu = torch.randn(3, 4)
        y_cpu = torch.randn(3, 4)
        assert torch._has_compatible_shallow_copy_type(x_cpu, y_cpu)
        # op should run on CPU
        assert_op_runs_on_neuron("aten::_has_compatible_shallow_copy_type")

        # Neuron to Neuron
        x_neuron = torch.randn(3, 4, device="neuron")
        y_neuron = torch.randn(3, 4, device="neuron")
        assert torch._has_compatible_shallow_copy_type(x_neuron, y_neuron)

    def test_cross_device_compatible(self):
        """Test that CPU and Neuron tensors are compatible (custom implementation)"""
        x_cpu = torch.randn(3, 4)
        y_neuron = torch.randn(3, 4, device="neuron")

        # Both directions should work
        assert torch._has_compatible_shallow_copy_type(x_cpu, y_neuron)
        assert torch._has_compatible_shallow_copy_type(y_neuron, x_cpu)
        assert_op_runs_on_neuron("aten::_has_compatible_shallow_copy_type")

    def test_tensor_data_assignment_neuron_to_cpu(self):
        """Test tensor data assignment from Neuron to CPU"""
        x = torch.randn(10, 5)
        y = x.to("neuron")

        x.data = y

        assert x.device.type == "neuron"
        assert torch.equal(x, y)
        assert_op_runs_on_neuron("aten::_has_compatible_shallow_copy_type")

    def test_tensor_data_assignment_cpu_to_neuron(self):
        """Test tensor data assignment from CPU to Neuron"""
        x = torch.randn(10, 5, device="neuron")
        y = x.to("cpu")

        x.data = y

        assert x.device.type == "cpu"
        assert torch.equal(x, y)
        assert_op_runs_on_neuron("aten::_has_compatible_shallow_copy_type")
