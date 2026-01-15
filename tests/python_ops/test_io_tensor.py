"""Tests for io_tensor module."""

import pytest
import torch

from torch_neuronx.python_ops import io_tensor


class TestIOTensor:
    """Test centralized I/O tensor creation."""

    def test_empty_neuron_device(self):
        """Verify empty works with neuron device."""
        t = io_tensor.empty((2, 3), device="neuron")
        assert t.device.type == "neuron"
        assert t.shape == (2, 3)

    def test_empty_neuron_device_with_dtype(self):
        """Verify empty works with neuron device and dtype."""
        t = io_tensor.empty((4, 5), dtype=torch.float32, device="neuron")
        assert t.device.type == "neuron"
        assert t.dtype == torch.float32
        assert t.shape == (4, 5)

    def test_empty_cpu_device(self):
        """Verify empty works with CPU device."""
        t = io_tensor.empty((2, 3), device="cpu")
        assert t.device.type == "cpu"
        assert t.shape == (2, 3)

    def test_empty_no_device(self):
        """Verify empty works without device parameter."""
        t = io_tensor.empty((2, 3))
        assert t.shape == (2, 3)

    def test_tensor_neuron_device(self):
        """Verify tensor works with neuron device."""
        t = io_tensor.tensor([1.0, 2.0, 3.0], device="neuron")
        assert t.device.type == "neuron"
        assert t.shape == (3,)

    def test_tensor_neuron_device_with_dtype(self):
        """Verify tensor works with neuron device and dtype."""
        t = io_tensor.tensor([1, 2, 3], dtype=torch.int32, device="neuron")
        assert t.device.type == "neuron"
        assert t.dtype == torch.int32

    def test_tensor_cpu_device(self):
        """Verify tensor works with CPU device."""
        t = io_tensor.tensor([1.0, 2.0, 3.0], device="cpu")
        assert t.device.type == "cpu"

    def test_tensor_no_device(self):
        """Verify tensor works without device parameter."""
        t = io_tensor.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_empty_torch_device_object(self):
        """Verify empty works with torch.device object."""
        device = torch.device("neuron")
        t = io_tensor.empty((2, 3), device=device)
        assert t.device.type == "neuron"

    def test_tensor_torch_device_object(self):
        """Verify tensor works with torch.device object."""
        device = torch.device("neuron")
        t = io_tensor.tensor([1.0, 2.0], device=device)
        assert t.device.type == "neuron"
