"""
Tests for multi device with Neuron backend
"""

import os
import unittest

import pytest
import torch

import torch_neuronx  # This registers the neuron device
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises, track_neuron_ops


@pytest.mark.multi_device
class TestNeuronDevice(unittest.TestCase):
    """Test multi-device creation and operations"""

    def setup_method(self, method):
        os.environ["NEURON_RT_NUM_CORES"] = "8"

    def test_multiple_devices(self):
        """Test if multiple devices can be created"""
        device_count = torch_neuronx.device_count()
        assert device_count > 1
        for i in range(device_count):
            device = torch.device(f"neuron:{i}")
            self.assertEqual(device.type, "neuron")
            self.assertEqual(device.index, i)

    @pytest.mark.xfail(
        reason="Test fails after setting TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1, "
        "issue with add"
    )
    def test_multiple_device_execution(self):
        """Test execution done on multiple devices"""
        # need to separate the execution of different devices
        t0 = torch.tensor([2.0]).to("neuron:0")
        out0 = t0 + 2
        assert out0.device.index == 0
        assert torch.allclose(out0.cpu(), torch.tensor([4.0]))

        t1 = torch.tensor([2.0]).to("neuron:1")
        out1 = t1 + 2
        assert out1.device.index == 1
        assert torch.allclose(out1.cpu(), torch.tensor([4.0]))

    @assert_raises(
        RuntimeError,
        match="Device index 20 is out of range. Valid range is 0 to 7 for this process",
    )
    def test_accessing_device_out_of_range(self):
        """Test execution done on multiple devices"""
        _ = torch.tensor([2.0]).to("neuron:20")

    def test_resize_zero_preserves_device(self):
        """Test that resize_(0) preserves device across multiple devices."""
        device_count = torch_neuronx.device_count()
        if device_count < 2:
            pytest.skip("Need at least 2 neuron devices")

        for device_idx in range(min(device_count, 4)):  # Test up to 4 devices
            device = f"neuron:{device_idx}"

            # Create tensor on specific device
            x = torch.randn(5, 3, device=device)

            # Verify initial state
            assert x.device.index == device_idx
            assert x.storage().device.index == device_idx

            # Resize to zero (this triggers the bug)
            x._typed_storage()._resize_(0)

            # Device should be preserved
            assert x.device.index == device_idx
            assert (
                x.storage().device.index == device_idx
            ), f"Device {device_idx}: Storage device changed to {x.storage().device}"

    def test_ones_explicit_device_index(self):
        """Test that explicit device argument is respected, not overridden by current device."""
        device_count = torch_neuronx.device_count()
        if device_count < 2:
            pytest.skip("Need at least 2 neuron devices")

        with track_neuron_ops():
            x0 = torch.ones((2, 2), device="neuron:1")
            assert x0.device.index == 1, f"Expected device index 1, got {x0.device.index}"
            assert_op_runs_on_neuron("aten::ones")
