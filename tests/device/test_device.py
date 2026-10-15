"""
Tests for torch.device with Neuron backend
"""

import unittest

import torch

import torch_neuronx  # This registers the neuron device


class TestNeuronDevice(unittest.TestCase):
    """Test device creation and properties for Neuron devices"""

    def test_device_without_index(self):
        """Test creating a neuron device without specifying index"""
        device = torch.device("neuron")
        self.assertEqual(device.type, "neuron")
        self.assertIsNone(device.index)

    def test_device_with_index(self):
        """Test creating a neuron device with index 0"""
        device = torch.device("neuron:0")
        self.assertEqual(device.type, "neuron")
        self.assertEqual(device.index, 0)

    def test_device_count(self):
        """Test getting the number of Neuron devices"""
        device_count = torch_neuronx.device_count()
        self.assertIsInstance(device_count, int)
        self.assertGreaterEqual(device_count, 0)

    def test_get_device_properties(self):
        """Test getting device properties"""
        device_count = torch_neuronx.device_count()
        if device_count > 0:
            # Test getting properties for device 0
            props = torch_neuronx.get_device_properties(0)
            self.assertIsNotNone(props)
            # Check that properties object has expected attributes
            self.assertTrue(hasattr(props, "name"))
            self.assertTrue(hasattr(props, "total_memory"))
            self.assertIsInstance(props.name, str)
            self.assertIsInstance(props.total_memory, int)
            self.assertGreater(props.total_memory, 0)

            # Test with string device identifier
            props_str = torch_neuronx.get_device_properties("neuron:0")
            self.assertEqual(props.name, props_str.name)
            self.assertEqual(props.total_memory, props_str.total_memory)
        else:
            # Skip test if no devices available
            self.skipTest("No Neuron devices available")

    def test_get_device_properties_invalid_device(self):
        """Test getting properties for invalid device raises error"""
        device_count = torch_neuronx.device_count()
        # Try to get properties for a device that doesn't exist
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            torch_neuronx.get_device_properties(device_count + 10)

        # Test negative device ID
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            torch_neuronx.get_device_properties(-1)

        # Test device ID exceeding uint32_t max (4294967295)
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            torch_neuronx.get_device_properties(4294967296)

    def test_neuron_device_context_manager(self):
        """Test that torch.neuron.device context manager works correctly."""
        initial_device = torch.neuron.current_device()

        # Test valid device context
        with torch.neuron.device(0):
            device_idx = torch.neuron.current_device()
            self.assertEqual(device_idx, 0)
        assert torch.neuron.current_device() == initial_device

        # Test invalid device raises error
        with self.assertRaises(ValueError):
            device_ctx = torch.neuron.device(1)
            device_ctx.__enter__()


if __name__ == "__main__":
    unittest.main()
