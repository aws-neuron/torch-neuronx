"""
Tests for torch.neuron module APIs
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

import torch_neuronx
from torch_neuronx import neuron


class TestNeuronModule(unittest.TestCase):
    """Test torch.neuron module APIs"""

    @patch("torch_neuronx.set_device")
    def test_set_device_int(self, mock_set):
        """Test set_device with integer"""
        neuron.set_device(1)
        mock_set.assert_called_once_with(1)

    @patch("torch_neuronx.set_device")
    def test_set_device_zero(self, mock_set):
        """Test set_device with device index 0"""
        neuron.set_device(0)
        mock_set.assert_called_once_with(0)

    @patch("torch_neuronx.set_device")
    def test_set_device_torch_device(self, mock_set):
        """Test set_device with torch.device"""
        device = torch.device("neuron:2")
        neuron.set_device(device)
        mock_set.assert_called_once_with(2)

    @patch("torch_neuronx.set_device")
    def test_set_device_torch_device_zero(self, mock_set):
        """Test set_device with torch.device index 0"""
        device = torch.device("neuron:0")
        neuron.set_device(device)
        mock_set.assert_called_once_with(0)

    @patch("torch_neuronx.set_device")
    @patch("torch_neuronx.neuron.current_device")
    def test_set_device_none(self, mock_current, mock_set):
        """Test set_device with None uses current_device"""
        mock_current.return_value = 3
        neuron.set_device(None)
        mock_set.assert_called_once_with(3)

    def test_get_device_name(self):
        """Test get_device_name function"""
        self.assertEqual(neuron.get_device_name(), "neuron:0")
        self.assertEqual(neuron.get_device_name(1), "neuron:1")

    def test_get_device_name_with_device_object(self):
        """Test get_device_name with torch.device object"""
        device = torch.device("neuron:2")
        device_name = neuron.get_device_name(device)
        self.assertEqual(device_name, "neuron:2")

    def test_get_device_name_invalid_type(self):
        """Test get_device_name with invalid type raises ValueError"""
        with self.assertRaises(ValueError):
            neuron.get_device_name("invalid")

    @patch("torch_neuronx._C._neuron_getDeviceProperties")
    @patch("torch_neuronx.neuron.current_device")
    def test_get_device_properties(self, mock_current, mock_get_props):
        """Test get_device_properties function"""
        mock_current.return_value = 0
        mock_props = MagicMock()
        mock_props.total_memory = 0
        mock_get_props.return_value = mock_props

        props = neuron.get_device_properties()
        self.assertEqual(props.total_memory, 0)
        mock_get_props.assert_called_once_with(0)

    @patch("torch_neuronx._C._neuron_getDeviceProperties")
    def test_get_device_properties_with_string_device(self, mock_get_props):
        """Test get_device_properties parses device string correctly"""
        mock_props = MagicMock()
        mock_props.total_memory = 0
        mock_get_props.return_value = mock_props

        props = neuron.get_device_properties("neuron:0")
        self.assertEqual(props.total_memory, 0)
        mock_get_props.assert_called_once_with(0)  # Should parse "neuron:0" to device_id=0

    def test_empty_cache(self):
        """Test empty_cache function"""
        # Should not raise exception
        neuron.empty_cache()


if __name__ == "__main__":
    unittest.main()
