"""Tests for torch.accelerator.get_device_capability on Neuron backend."""

import unittest

import torch

import torch_neuronx


class TestDeviceCapability(unittest.TestCase):
    def setUp(self):
        if torch_neuronx.device_count() == 0:
            self.skipTest("No Neuron devices available")

    def test_returns_dict_with_supported_dtypes(self):
        caps = torch.accelerator.get_device_capability()
        self.assertIsInstance(caps, dict)
        self.assertIn("supported_dtypes", caps)
        self.assertIsInstance(caps["supported_dtypes"], set)

    def test_supported_dtypes_present(self):
        dtypes = torch.accelerator.get_device_capability()["supported_dtypes"]
        expected = {
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        }
        for dt in expected:
            self.assertIn(dt, dtypes, f"{dt} should be supported")

    def test_unsupported_dtypes_absent(self):
        dtypes = torch.accelerator.get_device_capability()["supported_dtypes"]
        for dt in (torch.complex32, torch.complex64, torch.complex128):
            self.assertNotIn(dt, dtypes, f"{dt} should not be supported")

    def test_explicit_device_index(self):
        caps = torch.accelerator.get_device_capability(0)
        self.assertIn("supported_dtypes", caps)
        self.assertIsInstance(caps["supported_dtypes"], set)


if __name__ == "__main__":
    unittest.main()
