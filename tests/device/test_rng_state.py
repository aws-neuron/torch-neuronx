"""
Tests for torch.neuron RNG APIs (get_rng_state, set_rng_state, manual_seed, etc.)
"""

import unittest

import pytest
import torch

import torch_neuronx


class TestNeuronRNG(unittest.TestCase):
    """Test torch.neuron RNG APIs"""

    def test_generator_device(self):
        """Test generator reports correct device type"""
        gen = torch_neuronx._C._get_default_generator(0)
        self.assertEqual(gen.device.type, "neuron")

    def test_rng_state_format(self):
        """Test RNG state is 16-byte philox format"""
        state = torch.neuron.get_rng_state(0)
        self.assertEqual(len(state), 16)
        self.assertEqual(state.dtype, torch.uint8)

    def test_manual_seed(self):
        """Test torch.neuron.manual_seed() and initial_seed()"""
        torch.neuron.manual_seed(2)
        self.assertEqual(torch.neuron.initial_seed(), 2)

        torch.neuron.manual_seed(12345)
        self.assertEqual(torch.neuron.initial_seed(), 12345)

    def test_manual_seed_all(self):
        """Test manual_seed_all sets seed for all devices"""
        torch.neuron.manual_seed_all(1234)
        self.assertEqual(torch.neuron.initial_seed(), 1234)

    def test_seed(self):
        """Test random seed generation"""
        torch.neuron.seed()
        seed = torch.neuron.initial_seed()
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)

    def test_generator_manual_seed(self):
        """Test generator.manual_seed() directly"""
        gen = torch_neuronx._C._get_default_generator(0)
        gen.manual_seed(42)
        self.assertEqual(gen.initial_seed(), 42)

    def test_torch_manual_seed_seeds_neuron(self):
        """Test torch.manual_seed() seeds neuron devices"""
        torch.manual_seed(2)
        self.assertEqual(torch.neuron.initial_seed(), 2)

    def test_get_set_rng_state_roundtrip(self):
        """Test get_rng_state/set_rng_state round-trip"""
        torch.neuron.manual_seed(42)
        state = torch.neuron.get_rng_state()

        torch.neuron.manual_seed(100)
        self.assertEqual(torch.neuron.initial_seed(), 100)

        torch.neuron.set_rng_state(state)
        self.assertEqual(torch.neuron.initial_seed(), 42)

    def test_neuron_rng_independent_from_cpu(self):
        """Test that Neuron RNG operations don't interfere with CPU RNG state"""
        # CPU-only baseline
        torch.manual_seed(42)
        cpu_only = [torch.randn(5).tolist() for _ in range(3)]

        # CPU + Neuron RNG operations interleaved
        torch.manual_seed(42)
        torch.neuron.manual_seed(123)
        _ = torch.neuron.get_rng_state()
        torch.neuron.seed()
        _ = torch.neuron.initial_seed()
        cpu_with_neuron = [torch.randn(5).tolist() for _ in range(3)]

        self.assertEqual(cpu_only, cpu_with_neuron)

    @pytest.mark.xfail(
        reason="torch.randn(device='neuron') falls back to CPU, consuming CPU RNG state"
    )
    def test_neuron_tensor_generation_independent_from_cpu(self):
        """Test that generating random tensors on Neuron doesn't affect CPU RNG.

        NOTE: Currently xfail because torch.randn(device="neuron") falls back to CPU,
        which consumes CPU RNG state. This test will pass once native Neuron random
        ops are implemented.
        """
        # CPU-only baseline
        torch.manual_seed(42)
        cpu_only = [torch.randn(5).tolist() for _ in range(3)]

        # CPU + Neuron tensor generation interleaved
        torch.manual_seed(42)
        torch.neuron.manual_seed(123)
        _ = torch.randn(100, device="neuron")  # Generate on Neuron first
        cpu_with_neuron = [torch.randn(5).tolist() for _ in range(3)]

        self.assertEqual(cpu_only, cpu_with_neuron)
