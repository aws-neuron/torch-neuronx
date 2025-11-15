"""Test base kernel functionality."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.kernels import BaseNeuronKernel


class TestBaseKernel:
    """Test base kernel functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        kernel = BaseNeuronKernel()

        # Create test tensors
        t1 = torch.randn(2, 3, dtype=torch.float32)
        t2 = torch.randn(3, 4, dtype=torch.float32)

        # Generate cache key
        key1 = kernel.get_cache_key("test_op", t1, t2)

        # Key should contain operation name and tensor info
        assert "test_op" in key1
        assert "2, 3" in key1  # shape of t1
        assert "3, 4" in key1  # shape of t2
        assert "torch.float32" in key1

        # Same inputs should generate same key
        key2 = kernel.get_cache_key("test_op", t1, t2)
        assert key1 == key2

        # Different shapes should generate different keys
        t3 = torch.randn(2, 5, dtype=torch.float32)
        key3 = kernel.get_cache_key("test_op", t1, t3)
        assert key1 != key3

    def test_cache_key_generation_with_kwargs(self):
        """Test cache key generation."""
        kernel = BaseNeuronKernel()

        # Create test tensors
        t1 = torch.randn(2, 3, dtype=torch.float32)
        t2 = torch.randn(3, 4, dtype=torch.float32)
        kwargs1 = {"value": 1}

        # Generate cache key
        key1 = kernel.get_cache_key("test_op", t1, t2, kwargs=kwargs1)

        # Key should contain operation name and tensor info
        assert "test_op" in key1
        assert "2, 3" in key1  # shape of t1
        assert "3, 4" in key1  # shape of t2
        assert "torch.float32" in key1
        assert "value" in key1

        # Different non-static kwarg with the same dtype should get the same key
        kwargs2 = {"value": 2}
        key2 = kernel.get_cache_key("test_op", t1, t2, kwargs=kwargs2)
        assert key1 == key2

        # Non-static kwarg with different dtype shoudl get different keys
        kwargs3 = {"value": 2.0}
        key3 = kernel.get_cache_key("test_op", t1, t2, kwargs=kwargs3)
        assert key1 != key3

    def test_tensor_validation(self):
        """Test tensor validation."""
        kernel = BaseNeuronKernel()

        # Valid neuron tensor
        neuron_tensor = torch.ones(2, 2).to("neuron")
        kernel._validate_tensor(neuron_tensor, "test")  # Should not raise

    @assert_raises(ValueError, match="Expected tensor on Neuron device")
    def test_tensor_validation_cpu_tensor_error(self):
        """Test tensor validation with CPU tensor."""
        kernel = BaseNeuronKernel()
        cpu_tensor = torch.ones(2, 2)
        kernel._validate_tensor(cpu_tensor, "test")

    @assert_raises(TypeError, match="Expected PyTorch tensor")
    def test_tensor_validation_non_tensor_error(self):
        """Test tensor validation with non-tensor."""
        kernel = BaseNeuronKernel()
        kernel._validate_tensor([1, 2, 3], "test")

    def test_cache_clearing(self):
        """Test cache clearing methods."""
        # Create new kernel instances to ensure clean state
        kernel1 = BaseNeuronKernel()
        kernel2 = BaseNeuronKernel()

        # Add some dummy cache entries
        BaseNeuronKernel._neff_cache["test_key1"] = ("test.neff", b"test_data1")
        BaseNeuronKernel._neff_cache["test_key2"] = ("test2.neff", b"test_data2")

        assert len(BaseNeuronKernel._neff_cache) >= 2

        # Clear NEFF cache
        BaseNeuronKernel.clear_neff_cache()
        assert len(BaseNeuronKernel._neff_cache) == 0

        # Both instances should see the cleared cache
        assert len(kernel1._neff_cache) == 0
        assert len(kernel2._neff_cache) == 0
