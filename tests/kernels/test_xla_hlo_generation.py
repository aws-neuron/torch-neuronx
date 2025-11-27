"""Test XLA HLO generation without full compilation."""

import jax
import jax.numpy as jnp
import pytest
import torch

import torch_neuronx


class TestXLAHLOGeneration:
    """Test XLA HLO generation functionality."""

    def test_hlo_generation_simple_add(self):
        """Test HLO generation for simple addition."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Define a simple JAX function
        def add_fn(x, y):
            return x + y

        # Create kernel
        kernel = TorchNeuronXLAKernel(add_fn, "add_op")

        # Create sample tensors (CPU is fine for HLO generation)
        a = torch.ones(4, 4, dtype=torch.float32)
        b = torch.ones(4, 4, dtype=torch.float32)

        # Generate HLO
        hlo = kernel.compile_jax_to_hlo(add_fn, a, b)
        hlo_text = hlo.to_string()

        # Verify HLO contains expected operations
        assert "HloModule" in hlo_text
        assert "add" in hlo_text.lower()
        assert "f32[4,4]" in hlo_text  # Shape and dtype info

        print("Generated HLO:")
        print(hlo_text[:500] + "..." if len(hlo_text) > 500 else hlo_text)

    def test_hlo_generation_matmul(self):
        """Test HLO generation for matrix multiplication."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Define matmul function
        def matmul_fn(x, y):
            return jnp.matmul(x, y)

        # Create kernel
        kernel = TorchNeuronXLAKernel(matmul_fn, "matmul_op")

        # Create sample tensors with compatible shapes
        a = torch.randn(8, 16, dtype=torch.float32)
        b = torch.randn(16, 32, dtype=torch.float32)

        # Generate HLO
        hlo = kernel.compile_jax_to_hlo(matmul_fn, a, b)
        hlo_text = hlo.to_string()

        # Verify HLO contains expected operations
        assert "HloModule" in hlo_text
        assert "dot" in hlo_text.lower()
        assert "f32[8,16]" in hlo_text  # Input shape
        assert "f32[16,32]" in hlo_text  # Input shape
        assert "f32[8,32]" in hlo_text  # Output shape

    def test_hlo_generation_different_dtypes(self):
        """Test HLO generation with different data types."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def multiply_fn(x, y):
            return x * y

        kernel = TorchNeuronXLAKernel(multiply_fn, "multiply_op")

        # Test with int32
        a_int = torch.ones(2, 2, dtype=torch.int32)
        b_int = torch.ones(2, 2, dtype=torch.int32)
        hlo = kernel.compile_jax_to_hlo(multiply_fn, a_int, b_int)
        assert "s32[2,2]" in hlo.to_string()

        # Test with float16
        a_f16 = torch.ones(2, 2, dtype=torch.float16)
        b_f16 = torch.ones(2, 2, dtype=torch.float16)
        hlo = kernel.compile_jax_to_hlo(multiply_fn, a_f16, b_f16)
        assert "f16[2,2]" in hlo.to_string()
