"""Correctness tests for IR cache functionality."""

import pytest
import torch

from torch_neuron import TorchNeuronNKIKernel


class TestIRCacheCorrectness:
    """Test IR cache correctness across various scenarios."""

    def setup_method(self):
        """Clear all caches before each test."""
        TorchNeuronNKIKernel.clear_all_caches()

    def test_ir_cache_numerical_correctness(self):
        """Verify cached IR produces identical results to fresh trace."""

        @TorchNeuronNKIKernel.trace
        def numerical_kernel(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            result = x_tile * 2.5 + 1.0
            nl.store(y, value=result)

        # First call - populate caches
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        numerical_kernel(x1, y1)
        output1 = y1.clone()

        # Clear NEFF cache but keep IR cache
        TorchNeuronNKIKernel.clear_neff_cache()
        TorchNeuronNKIKernel.clear_model_cache()

        # Second call - should use cached IR but recompile NEFF
        x2 = x1.clone()  # Same values
        y2 = torch.empty(64, device="neuron")
        numerical_kernel(x2, y2)
        output2 = y2.clone()

        # Results should be bit-exact
        assert torch.equal(output1, output2), "Cached IR produced different results"

    def test_ir_cache_with_different_tensor_values(self):
        """Test that IR cache works with different tensor values (same shape)."""

        @TorchNeuronNKIKernel.trace
        def value_kernel(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            result = x_tile * 3.0
            nl.store(y, value=result)

        # First call with tensor1
        x1 = torch.ones(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        value_kernel(x1, y1)

        # Verify IR cache was populated
        assert len(TorchNeuronNKIKernel._ir_cache) == 1

        # Second call with tensor2 (same shape, different values)
        x2 = torch.ones(64, device="neuron") * 2.0
        y2 = torch.empty(64, device="neuron")
        value_kernel(x2, y2)

        # Should use same cached IR
        assert len(TorchNeuronNKIKernel._ir_cache) == 1

        # Outputs should be different (based on input values)
        expected1 = x1 * 3.0
        expected2 = x2 * 3.0
        assert torch.allclose(y1, expected1)
        assert torch.allclose(y2, expected2)
        assert not torch.allclose(y1, y2)

    def test_ir_cache_key_generation_uniqueness(self):
        """Test that cache keys are unique for different configurations."""

        @TorchNeuronNKIKernel.trace
        def key_kernel(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)

        # Test different shapes
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        key_kernel(x1, y1)

        x2 = torch.randn(32, device="neuron")
        y2 = torch.empty(32, device="neuron")
        key_kernel(x2, y2)

        # Should have 2 cache entries for different shapes
        assert len(TorchNeuronNKIKernel._ir_cache) == 2

        # Test different dtypes
        x3 = torch.randn(64, device="neuron", dtype=torch.float16)
        y3 = torch.empty(64, device="neuron", dtype=torch.float16)
        key_kernel(x3, y3)

        # Should have 3 cache entries (different dtype)
        assert len(TorchNeuronNKIKernel._ir_cache) == 3

    def test_ir_cache_with_multiple_kernels(self):
        """Test IR cache with multiple different kernel types."""

        @TorchNeuronNKIKernel.trace
        def kernel1(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile * 2.0)

        @TorchNeuronNKIKernel.trace
        def kernel2(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile + 1.0)

        @TorchNeuronNKIKernel.trace
        def kernel3(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile * x_tile)

        # Call each kernel multiple times
        x = torch.randn(64, device="neuron")
        y = torch.empty(64, device="neuron")

        # First calls - populate cache
        kernel1(x, y)
        output1_first = y.clone()

        kernel2(x, y)
        output2_first = y.clone()

        kernel3(x, y)
        output3_first = y.clone()

        # Should have 3 cache entries (one per kernel)
        assert len(TorchNeuronNKIKernel._ir_cache) == 3

        # Second calls - use cache
        kernel1(x, y)
        output1_second = y.clone()

        kernel2(x, y)
        output2_second = y.clone()

        kernel3(x, y)
        output3_second = y.clone()

        # Cache size should not change
        assert len(TorchNeuronNKIKernel._ir_cache) == 3

        # Verify no cross-contamination
        assert torch.allclose(output1_first, output1_second)
        assert torch.allclose(output2_first, output2_second)
        assert torch.allclose(output3_first, output3_second)

        # Verify outputs are different from each other
        assert not torch.allclose(output1_first, output2_first)
        assert not torch.allclose(output2_first, output3_first)
        assert not torch.allclose(output1_first, output3_first)

    def test_ir_cache_with_scalar_parameters(self):
        """Test IR cache with kernels that have scalar parameters."""

        @TorchNeuronNKIKernel.trace
        def scalar_kernel(x, y, scale):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            result = x_tile * scale
            nl.store(y, value=result)

        x = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        y2 = torch.empty(64, device="neuron")

        # Call with different scalar values
        scalar_kernel(x, y1, 2.0)
        scalar_kernel(x, y2, 3.0)

        # Outputs should be different based on scalar parameter
        expected1 = x * 2.0
        expected2 = x * 3.0
        assert torch.allclose(y1, expected1), f"y1 mismatch: got {y1[:5]}, expected {expected1[:5]}"
        assert torch.allclose(y2, expected2), f"y2 mismatch: got {y2[:5]}, expected {expected2[:5]}"

    def test_ir_cache_with_2d_tensors(self):
        """Test IR cache correctness with 2D tensors."""

        @TorchNeuronNKIKernel.trace
        def tensor_2d_kernel(x, y):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)

        # First call
        x1 = torch.randn(16, 32, device="neuron")
        y1 = torch.empty(16, 32, device="neuron")
        tensor_2d_kernel(x1, y1)

        # Second call with same shape
        x2 = torch.randn(16, 32, device="neuron")
        y2 = torch.empty(16, 32, device="neuron")
        tensor_2d_kernel(x2, y2)

        # Should use cached IR
        assert len(TorchNeuronNKIKernel._ir_cache) == 1

        # Verify correctness
        assert torch.allclose(y1, x1)
        assert torch.allclose(y2, x2)
