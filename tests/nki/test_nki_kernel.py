"""Tests for TorchNeuronNKIKernel."""

from unittest.mock import MagicMock, patch

import neuronxcc.nki.typing as nt
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx import TorchNeuronNKIKernel


class TestBasicExecution:
    """Test basic kernel execution functionality."""

    def test_simple_dma_copy(self):
        """Test a simple DMA copy kernel."""

        @TorchNeuronNKIKernel.trace
        def dma_copy(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Create input and output tensors
        x = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        # Execute kernel
        dma_copy(x, y)

        # Verify output matches input
        assert torch.allclose(x, y)

    def test_tensor_scale(self):
        """Test scaling a tensor by a constant."""

        @TorchNeuronNKIKernel.trace
        def tensor_scale(x, y: nt.mutable_tensor, scale):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x[0:128])
            y_tile = x_tile * scale
            nl.store(y[0:128], value=y_tile)
            return y

        x = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")
        scale = 2.5

        tensor_scale(x, y, scale)

        expected = x * scale
        assert torch.allclose(y, expected)

    def test_element_wise_add(self):
        """Test element-wise addition of two tensors."""

        @TorchNeuronNKIKernel.trace
        def tensor_add(x1, x2, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x1_tile = nl.load(x1[0:128])
            x2_tile = nl.load(x2[0:128])
            y_tile = x1_tile + x2_tile
            nl.store(y[0:128], value=y_tile)
            return y

        x1 = torch.randn(128, device="neuron")
        x2 = torch.randn(128, device="neuron")
        y = torch.empty(128, device="neuron")

        tensor_add(x1, x2, y)

        expected = x1 + x2
        assert torch.allclose(y, expected)

    def test_different_shapes(self):
        """Test kernels with different tensor shapes."""

        @TorchNeuronNKIKernel.trace
        def copy_2d(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Test various 2D shapes
        shapes = [(10, 10), (16, 32), (128, 64)]

        for shape in shapes:
            x = torch.randn(*shape, device="neuron")
            y = torch.empty(*shape, device="neuron")
            copy_2d(x, y)
            assert torch.allclose(x, y)

    def test_different_dtypes(self):
        """Test kernels with different data types."""

        @TorchNeuronNKIKernel.trace
        def copy_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Test various dtypes
        dtypes = [torch.float32, torch.float16, torch.int32]

        for dtype in dtypes:
            if dtype in [torch.int32]:
                x = torch.randint(0, 100, (64,), device="neuron", dtype=dtype)
            else:
                x = torch.randn(64, device="neuron", dtype=dtype)
            y = torch.empty(64, device="neuron", dtype=dtype)

            copy_kernel(x, y)
            assert torch.equal(x, y)


class TestCaching:
    """Test NEFF and model caching functionality."""

    def setup_method(self):
        """Clear caches before each test."""
        TorchNeuronNKIKernel.clear_all_caches()

    def test_neff_caching(self):
        """Test that NEFFs are cached for same shapes/dtypes."""

        @TorchNeuronNKIKernel.trace
        def cached_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            # Handle different sizes with slicing
            size = x.shape[0]
            if size <= 128:
                x_tile = nl.load(x)
                nl.store(y, value=x_tile)
            else:
                # Process in chunks of 128
                x_tile = nl.load(x[0:128])
                nl.store(y[0:128], value=x_tile)
            return y

        # First call - should compile
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")

        # Check cache before first call
        initial_cache_size = len(TorchNeuronNKIKernel._neff_cache)

        cached_kernel(x1, y1)

        # Check cache after first call
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_cache_size + 1

        # Second call with same shape - should use cache
        x2 = torch.randn(64, device="neuron")
        y2 = torch.empty(64, device="neuron")

        cached_kernel(x2, y2)

        # Cache size should not change
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_cache_size + 1

        # Third call with different shape - should compile new
        x3 = torch.randn(32, device="neuron")
        y3 = torch.empty(32, device="neuron")

        cached_kernel(x3, y3)

        # Cache size should increase
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_cache_size + 2

    def test_model_handle_caching(self):
        """Test that model handles are cached for identical NEFFs."""

        @TorchNeuronNKIKernel.trace
        def model_cached_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            y_tile = x_tile * 2.0
            nl.store(y, value=y_tile)
            return y

        # Check initial model cache
        initial_models = len(TorchNeuronNKIKernel._loaded_models)

        # First execution
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        model_cached_kernel(x1, y1)

        # Should have loaded one model
        assert len(TorchNeuronNKIKernel._loaded_models) == initial_models + 1

        # Second execution with same shape
        x2 = torch.randn(64, device="neuron")
        y2 = torch.empty(64, device="neuron")
        model_cached_kernel(x2, y2)

        # Should still have only one model loaded
        assert len(TorchNeuronNKIKernel._loaded_models) == initial_models + 1

    def test_cache_clearing(self):
        """Test cache clearing methods."""

        @TorchNeuronNKIKernel.trace
        def clear_test_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Execute to populate caches
        x = torch.randn(32, device="neuron")
        y = torch.empty(32, device="neuron")
        clear_test_kernel(x, y)

        # Verify caches are populated
        assert len(TorchNeuronNKIKernel._neff_cache) > 0
        assert len(TorchNeuronNKIKernel._loaded_models) > 0

        # Clear NEFF cache only
        TorchNeuronNKIKernel.clear_neff_cache()
        assert len(TorchNeuronNKIKernel._neff_cache) == 0
        assert len(TorchNeuronNKIKernel._loaded_models) > 0

        # Clear model cache only
        TorchNeuronNKIKernel.clear_model_cache()
        assert len(TorchNeuronNKIKernel._loaded_models) == 0


class TestMultipleKernels:
    """Test using multiple different kernels."""

    def test_multiple_kernel_definitions(self):
        """Test that multiple kernels can coexist."""

        @TorchNeuronNKIKernel.trace
        def kernel_add(x, y, out: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            y_tile = nl.load(y)
            result = x_tile + y_tile
            nl.store(out, value=result)
            return out

        @TorchNeuronNKIKernel.trace
        def kernel_multiply(x, y, out: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            y_tile = nl.load(y)
            result = x_tile * y_tile
            nl.store(out, value=result)
            return out

        # Test add kernel
        x1 = torch.randn(64, device="neuron")
        y1 = torch.randn(64, device="neuron")
        out1 = torch.empty(64, device="neuron")
        kernel_add(x1, y1, out1)
        assert torch.allclose(out1, x1 + y1)

        # Test multiply kernel
        x2 = torch.randn(64, device="neuron")
        y2 = torch.randn(64, device="neuron")
        out2 = torch.empty(64, device="neuron")
        kernel_multiply(x2, y2, out2)
        assert torch.allclose(out2, x2 * y2)

    def test_kernel_with_different_signatures(self):
        """Test kernels with different parameter signatures."""

        @TorchNeuronNKIKernel.trace
        def unary_kernel(x, out: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            result = x_tile * x_tile  # square
            nl.store(out, value=result)
            return out

        @TorchNeuronNKIKernel.trace
        def ternary_kernel(x, y, z, out: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            y_tile = nl.load(y)
            z_tile = nl.load(z)
            result = x_tile + y_tile * z_tile
            nl.store(out, value=result)
            return out

        # Test unary kernel
        x = torch.randn(32, device="neuron")
        out1 = torch.empty(32, device="neuron")
        unary_kernel(x, out1)
        assert torch.allclose(out1, x * x)

        # Test ternary kernel
        y = torch.randn(32, device="neuron")
        z = torch.randn(32, device="neuron")
        out2 = torch.empty(32, device="neuron")
        ternary_kernel(x, y, z, out2)
        assert torch.allclose(out2, x + y * z)


class TestIRCaching:
    """Test IR caching functionality."""

    def setup_method(self):
        """Clear all caches before each test."""
        TorchNeuronNKIKernel.clear_all_caches()

    def test_ir_cache_hit_on_same_shapes(self):
        """Test that IR is cached and reused for same tensor shapes."""

        @TorchNeuronNKIKernel.trace
        def cached_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # First call - should populate IR cache
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")

        initial_ir_cache_size = len(TorchNeuronNKIKernel._ir_cache)
        cached_kernel(x1, y1)

        # Verify IR cache was populated
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 1

        # Second call with same shape - should hit IR cache
        x2 = torch.randn(64, device="neuron")
        y2 = torch.empty(64, device="neuron")
        cached_kernel(x2, y2)

        # IR cache size should not increase
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 1

    def test_ir_cache_miss_on_different_shapes(self):
        """Test that different shapes create new IR cache entries."""

        @TorchNeuronNKIKernel.trace
        def shape_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            size = x.shape[0]
            if size <= 128:
                x_tile = nl.load(x)
                nl.store(y, value=x_tile)
            else:
                x_tile = nl.load(x[0:128])
                nl.store(y[0:128], value=x_tile)
            return y

        # First call with shape [64]
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")
        shape_kernel(x1, y1)

        initial_cache_size = len(TorchNeuronNKIKernel._ir_cache)

        # Second call with different shape [32]
        x2 = torch.randn(32, device="neuron")
        y2 = torch.empty(32, device="neuron")
        shape_kernel(x2, y2)

        # IR cache should have new entry
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_cache_size + 1

    def test_ir_cache_preserves_return_structure(self):
        """Test that cached IR returns correct structure (single vs tuple)."""

        @TorchNeuronNKIKernel.trace
        def single_return_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        @TorchNeuronNKIKernel.trace
        def tuple_return_kernel(x, y1: nt.mutable_tensor, y2: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y1, value=x_tile)
            nl.store(y2, value=x_tile * 2.0)
            return y1, y2

        # Test single return
        x = torch.randn(64, device="neuron")
        y = torch.empty(64, device="neuron")
        result1 = single_return_kernel(x, y)
        assert isinstance(result1, torch.Tensor) or result1 is None

        # Second call should use cache and preserve structure
        result2 = single_return_kernel(x, y)
        assert type(result1) is type(result2)

        # Test tuple return
        y1 = torch.empty(64, device="neuron")
        y2 = torch.empty(64, device="neuron")
        result3 = tuple_return_kernel(x, y1, y2)

        # Second call should use cache and preserve tuple structure
        result4 = tuple_return_kernel(x, y1, y2)
        assert type(result3) is type(result4)

    def test_ir_cache_with_neff_cache(self):
        """Test IR cache works correctly with NEFF cache."""

        @TorchNeuronNKIKernel.trace
        def dual_cache_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            size = x.shape[0]
            if size <= 128:
                x_tile = nl.load(x)
                nl.store(y, value=x_tile)
            else:
                x_tile = nl.load(x[0:128])
                nl.store(y[0:128], value=x_tile)
            return y

        # First call: IR cache miss, NEFF cache miss
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")

        initial_ir_size = len(TorchNeuronNKIKernel._ir_cache)
        initial_neff_size = len(TorchNeuronNKIKernel._neff_cache)

        dual_cache_kernel(x1, y1)

        # Both caches should have new entries
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_size + 1
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_neff_size + 1

        # Second call: IR cache hit, NEFF cache hit
        x2 = torch.randn(64, device="neuron")
        y2 = torch.empty(64, device="neuron")
        dual_cache_kernel(x2, y2)

        # Cache sizes should not change
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_size + 1
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_neff_size + 1

        # Third call with different shape: IR cache miss, NEFF cache miss
        x3 = torch.randn(32, device="neuron")
        y3 = torch.empty(32, device="neuron")
        dual_cache_kernel(x3, y3)

        # Both caches should have new entries
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_size + 2
        assert len(TorchNeuronNKIKernel._neff_cache) == initial_neff_size + 2

    def test_clear_ir_cache(self):
        """Test IR cache clearing."""

        @TorchNeuronNKIKernel.trace
        def clear_test_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Populate caches
        x = torch.randn(64, device="neuron")
        y = torch.empty(64, device="neuron")
        clear_test_kernel(x, y)

        # Verify all caches are populated
        assert len(TorchNeuronNKIKernel._ir_cache) > 0
        assert len(TorchNeuronNKIKernel._neff_cache) > 0
        assert len(TorchNeuronNKIKernel._loaded_models) > 0

        # Clear IR cache only
        TorchNeuronNKIKernel.clear_ir_cache()

        # Verify IR cache is empty, others intact
        assert len(TorchNeuronNKIKernel._ir_cache) == 0
        assert len(TorchNeuronNKIKernel._neff_cache) > 0
        assert len(TorchNeuronNKIKernel._loaded_models) > 0

    def test_clear_all_caches_includes_ir(self):
        """Test that clear_all_caches clears IR cache."""

        @TorchNeuronNKIKernel.trace
        def all_cache_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Populate all caches
        x = torch.randn(64, device="neuron")
        y = torch.empty(64, device="neuron")
        all_cache_kernel(x, y)

        # Verify all caches are populated
        assert len(TorchNeuronNKIKernel._ir_cache) > 0
        assert len(TorchNeuronNKIKernel._neff_cache) > 0
        assert len(TorchNeuronNKIKernel._loaded_models) > 0

        # Clear all caches
        TorchNeuronNKIKernel.clear_all_caches()

        # Verify all caches are empty
        assert len(TorchNeuronNKIKernel._ir_cache) == 0
        assert len(TorchNeuronNKIKernel._neff_cache) == 0
        assert len(TorchNeuronNKIKernel._loaded_models) == 0

    def test_ir_cache_skips_tracing_on_cache_hit(self):
        """Test that IR cache hit skips expensive tracing (specialize_and_call).

        This test verifies the optimization where:
        - Call 1: Goes through call_impl → specialize_and_call (tracing)
        - Calls 2-4: Skip specialize_and_call, use cached IR directly via post_process_call
        """
        # Track method calls using a call counter
        call_counts = {"specialize_and_call": 0, "post_process_call": 0}

        # Save original methods
        original_specialize = TorchNeuronNKIKernel.specialize_and_call
        original_post_process = TorchNeuronNKIKernel.post_process_call

        def tracked_specialize(self, boundargs):
            call_counts["specialize_and_call"] += 1
            return original_specialize(self, boundargs)

        def tracked_post_process(self, ir, ret, boundargs):
            call_counts["post_process_call"] += 1
            return original_post_process(self, ir, ret, boundargs)

        # Monkey-patch the methods
        TorchNeuronNKIKernel.specialize_and_call = tracked_specialize
        TorchNeuronNKIKernel.post_process_call = tracked_post_process

        try:

            @TorchNeuronNKIKernel.trace
            def trace_test_kernel(x, y: nt.mutable_tensor):
                import neuronxcc.nki.language as nl

                x_tile = nl.load(x)
                nl.store(y, value=x_tile)
                return y

            x = torch.randn(64, device="neuron")
            y = torch.empty(64, device="neuron")

            # Call 1: Should go through full tracing flow
            call_counts["specialize_and_call"] = 0
            call_counts["post_process_call"] = 0
            trace_test_kernel(x, y)
            assert (
                call_counts["specialize_and_call"] == 1
            ), "Call 1 should invoke specialize_and_call (tracing)"
            assert call_counts["post_process_call"] == 1, "Call 1 should invoke post_process_call"

            # Call 2: Should use IR cache, skip specialize_and_call
            call_counts["specialize_and_call"] = 0
            call_counts["post_process_call"] = 0
            trace_test_kernel(x, y)
            assert (
                call_counts["specialize_and_call"] == 0
            ), "Call 2 should NOT invoke specialize_and_call (cache hit)"
            assert (
                call_counts["post_process_call"] == 1
            ), "Call 2 should invoke post_process_call with cached IR"

            # Call 3: Should also use IR cache
            call_counts["specialize_and_call"] = 0
            call_counts["post_process_call"] = 0
            trace_test_kernel(x, y)
            assert (
                call_counts["specialize_and_call"] == 0
            ), "Call 3 should NOT invoke specialize_and_call (cache hit)"
            assert (
                call_counts["post_process_call"] == 1
            ), "Call 3 should invoke post_process_call with cached IR"

            # Call 4: Should also use IR cache
            call_counts["specialize_and_call"] = 0
            call_counts["post_process_call"] = 0
            trace_test_kernel(x, y)
            assert (
                call_counts["specialize_and_call"] == 0
            ), "Call 4 should NOT invoke specialize_and_call (cache hit)"
            assert (
                call_counts["post_process_call"] == 1
            ), "Call 4 should invoke post_process_call with cached IR"

        finally:
            # Restore original methods
            TorchNeuronNKIKernel.specialize_and_call = original_specialize
            TorchNeuronNKIKernel.post_process_call = original_post_process


class TestErrorHandling:
    """Test error handling and validation."""

    @assert_raises(ValueError, match="expects tensors on Neuron device")
    def test_cpu_tensor_error(self):
        """Test that CPU tensors raise appropriate error."""

        @TorchNeuronNKIKernel.trace
        def error_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Create CPU tensors
        x = torch.randn(64)  # CPU tensor
        y = torch.empty(64)  # CPU tensor

        # Should raise ValueError about device
        error_kernel(x, y)

    @assert_raises(ValueError, match="expects tensors on Neuron device")
    def test_mixed_device_error(self):
        """Test error when tensors are on different devices."""

        @TorchNeuronNKIKernel.trace
        def mixed_device_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # One on Neuron, one on CPU
        x = torch.randn(64, device="neuron")
        y = torch.empty(64)  # CPU

        mixed_device_kernel(x, y)

    @assert_raises(TypeError)
    def test_non_tensor_input(self):
        """Test error with non-tensor inputs."""

        @TorchNeuronNKIKernel.trace
        def non_tensor_kernel(x, y: nt.mutable_tensor):
            import neuronxcc.nki.language as nl

            x_tile = nl.load(x)
            nl.store(y, value=x_tile)
            return y

        # Pass numpy array instead of tensor
        import numpy as np

        x = np.random.randn(64)
        y = torch.empty(64, device="neuron")

        # The kernel compilation will fail since numpy arrays aren't supported
        non_tensor_kernel(x, y)


class TestNKIKernelKey:
    """Test NKIKernelKey argument handling and caching."""

    def setup_method(self):
        """Clear caches before each test."""
        TorchNeuronNKIKernel.clear_all_caches()

    def test_tensor_arguments(self):
        """Test that tensor arguments are properly captured in cache key."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        # Mock boundargs with tensor arguments
        boundargs = SimpleNamespace()
        boundargs.arguments = {
            "x": torch.randn(64, device="neuron", dtype=torch.float32),
            "y": torch.empty(32, 16, device="neuron", dtype=torch.float16),
        }

        key = NKIKernelKey.from_boundargs("test_func", boundargs, "compile_opts")

        assert key.func_name == "test_func"
        assert key.compile_opts == "compile_opts"
        assert len(key.input_tensors) == 2
        assert len(key.input_scalars) == 0
        assert len(key.input_tuples) == 0

        # Check tensor info is captured correctly
        tensor_info = {name: (shape, dtype) for name, shape, dtype in key.input_tensors}
        assert tensor_info["x"] == ((64,), "torch.float32")
        assert tensor_info["y"] == ((32, 16), "torch.float16")

    def test_scalar_arguments(self):
        """Test that scalar arguments are properly captured in cache key."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        boundargs = SimpleNamespace()
        boundargs.arguments = {
            "scale": 2.5,
            "count": 42,
            "enabled": True,
            "name": "test_kernel",
            "seed": None,
        }

        key = NKIKernelKey.from_boundargs("test_func", boundargs)

        assert len(key.input_tensors) == 0
        assert len(key.input_scalars) == 5
        assert len(key.input_tuples) == 0

        # Check scalar info is captured correctly
        scalar_info = {name: (type_name, value) for name, type_name, value in key.input_scalars}
        assert scalar_info["scale"] == ("float", "2.5")
        assert scalar_info["count"] == ("int", "42")
        assert scalar_info["enabled"] == ("bool", "True")
        assert scalar_info["name"] == ("str", "test_kernel")
        assert scalar_info["seed"] == ("NoneType", "None")

    def test_tuple_arguments(self):
        """Test that tuple arguments are properly captured in cache key."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        boundargs = SimpleNamespace()
        boundargs.arguments = {
            "shape": (64, 32),
            "strides": (1, 2, 3),
            "empty_tuple": (),
        }

        key = NKIKernelKey.from_boundargs("test_func", boundargs)

        assert len(key.input_tensors) == 0
        assert len(key.input_scalars) == 0
        assert len(key.input_tuples) == 3

        # Check tuple info is captured correctly
        tuple_info = dict(key.input_tuples)
        assert tuple_info["shape"] == (64, 32)
        assert tuple_info["strides"] == (1, 2, 3)
        assert tuple_info["empty_tuple"] == ()

    def test_custom_object_arguments(self):
        """Test that custom objects are properly handled in cache key."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        # Create a mock custom object
        class MockConfig:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"MockConfig(value={self.value})"

        config = MockConfig(42)

        # Test successful custom object
        boundargs_good = SimpleNamespace()
        boundargs_good.arguments = {"config": config}

        key = NKIKernelKey.from_boundargs("test_func", boundargs_good)

        assert len(key.input_tensors) == 0
        assert len(key.input_scalars) == 1
        assert len(key.input_tuples) == 0

        # Check custom object handling
        scalar_info = {name: (type_name, value) for name, type_name, value in key.input_scalars}
        assert scalar_info["config"] == ("MockConfig", "MockConfig(value=42)")

        # Test object that fails str() conversion - should raise error
        class BadStrObject:
            def __str__(self):
                raise RuntimeError("str() failed")

        bad_obj = BadStrObject()
        boundargs_bad = SimpleNamespace()
        boundargs_bad.arguments = {"bad_obj": bad_obj}

        # Should raise TypeError for objects with broken __str__()
        with pytest.raises(
            TypeError, match="Cannot create cache key for argument 'bad_obj'"
        ) as exc_info:
            NKIKernelKey.from_boundargs("test_func", boundargs_bad)
            torch_neuronx.synchronize()

        # Verify exception chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_mixed_arguments(self):
        """Test cache key with mixed argument types."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        class FlashConfig:
            def __init__(self, block_size):
                self.block_size = block_size

            def __str__(self):
                return f"FlashConfig(block_size={self.block_size})"

        boundargs = SimpleNamespace()
        boundargs.arguments = {
            "query": torch.randn(128, 64, device="neuron"),
            "key": torch.randn(128, 64, device="neuron"),
            "scale": 0.125,
            "dropout_p": 0.1,
            "is_causal": True,
            "shape": (128, 64),
            "config": FlashConfig(64),
            "seed": None,
        }

        key = NKIKernelKey.from_boundargs("attention_kernel", boundargs, "-O2")

        assert key.func_name == "attention_kernel"
        assert key.compile_opts == "-O2"
        assert len(key.input_tensors) == 2
        assert len(key.input_scalars) == 5
        assert len(key.input_tuples) == 1

        # Verify all types are handled correctly
        tensor_names = {name for name, _, _ in key.input_tensors}
        assert tensor_names == {"query", "key"}

        scalar_names = {name for name, _, _ in key.input_scalars}
        assert scalar_names == {"scale", "dropout_p", "is_causal", "config", "seed"}

        tuple_names = {name for name, _ in key.input_tuples}
        assert tuple_names == {"shape"}

    def test_cache_key_equality(self):
        """Test that identical arguments produce identical cache keys."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        # Create two identical boundargs
        def create_boundargs():
            boundargs = SimpleNamespace()
            boundargs.arguments = {
                "x": torch.randn(64, device="neuron"),
                "scale": 2.0,
                "shape": (64, 32),
            }
            return boundargs

        key1 = NKIKernelKey.from_boundargs("test", create_boundargs())
        key2 = NKIKernelKey.from_boundargs("test", create_boundargs())

        # Keys should be equal (same structure, different tensor instances)
        assert key1.func_name == key2.func_name
        assert key1.compile_opts == key2.compile_opts
        assert key1.input_scalars == key2.input_scalars
        assert key1.input_tuples == key2.input_tuples
        # Tensor shapes and dtypes should match
        assert len(key1.input_tensors) == len(key2.input_tensors)

    def test_cache_key_differences(self):
        """Test that different arguments produce different cache keys."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        # Different function names
        boundargs = SimpleNamespace()
        boundargs.arguments = {"x": torch.randn(64, device="neuron")}

        key1 = NKIKernelKey.from_boundargs("func1", boundargs)
        key2 = NKIKernelKey.from_boundargs("func2", boundargs)
        assert key1 != key2

        # Different compile options
        key3 = NKIKernelKey.from_boundargs("func1", boundargs, "-O1")
        key4 = NKIKernelKey.from_boundargs("func1", boundargs, "-O2")
        assert key3 != key4

        # Different tensor shapes
        boundargs1 = SimpleNamespace()
        boundargs1.arguments = {"x": torch.randn(64, device="neuron")}
        boundargs2 = SimpleNamespace()
        boundargs2.arguments = {"x": torch.randn(32, device="neuron")}

        key5 = NKIKernelKey.from_boundargs("func", boundargs1)
        key6 = NKIKernelKey.from_boundargs("func", boundargs2)
        assert key5 != key6

        # Different scalar values
        boundargs3 = SimpleNamespace()
        boundargs3.arguments = {"scale": 1.0}
        boundargs4 = SimpleNamespace()
        boundargs4.arguments = {"scale": 2.0}

        key7 = NKIKernelKey.from_boundargs("func", boundargs3)
        key8 = NKIKernelKey.from_boundargs("func", boundargs4)
        assert key7 != key8

    def test_cache_key_sorting(self):
        """Test that argument order doesn't affect cache key (due to sorting)."""
        from types import SimpleNamespace

        from torch_neuronx.nki_kernel import NKIKernelKey

        # Same arguments in different order
        boundargs1 = SimpleNamespace()
        boundargs1.arguments = {
            "x": torch.randn(64, device="neuron"),
            "y": torch.randn(32, device="neuron"),
            "scale": 2.0,
            "offset": 1.0,
        }

        boundargs2 = SimpleNamespace()
        boundargs2.arguments = {
            "offset": 1.0,
            "y": torch.randn(32, device="neuron"),
            "scale": 2.0,
            "x": torch.randn(64, device="neuron"),
        }

        key1 = NKIKernelKey.from_boundargs("test", boundargs1)
        key2 = NKIKernelKey.from_boundargs("test", boundargs2)

        # Should be equal due to sorting
        assert key1.input_tensors == key2.input_tensors
        assert key1.input_scalars == key2.input_scalars

    def test_ir_caching_with_different_argument_types(self):
        """Test that IR caching works correctly with different argument types."""

        @TorchNeuronNKIKernel.trace
        def mixed_args_kernel(x, y: nt.mutable_tensor, scale, shape, enabled):
            import neuronxcc.nki.language as nl

            if enabled:
                x_tile = nl.load(x)
                y_tile = x_tile * scale
                nl.store(y, value=y_tile)
            return y

        # First call - should populate IR cache
        x1 = torch.randn(64, device="neuron")
        y1 = torch.empty(64, device="neuron")

        initial_ir_cache_size = len(TorchNeuronNKIKernel._ir_cache)
        mixed_args_kernel(x1, y1, 2.0, (64,), True)

        # IR cache should have new entry
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 1

        # Second call with same arguments - should hit IR cache
        x2 = torch.randn(64, device="neuron")
        y2 = torch.empty(64, device="neuron")
        mixed_args_kernel(x2, y2, 2.0, (64,), True)

        # IR cache size should not change
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 1

        # Third call with different scalar - should miss IR cache
        x3 = torch.randn(64, device="neuron")
        y3 = torch.empty(64, device="neuron")
        mixed_args_kernel(x3, y3, 3.0, (64,), True)  # Different scale

        # IR cache should have new entry
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 2

        # Fourth call with different tuple - should miss IR cache
        x4 = torch.randn(64, device="neuron")
        y4 = torch.empty(64, device="neuron")
        mixed_args_kernel(x4, y4, 2.0, (32,), True)  # Different shape

        # IR cache should have new entry
        assert len(TorchNeuronNKIKernel._ir_cache) == initial_ir_cache_size + 3
