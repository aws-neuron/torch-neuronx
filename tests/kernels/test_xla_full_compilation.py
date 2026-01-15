"""Test full XLA compilation and execution path."""

import os

import pytest
import torch

import torch_neuronx


class TestXLAFullCompilationBasic:
    """Test XLA kernel compilation to NEFF and execution."""

    def test_simple_add_full_compilation(self):
        """Test full compilation and execution of simple addition."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Initialize neuron runtime
        torch_neuronx._lazy_init()

        # Define a simple JAX function
        def add_fn(x, y):
            return x + y

        # Create kernel
        kernel = TorchNeuronXLAKernel(add_fn, "add_op")

        # Create test tensors on neuron device
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Pre-allocate output
        output = torch.empty(4, 4, dtype=torch.float32).to("neuron")

        # Execute kernel - this should compile HLO -> NEFF and execute
        result = kernel(a, b, output=output)

        # Verify result
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0",
    reason="Legacy compilation logic requires legacy execution mode",
)
class TestXLAFullCompilationLegacy:
    """Test XLA kernel compilation with legacy execution mode."""

    def test_legacy_compilation_caching(self):
        """Test that NEFF compilation is cached properly."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Clear caches
        TorchNeuronXLAKernel.clear_all_caches()

        def multiply_fn(x, y):
            return x * y

        kernel = TorchNeuronXLAKernel(multiply_fn, "multiply_op")

        # First execution - should compile
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to("neuron")
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32).to("neuron")
        output1 = torch.empty_like(a)

        initial_cache_size = len(kernel._neff_cache)
        result1 = kernel(a, b, output=output1)

        # Cache should have one entry
        assert len(kernel._neff_cache) == initial_cache_size + 1

        # Second execution with same shapes - should use cache
        c = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32).to("neuron")
        d = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float32).to("neuron")
        output2 = torch.empty_like(c)

        result2 = kernel(c, d, output=output2)

        # Cache size should not increase
        assert len(kernel._neff_cache) == initial_cache_size + 1

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
        torch.testing.assert_close(result2.cpu(), torch.tensor([[15.0, 18.0], [21.0, 24.0]]))

    def test_legacy_different_shapes_compile_separately(self):
        """Test that different shapes result in separate compilations."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Clear caches
        TorchNeuronXLAKernel.clear_all_caches()

        def add_fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(add_fn, "add_op")

        # First shape
        a1 = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b1 = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        output1 = torch.empty_like(a1)

        initial_cache_size = len(kernel._neff_cache)
        result1 = kernel(a1, b1, output=output1)

        # Should have one cache entry
        assert len(kernel._neff_cache) == initial_cache_size + 1

        # Different shape
        a2 = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        b2 = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        output2 = torch.empty_like(a2)

        result2 = kernel(a2, b2, output=output2)

        # Should have two cache entries now
        assert len(kernel._neff_cache) == initial_cache_size + 2

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 2)
        torch.testing.assert_close(result2.cpu(), torch.ones(3, 3) * 2)


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "1",
    reason="Compilation logic requires async execution mode",
)
class TestXLAFullCompilation:
    """Test XLA kernel compilation with async execution mode."""

    def setup_method(self):
        """Set env var to collect compilation stats/metrics"""
        # Set env var to enable metrics collection
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"

    def teardown_method(self):
        """Reset environment variables."""
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)

    def test_compilation_caching(self):
        """Test that NEFF compilation is cached properly."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Clear caches using proper neuron bindings
        torch_neuronx._C._clear_compilation_cache()

        def multiply_fn(x, y):
            return x * y

        kernel = TorchNeuronXLAKernel(multiply_fn, "multiply_op")

        # First execution - should compile
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to("neuron")
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32).to("neuron")
        output1 = torch.empty_like(a)

        # Get initial cache stats
        initial_stats = torch_neuronx._C._get_compilation_cache_stats()
        result1 = kernel(a, b, output=output1)
        torch.neuron.synchronize()

        # Cache should have one entry
        stats_after_first = torch_neuronx._C._get_compilation_cache_stats()
        assert stats_after_first["total_entries"] == initial_stats["total_entries"] + 1

        # Second execution with same shapes - should use cache
        c = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32).to("neuron")
        d = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float32).to("neuron")
        output2 = torch.empty_like(c)

        result2 = kernel(c, d, output=output2)
        torch.neuron.synchronize()

        # Cache size should not increase (cache hit)
        stats_after_second = torch_neuronx._C._get_compilation_cache_stats()
        assert stats_after_second["total_entries"] == stats_after_first["total_entries"]
        assert stats_after_second["cache_hits"] > initial_stats["cache_hits"]

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
        torch.testing.assert_close(result2.cpu(), torch.tensor([[15.0, 18.0], [21.0, 24.0]]))

    def test_different_shapes_compile_separately(self):
        """Test that different shapes result in separate compilations."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

        # Clear caches using proper neuron bindings
        torch_neuronx._C._clear_compilation_cache()

        def add_fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(add_fn, "add_op")

        # First shape
        a1 = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b1 = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        output1 = torch.empty_like(a1)

        # Get initial cache stats
        initial_stats = torch_neuronx._C._get_compilation_cache_stats()
        result1 = kernel(a1, b1, output=output1)
        torch.neuron.synchronize()

        # Should have one cache entry
        stats_after_first = torch_neuronx._C._get_compilation_cache_stats()
        assert stats_after_first["total_entries"] == initial_stats["total_entries"] + 1

        # Different shape
        a2 = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        b2 = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        output2 = torch.empty_like(a2)

        result2 = kernel(a2, b2, output=output2)
        torch.neuron.synchronize()

        # Should have two cache entries now (different shapes compile separately)
        stats_after_second = torch_neuronx._C._get_compilation_cache_stats()
        assert stats_after_second["total_entries"] == initial_stats["total_entries"] + 2

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 2)
        torch.testing.assert_close(result2.cpu(), torch.ones(3, 3) * 2)
