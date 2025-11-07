"""Test XLA kernel functionality."""

import jax
import jax.numpy as jnp
import pytest
import torch

import torch_neuron
from torch_neuron.utils import is_sync_mode_enabled


class TestXLAKernel:
    """Test XLA kernel compilation and execution."""

    def test_simple_add_kernel(self):
        """Test a simple element-wise addition kernel."""

        from torch_neuron.kernels import TorchNeuronXLAKernel

        # Define a simple JAX function
        def add_fn(x, y):
            return x + y

        # Create kernel
        kernel = TorchNeuronXLAKernel(add_fn, "add_op")
        assert kernel.op_name == "add_op", "The op_name should match the specified one"

        # Create test tensors
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        output = torch.empty(4, 4, dtype=torch.float32).to("neuron")

        # Execute kernel
        result = kernel(a, b, output=output)

        # Verify result
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_cache_reuse(self):
        """Test that compiled NEFFs are cached properly."""

        from torch_neuron.kernels import TorchNeuronXLAKernel

        # Clear caches
        TorchNeuronXLAKernel.clear_all_caches()

        def multiply_fn(x, y):
            return x * y

        kernel = TorchNeuronXLAKernel(multiply_fn, "multiply_op")

        # First execution - should compile
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to("neuron")
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32).to("neuron")
        output1 = torch.empty_like(a)

        if is_sync_mode_enabled():
            initial_stats = len(kernel._neff_cache)
        else:
            initial_stats = torch_neuron._C._get_compilation_cache_stats()
        result1 = kernel(a, b, output=output1)
        torch.neuron.synchronize()

        # Cache should have one entry
        if is_sync_mode_enabled():
            stats_after_first = len(kernel._neff_cache)
            assert stats_after_first == initial_stats + 1
        else:
            stats_after_first = torch_neuron._C._get_compilation_cache_stats()
            assert stats_after_first["total_entries"] == initial_stats["total_entries"] + 1

        # Second execution with same shapes - should use cache
        c = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32).to("neuron")
        d = torch.tensor([[3.0, 3.0], [3.0, 3.0]], dtype=torch.float32).to("neuron")
        output2 = torch.empty_like(c)

        result2 = kernel(c, d, output=output2)
        torch.neuron.synchronize()

        # Cache size should not increase
        if is_sync_mode_enabled():
            stats_after_second = len(kernel._neff_cache)
            assert stats_after_second == stats_after_first
        else:
            stats_after_second = torch_neuron._C._get_compilation_cache_stats()
            assert stats_after_second["total_entries"] == stats_after_first["total_entries"]
            assert stats_after_second["cache_hits"] > initial_stats["cache_hits"]

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
        torch.testing.assert_close(result2.cpu(), torch.tensor([[15.0, 18.0], [21.0, 24.0]]))

    def test_shape_inference_simple(self):
        """Test automatic output shape inference for simple operations."""
        from torch_neuron.kernels import TorchNeuronXLAKernel

        # Test element-wise operation (same shape)
        def add_fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(add_fn, "add_op")

        a = torch.ones(3, 5, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 5, dtype=torch.float32).to("neuron")

        # Call without output - should infer shape
        result = kernel(a, b)

        assert result.shape == (3, 5)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), torch.ones(3, 5) * 2)

    def test_shape_inference_reduction(self):
        """Test shape inference for reduction operations."""
        from torch_neuron.kernels import TorchNeuronXLAKernel

        # Test reduction operation
        def sum_axis1(x):
            return jnp.sum(x, axis=1)

        kernel = TorchNeuronXLAKernel(sum_axis1, "sum_axis1_op")

        a = torch.ones(4, 6, dtype=torch.float32).to("neuron")

        # Call without output - should infer reduced shape
        result = kernel(a)

        assert result.shape == (4,)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), torch.ones(4) * 6)

    def test_shape_inference_matmul(self):
        """Test shape inference for matrix multiplication."""
        from torch_neuron.kernels import TorchNeuronXLAKernel

        # Test matmul with different shapes
        def matmul_fn(x, y):
            return jnp.matmul(x, y)

        kernel = TorchNeuronXLAKernel(matmul_fn, "matmul_op")

        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 5, dtype=torch.float32).to("neuron")

        # Call without output - should infer matmul output shape
        result = kernel(a, b)

        assert result.shape == (3, 5)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), torch.ones(3, 5) * 4)

    def test_shape_inference_dtype_preservation(self):
        """Test that dtypes are correctly preserved through shape inference."""
        from torch_neuron.kernels import TorchNeuronXLAKernel

        def identity_fn(x):
            return x

        kernel = TorchNeuronXLAKernel(identity_fn, "identity_op")

        # Test different dtypes
        for torch_dtype, expected_val in [
            (torch.float32, 1.0),
            (torch.int32, 1),
            (torch.float16, 1.0),
        ]:
            a = torch.ones(2, 3, dtype=torch_dtype).to("neuron") * expected_val
            result = kernel(a)

            assert result.shape == (2, 3)
            assert result.dtype == torch_dtype
            torch.testing.assert_close(
                result.cpu(), torch.ones(2, 3, dtype=torch_dtype) * expected_val
            )
