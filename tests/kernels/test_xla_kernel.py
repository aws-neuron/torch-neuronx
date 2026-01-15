"""Test XLA kernel functionality."""

import os

import jax
import jax.numpy as jnp
import pytest
import torch

import torch_neuronx
from torch_neuronx.utils import is_sync_mode_enabled


class TestXLAKernel:
    """Test XLA kernel compilation and execution."""

    def test_component_static_argnums_validation(self):
        """Test that accessing static_argnums before setting raises RuntimeError in components."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def test_fn(x):
            return x

        kernel = TorchNeuronXLAKernel(test_fn, "test_op")

        # Should raise RuntimeError when accessing component static_argnums before setting
        with pytest.raises(RuntimeError, match="static_argnums must be set before use"):
            _ = kernel.jax_compiler.static_argnums

    def test_negative_static_argnums_normalization(self):
        """Test that negative static_argnums are properly normalized for different input lengths."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def test_fn(*args):
            # This function expects the last argument to be static (compile-time constant)
            # If static_argnums is wrong, JAX compilation will fail or behave incorrectly
            *tensors, static_multiplier = args
            result = tensors[0]
            for tensor in tensors[1:]:
                result = result + tensor
            # Use static_multiplier in a way that requires it to be compile-time constant
            return jnp.repeat(result, static_multiplier, axis=0)

        # Create kernel with negative static argnum (-1 = last argument)
        kernel = TorchNeuronXLAKernel(test_fn, "test_op", static_argnums=(-1,))

        # Test with 3 inputs: 2 tensors + static_multiplier=2
        a = torch.ones(1, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(1, 2, dtype=torch.float32).to("neuron")
        result1 = kernel(a, b, 2)  # Should repeat (1+1) 2 times along axis 0

        # Test with 4 inputs: 3 tensors + static_multiplier=3
        c = torch.ones(1, 2, dtype=torch.float32).to("neuron")
        result2 = kernel(a, b, c, 3)  # Should repeat (1+1+1) 3 times along axis 0

        # Verify shapes - this would fail if static_argnums normalization is broken
        assert result1.shape == (2, 2), f"Expected shape (2, 2), got {result1.shape}"
        assert result2.shape == (3, 2), f"Expected shape (3, 2), got {result2.shape}"

        # Verify values
        assert torch.allclose(result1, torch.full((2, 2), 2.0).to("neuron"))
        assert torch.allclose(result2, torch.full((3, 2), 3.0).to("neuron"))

    def test_simple_add_kernel(self):
        """Test a simple element-wise addition kernel."""

        from torch_neuronx.kernels import TorchNeuronXLAKernel

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
        # set env var to collect metrics
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"

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

        if is_sync_mode_enabled():
            initial_stats = len(kernel._neff_cache)
        else:
            initial_stats = torch_neuronx._C._get_compilation_cache_stats()
        result1 = kernel(a, b, output=output1)
        torch.neuron.synchronize()

        # Cache should have one entry
        if is_sync_mode_enabled():
            stats_after_first = len(kernel._neff_cache)
            assert stats_after_first == initial_stats + 1
        else:
            stats_after_first = torch_neuronx._C._get_compilation_cache_stats()
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
            stats_after_second = torch_neuronx._C._get_compilation_cache_stats()
            assert stats_after_second["total_entries"] == stats_after_first["total_entries"]
            assert stats_after_second["cache_hits"] > initial_stats["cache_hits"]

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
        torch.testing.assert_close(result2.cpu(), torch.tensor([[15.0, 18.0], [21.0, 24.0]]))
        # reset environment variable
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)

    def test_shape_inference_simple(self):
        """Test automatic output shape inference for simple operations."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

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
        from torch_neuronx.kernels import TorchNeuronXLAKernel

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
        from torch_neuronx.kernels import TorchNeuronXLAKernel

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
        from torch_neuronx.kernels import TorchNeuronXLAKernel

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

    def test_donate_argnums_basic(self):
        """Test that donate_argnums enables buffer donation for in-place operations."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def inplace_add(x, y):
            """Add y to x, returning the result."""
            return x + y

        kernel = TorchNeuronXLAKernel(inplace_add, "inplace_add_op")

        # Create input tensors
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Get tensor data pointer before operation
        a_ptr = a.data_ptr()

        # Execute with donate_argnums=(0,) to donate first argument
        # This tells JAX that input 'a' can be reused for the output
        result = kernel(a, b, output=a, donate_argnums=(0,))

        # Verify result is correct
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(result.cpu(), expected)

        # Verify buffer was reused (same pointer)
        assert result.data_ptr() == a_ptr, "Buffer should be reused when donated"

    def test_donate_argnums_multiple_args(self):
        """Test donate_argnums with multiple donated arguments."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def multi_update(x, y, z):
            """Update x and y based on z."""
            return x + z, y * z

        kernel = TorchNeuronXLAKernel(multi_update, "multi_update_op")

        # Create input tensors
        a = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 3, dtype=torch.float32).to("neuron") * 2
        c = torch.ones(3, 3, dtype=torch.float32).to("neuron") * 3

        # Get tensor data pointers before operation
        a_ptr = a.data_ptr()
        b_ptr = b.data_ptr()

        # Execute with donate_argnums=(0, 1) to donate first two arguments
        result1, result2 = kernel(a, b, c, output=(a, b), donate_argnums=(0, 1))

        # Verify results are correct
        expected1 = torch.ones(3, 3, dtype=torch.float32) * 4  # 1 + 3
        expected2 = torch.ones(3, 3, dtype=torch.float32) * 6  # 2 * 3
        torch.testing.assert_close(result1.cpu(), expected1)
        torch.testing.assert_close(result2.cpu(), expected2)

        # Verify buffers were reused
        assert result1.data_ptr() == a_ptr, "First buffer should be reused"
        assert result2.data_ptr() == b_ptr, "Second buffer should be reused"

    def test_donate_argnums_correctness(self):
        """Test that donate_argnums doesn't affect correctness."""
        from torch_neuronx.kernels import TorchNeuronXLAKernel

        def complex_op(x, y):
            """More complex operation to verify correctness."""
            return x * 2 + y * 3

        kernel = TorchNeuronXLAKernel(complex_op, "complex_op")

        # Create test data
        a = torch.randn(5, 5, dtype=torch.float32).to("neuron")
        b = torch.randn(5, 5, dtype=torch.float32).to("neuron")

        # Execute without donation
        output1 = torch.empty_like(a)
        result1 = kernel(a.clone(), b.clone(), output=output1)

        # Execute with donation
        a_donated = a.clone()
        output2 = a_donated  # Reuse input buffer
        result2 = kernel(a_donated, b.clone(), output=output2, donate_argnums=(0,))

        # Results should be identical
        torch.testing.assert_close(result1.cpu(), result2.cpu())

        # Verify expected result
        expected = a.cpu() * 2 + b.cpu() * 3
        torch.testing.assert_close(result2.cpu(), expected)
