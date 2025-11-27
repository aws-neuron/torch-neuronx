"""Test cases for JAX kernel caching functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from torch_neuronx.python_ops.jax.context import ExecutionContext
from torch_neuronx.python_ops.jax.kernel import JaxKernel


class TestJaxKernelCaching:
    """Test JAX kernel caching functionality."""

    @pytest.fixture
    def mock_jax_fn(self):
        """Mock JAX function."""
        return Mock()

    @pytest.fixture
    def kernel(self, mock_jax_fn):
        """Create a JAX kernel for testing."""
        with (
            patch("torch_neuronx.python_ops.jax.kernel.JaxCompiler"),
            patch("torch_neuronx.python_ops.jax.kernel.HloCompiler"),
            patch("torch_neuronx.python_ops.jax.kernel.CompilationCache"),
            patch("torch_neuronx.python_ops.jax.kernel.ArgumentProcessor"),
            patch("torch_neuronx.python_ops.jax.kernel.OutputHandler"),
        ):
            kernel = JaxKernel(
                jax_fn=mock_jax_fn,
                op_name="test_op",
                static_argnums=(1,),
                static_argnames=("axis",),
            )
            return kernel

    def test_generate_unified_cache_key_basic(self, kernel):
        """Test basic cache key generation."""
        inputs = (torch.randn(2, 3), 42)
        kwargs = {"axis": 0}

        with (
            patch.object(kernel, "get_cache_key", return_value="base_key"),
            patch.object(
                kernel.arg_processor, "add_static_to_cache_key", return_value="base_key_with_kwargs"
            ),
        ):
            cache_key = kernel._generate_unified_cache_key(inputs, kwargs)

            assert cache_key == "base_key_with_kwargs"
            kernel.get_cache_key.assert_called_once_with(
                "test_op", *inputs, kwargs={}, static_indices=(1,)
            )
            kernel.arg_processor.add_static_to_cache_key.assert_called_once_with("base_key", kwargs)

    def test_generate_unified_cache_key_with_context(self, kernel):
        """Test cache key generation with execution context."""
        inputs = (torch.randn(2, 3),)
        kwargs = {}
        context = Mock(spec=ExecutionContext)
        context.has_original_inputs.return_value = True
        context.expected_dtypes = [torch.float32, torch.int64]

        with (
            patch.object(kernel, "get_cache_key", return_value="base_key"),
            patch.object(
                kernel.arg_processor, "add_static_to_cache_key", return_value="base_key_with_kwargs"
            ),
        ):
            cache_key = kernel._generate_unified_cache_key(inputs, kwargs, context)

            expected = (
                "base_key_with_kwargs_has_original_inputs_dtype_torch.float32_dtype_torch.int64"
            )
            assert cache_key == expected

    def test_generate_unified_cache_key_context_no_extras(self, kernel):
        """Test cache key generation with context but no extra info."""
        inputs = (torch.randn(2, 3),)
        kwargs = {}
        context = Mock(spec=ExecutionContext)
        context.has_original_inputs.return_value = False
        context.expected_dtypes = None

        with (
            patch.object(kernel, "get_cache_key", return_value="base_key"),
            patch.object(
                kernel.arg_processor, "add_static_to_cache_key", return_value="base_key_with_kwargs"
            ),
        ):
            cache_key = kernel._generate_unified_cache_key(inputs, kwargs, context)

            assert cache_key == "base_key_with_kwargs"

    def test_output_spec_caching(self, kernel):
        """Test that output specs are cached and reused."""
        inputs = (torch.randn(2, 3),)
        kwargs = {}
        context = None

        # Mock the output handler
        mock_output_specs = ([Mock()], True, [torch.float32], [False])
        kernel.output_handler.infer_output_specs.return_value = mock_output_specs

        with patch.object(kernel, "_generate_unified_cache_key", return_value="test_cache_key"):
            # First call should compute and cache
            result1 = kernel._get_cached_output_specs(inputs, kwargs, context)

            # Second call should use cache
            result2 = kernel._get_cached_output_specs(inputs, kwargs, context)

            # Should return same result
            assert result1 == result2 == mock_output_specs

            # Should only call infer_output_specs once
            assert kernel.output_handler.infer_output_specs.call_count == 1

            # Should have cached the result
            assert "test_cache_key" in kernel._output_spec_cache
            assert kernel._output_spec_cache["test_cache_key"] == mock_output_specs

    def test_output_spec_cache_different_keys(self, kernel):
        """Test that different cache keys don't interfere."""
        inputs1 = (torch.randn(2, 3),)
        inputs2 = (torch.randn(4, 5),)
        kwargs = {}
        context = None

        mock_output_specs1 = ([Mock()], True, [torch.float32], [False])
        mock_output_specs2 = ([Mock()], False, [torch.int64], [True])

        kernel.output_handler.infer_output_specs.side_effect = [
            mock_output_specs1,
            mock_output_specs2,
        ]

        with patch.object(kernel, "_generate_unified_cache_key", side_effect=["key1", "key2"]):
            result1 = kernel._get_cached_output_specs(inputs1, kwargs, context)
            result2 = kernel._get_cached_output_specs(inputs2, kwargs, context)

            assert result1 == mock_output_specs1
            assert result2 == mock_output_specs2
            assert kernel.output_handler.infer_output_specs.call_count == 2

            # Both should be cached separately
            assert kernel._output_spec_cache["key1"] == mock_output_specs1
            assert kernel._output_spec_cache["key2"] == mock_output_specs2

    def test_cache_key_consistency_neff_vs_output_spec(self, kernel):
        """Test that NEFF and output spec caches use consistent base keys."""
        inputs = (torch.randn(2, 3),)
        kwargs = {"axis": 0}

        with (
            patch.object(kernel, "get_cache_key", return_value="base_key"),
            patch.object(
                kernel.arg_processor, "add_static_to_cache_key", return_value="base_key_with_kwargs"
            ),
        ):
            # NEFF cache key (no context)
            neff_key = kernel._generate_unified_cache_key(inputs, kwargs)

            # Output spec cache key (no context)
            output_spec_key = kernel._generate_unified_cache_key(inputs, kwargs, None)

            # Should be identical when no context is provided
            assert neff_key == output_spec_key == "base_key_with_kwargs"
