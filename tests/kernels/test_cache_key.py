"""Comprehensive tests for cache key generation in XLA kernels."""

import os

import jax.numpy as jnp
import pytest
import torch

import torch_neuronx
from torch_neuronx.kernels import TorchNeuronXLAKernel


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0",
    reason="Legacy cache logic requires legacy execution mode",
)
class TestLegacyCacheKeyGeneration:
    """Test cache key generation for XLA kernels."""

    def test_scalar_values_share_cache_key(self):
        """Test that different runtime scalar values produce the same cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, scalar):
            return x * scalar

        kernel = TorchNeuronXLAKernel(fn, "test_scalar_op")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Get cache keys for different scalar values
        key1 = kernel.get_cache_key(a, 2.0)
        key2 = kernel.get_cache_key(a, 3.0)

        # Should be the same - runtime scalar values don't affect compilation
        assert key1 == key2, "Runtime scalar values should not affect cache key"

        # Execute with different scalars - should reuse same NEFF
        result1 = kernel(a, 2.0)
        result2 = kernel(a, 3.0)

        # Only one NEFF should be compiled
        assert len(kernel._neff_cache) == 1

        # Verify results are correct
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 2.0)
        torch.testing.assert_close(result2.cpu(), torch.ones(2, 2) * 3.0)

    def test_scalar_types_affect_cache_key(self):
        """Test that different scalar types produce different cache keys."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, scalar):
            return x + scalar

        kernel = TorchNeuronXLAKernel(fn, "test_scalar_type")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Different scalar types produce different HLO and thus different cache keys
        result1 = kernel(a, 2.0)
        result2 = kernel(a, 2)  # int scalar - different HLO

        # Should compile twice - once for float scalar, once for int scalar
        assert len(kernel._neff_cache) == 2, "Different scalar types should create different NEFFs"

        # Results should be the same due to JAX's type promotion at execution
        torch.testing.assert_close(result1.cpu(), result2.cpu())

    def test_multiple_runtime_scalars(self):
        """Test cache key with multiple runtime scalar parameters."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, alpha, beta):
            return x * alpha + beta

        kernel = TorchNeuronXLAKernel(fn, "test_multi_scalar")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different scalar values
        result1 = kernel(a, 2.0, 3.0)  # 1 * 2 + 3 = 5
        result2 = kernel(a, 4.0, 5.0)  # 1 * 4 + 5 = 9

        # Only one compilation - runtime scalar values don't affect cache
        assert len(kernel._neff_cache) == 1

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 5.0)
        torch.testing.assert_close(result2.cpu(), torch.ones(2, 2) * 9.0)

    def test_static_arguments_affect_cache_key(self):
        """Test that static arguments are included in cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        # Create kernel with static_argnums for dim and keepdim
        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Get cache keys for different static argument values
        key1 = kernel.get_cache_key(a, 1, False)
        key2 = kernel.get_cache_key(a, (1, 2), False)
        key3 = kernel.get_cache_key(a, 1, True)

        # All should be different since static arguments differ
        assert key1 != key2, "Different static dim values should create different cache keys"
        assert key1 != key3, "Different static keepdim values should create different cache keys"
        assert key2 != key3, "All static argument combinations should be unique"

    def test_static_arguments_in_cache_key_string(self):
        """Test that static argument values appear correctly in cache key string."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        key1 = kernel.get_cache_key(a, 1, False)
        key2 = kernel.get_cache_key(a, (1, 2), False)
        key3 = kernel.get_cache_key(a, 1, True)

        # Verify static arguments appear in the cache key
        assert "static1_1" in key1, f"dim=1 should appear as static in key: {key1}"
        assert "static1_(1, 2)" in key2, f"dim=(1,2) should appear as static in key: {key2}"
        assert "static2_False" in key1, f"keepdim=False should appear as static in key: {key1}"
        assert "static2_True" in key3, f"keepdim=True should appear as static in key: {key3}"

    def test_static_arguments_create_separate_neffs(self):
        """Test that different static arguments compile separate NEFFs."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Execute operations with different static arguments
        result1 = kernel(a, 1, False)  # Shape: (4, 6)
        result2 = kernel(a, (1, 2), False)  # Shape: (4,)
        result3 = kernel(a, 1, True)  # Shape: (4, 1, 6)

        # Should have 3 different compilations
        assert (
            len(kernel._neff_cache) == 3
        ), f"Expected 3 NEFFs for different static args, got {len(kernel._neff_cache)}"

        # Verify output shapes are correct
        assert result1.shape == (4, 6)
        assert result2.shape == (4,)
        assert result3.shape == (4, 1, 6)

    def test_non_static_arguments_type_only(self):
        """Test that non-static arguments only include type in cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, scale):
            return x * scale

        # No static_argnums specified
        kernel = TorchNeuronXLAKernel(fn, "test_no_static")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        key = kernel.get_cache_key(a, 2.0)

        # Non-static scalars should show type only, not value
        assert "static1" not in key, f"Non-static args shouldn't be marked static: {key}"
        assert "s1_" in key, f"Non-static scalar should show type indicator: {key}"

    def test_tensor_shapes_affect_cache_key(self):
        """Test that tensor shapes are included in cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_shapes")

        a = torch.ones(2, 3, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 2, dtype=torch.float32).to("neuron")

        key1 = kernel.get_cache_key(a)
        key2 = kernel.get_cache_key(b)

        # Different shapes should produce different cache keys
        assert key1 != key2, "Different tensor shapes should create different cache keys"
        assert "(2, 3)" in key1, f"Shape (2, 3) should appear in key: {key1}"
        assert "(3, 2)" in key2, f"Shape (3, 2) should appear in key: {key2}"

    def test_tensor_dtypes_affect_cache_key(self):
        """Test that tensor dtypes are included in cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_dtypes")

        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(2, 2, dtype=torch.float16).to("neuron")

        key1 = kernel.get_cache_key(a)
        key2 = kernel.get_cache_key(b)

        # Different dtypes should produce different cache keys
        assert key1 != key2, "Different tensor dtypes should create different cache keys"
        assert "torch.float32" in key1, f"dtype float32 should appear in key: {key1}"
        assert "torch.float16" in key2, f"dtype float16 should appear in key: {key2}"

    def test_mixed_static_and_runtime_arguments(self):
        """Test cache key with mix of static and runtime arguments."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, dim, scale):
            # dim is static, scale is runtime
            return jnp.sum(x * scale, axis=dim)

        # Only dim (arg 1) is static
        kernel = TorchNeuronXLAKernel(fn, "test_mixed", static_argnums=(1,))
        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")

        # Same dim, different scale values
        key1 = kernel.get_cache_key(a, 0, 2.0)
        key2 = kernel.get_cache_key(a, 0, 3.0)

        # Same cache key (scale is runtime)
        assert key1 == key2, "Runtime scale values should not affect cache key"

        # Different dim, same scale
        key3 = kernel.get_cache_key(a, 1, 2.0)

        # Different cache key (dim is static)
        assert key1 != key3, "Static dim values should affect cache key"

        # Verify dim appears as static, scale doesn't
        assert "static1_0" in key1, f"dim=0 should appear as static: {key1}"
        assert "static1_1" in key3, f"dim=1 should appear as static: {key3}"
        assert "static2" not in key1, f"scale should not be static: {key1}"

    def test_different_op_names(self):
        """Test cache keys with differing op names."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x):
            return x * 2

        kernel_a = TorchNeuronXLAKernel(fn, "test_a")
        kernel_b = TorchNeuronXLAKernel(fn, "test_b")

        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")

        # Same fn, same inputs, different op name.
        key1 = kernel_a.get_cache_key(a)
        key2 = kernel_b.get_cache_key(a)

        assert key1 != key2, "Different op names should result in different cache keys"

    def test_static_indices_negative(self):
        """ "Test cache_key with negative static_argnames."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(-1, -2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        key1 = kernel.get_cache_key(a, 1, False)
        key2 = kernel.get_cache_key(a, (1, 2), False)
        key3 = kernel.get_cache_key(a, 1, True)

        # Verify negative static arguments appear in the cache key
        assert "static1_1" in key1, f"dim=1 should appear as static in key: {key1}"
        assert "static1_(1, 2)" in key2, f"dim=(1,2) should appear as static in key: {key2}"
        assert "static2_False" in key1, f"keepdim=False should appear as static in key: {key1}"
        assert "static2_True" in key3, f"keepdim=True should appear as static in key: {key3}"

    def test_donate_argnums_affects_cache_key(self):
        """Test that donate_argnums is included in cache key."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(fn, "test_buffer_donation")
        a = torch.ones(3, 3, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 3, dtype=torch.float32).to("neuron")

        # Get cache keys with different donate_argnums
        key_no_donate = kernel.get_cache_key(a, b)
        key_donate_0 = kernel.get_cache_key(a, b, donate_argnums=(0,))
        key_donate_01 = kernel.get_cache_key(a, b, donate_argnums=(0, 1))
        key_donate_1 = kernel.get_cache_key(a, b, donate_argnums=(1,))

        # All should be different
        assert key_no_donate != key_donate_0, "No donation vs donation should differ"
        assert key_donate_0 != key_donate_01, "Different donate_argnums should differ"
        assert key_donate_0 != key_donate_1, "Different donate indices should differ"
        assert key_donate_01 != key_donate_1, "Different donate combinations should differ"

        # Verify donate_argnums suffix appears in cache key string
        # Check that the donation suffix is at the end of the key
        assert not key_no_donate.endswith(
            "_donate_0"
        ), f"No donation key should not end with donation suffix: {key_no_donate}"
        assert key_donate_0.endswith(
            "_donate_0"
        ), f"donate_argnums=(0,) should end key: {key_donate_0}"
        assert key_donate_01.endswith(
            "_donate_0,1"
        ), f"donate_argnums=(0,1) should end key: {key_donate_01}"
        assert key_donate_1.endswith(
            "_donate_1"
        ), f"donate_argnums=(1,) should end key: {key_donate_1}"

    def test_donate_argnums_creates_separate_neffs(self):
        """Test that different donate_argnums compile separate NEFFs."""
        TorchNeuronXLAKernel.clear_all_caches()

        def fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(fn, "test_donate_neff")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different donate_argnums
        output1 = torch.empty_like(a)
        result1 = kernel(a.clone(), b.clone(), output=output1)

        output2 = a.clone()
        result2 = kernel(output2, b.clone(), output=output2, donate_argnums=(0,))

        output3 = torch.empty_like(a)
        result3 = kernel(a.clone(), b.clone(), output=output3, donate_argnums=(0, 1))

        # Should have 3 different compilations
        assert (
            len(kernel._neff_cache) == 3
        ), f"Expected 3 NEFFs for different donate_argnums, got {len(kernel._neff_cache)}"

        # Verify results are all correct
        expected = torch.ones(2, 2, dtype=torch.float32) * 2
        torch.testing.assert_close(result1.cpu(), expected)
        torch.testing.assert_close(result2.cpu(), expected)
        torch.testing.assert_close(result3.cpu(), expected)


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "1",
    reason="Cache logic requires async execution mode",
)
class TestCacheKeyGeneration:
    """Test cache key generation for XLA kernels."""

    def setup_method(self):
        """Set env var to collect compilation stats/metrics"""
        # Set env var to enable metrics collection
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"

    def teardown_method(self):
        """Reset environment variables."""
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)

    def test_scalar_values_share_cache_key(self):
        """Test that different runtime scalar values produce the same cache key."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, scalar):
            return x * scalar

        kernel = TorchNeuronXLAKernel(fn, "test_scalar_op")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different scalars - should reuse same NEFF
        result1 = kernel(a, 2.0)
        result2 = kernel(a, 3.0)
        torch.neuron.synchronize()

        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 1
        ), f"Expected 1 cache entry for the same static args, got {cache_stats['total_entries']}"

        assert cache_stats["cache_hits"] == 1, "Expected 1 cache hit for the second kernel"

        # Get actual cache keys - should only have 1 entry since keys are the same
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 1, f"Expected 1 cache entry, got {len(all_cached_keys)}"

        # Verify the cache key contains the operation name
        cache_key = next(iter(all_cached_keys))
        assert (
            "test_scalar_op" in cache_key
        ), f"Cache key should contain operation name: {cache_key}"

        # Verify results are correct
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 2.0)
        torch.testing.assert_close(result2.cpu(), torch.ones(2, 2) * 3.0)

    def test_scalar_types_affect_cache_key(self):
        """Test that different scalar types produce different cache keys."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, scalar):
            return x + scalar

        kernel = TorchNeuronXLAKernel(fn, "test_scalar_type")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different scalar types
        result1 = kernel(a, 2.0)
        result2 = kernel(a, 2)  # int scalar - different HLO
        torch.neuron.synchronize()

        # Get actual cache keys - should have 2 entries for different scalar types
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 2
        ), f"Expected 2 cache entries for different scalar types, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_scalar_type" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 2 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 2, (
            f"Expected 2 cache entries for different scalar types, "
            f"got {cache_stats['total_entries']}"
        )

        # Results should be the same due to JAX's type promotion at execution
        torch.testing.assert_close(result1.cpu(), result2.cpu())

    def test_multiple_runtime_scalars(self):
        """Test cache key with multiple runtime scalar parameters."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, alpha, beta):
            return x * alpha + beta

        kernel = TorchNeuronXLAKernel(fn, "test_multi_scalar")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different scalar values
        result1 = kernel(a, 2.0, 3.0)  # 1 * 2 + 3 = 5
        result2 = kernel(a, 4.0, 5.0)  # 1 * 4 + 5 = 9
        torch.neuron.synchronize()

        # Only one compilation - runtime scalar values don't affect cache
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 1, f"Expected 1 cache entry, got {len(all_cached_keys)}"

        # Verify the cache key contains the operation name
        cache_key = next(iter(all_cached_keys))
        assert (
            "test_multi_scalar" in cache_key
        ), f"Cache key should contain operation name: {cache_key}"

        # Verify cache stats show 1 compilation and 1 hit
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 1
        ), f"Expected 1 cache entry for runtime scalars, got {cache_stats['total_entries']}"
        assert cache_stats["cache_hits"] == 1, "Expected 1 cache hit for the second execution"

        # Verify results
        torch.testing.assert_close(result1.cpu(), torch.ones(2, 2) * 5.0)
        torch.testing.assert_close(result2.cpu(), torch.ones(2, 2) * 9.0)

    def test_static_arguments_affect_cache_key(self):
        """Test that static arguments are included in cache key."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        # Create kernel with static_argnums for dim and keepdim
        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel(a, 1, False)
        kernel(a, (1, 2), False)
        kernel(a, 1, True)
        torch.neuron.synchronize()

        # Get actual cache keys - should have 3 entries for different static args
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 3, f"Expected 3 cache entries, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_static_op" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 3 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 3, (
            f"Expected 3 cache entries for different static args, "
            f"got {cache_stats['total_entries']}"
        )

    def test_static_arguments_in_cache_key_string(self):
        """Test that static argument values appear correctly in cache key string."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Execute operations and verify cache behavior
        kernel(a, 1, False)
        torch.neuron.synchronize()
        actual_keys_1 = torch_neuronx._C._get_all_cache_keys()

        kernel(a, (1, 2), False)
        torch.neuron.synchronize()
        actual_keys_2 = torch_neuronx._C._get_all_cache_keys()

        kernel(a, 1, True)
        torch.neuron.synchronize()
        actual_keys_3 = torch_neuronx._C._get_all_cache_keys()

        # Each execution should create a separate cache entry due to different static args
        assert len(actual_keys_1) >= 1, "First execution should create cache entry"
        assert len(actual_keys_2) > len(
            actual_keys_1
        ), "Second execution should create new cache entry"
        assert len(actual_keys_3) > len(
            actual_keys_2
        ), "Third execution should create new cache entry"

        # Verify we have 3 different cache entries
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] >= 3, (
            f"Expected at least 3 cache entries for different static args, "
            f"got {cache_stats['total_entries']}"
        )

        # Verify all cache keys contain the operation name
        final_keys = torch_neuronx._C._get_all_cache_keys()
        assert all(
            "test_static_op" in key for key in final_keys
        ), f"All cache keys should contain operation name: {final_keys}"

    def test_static_arguments_create_separate_neffs(self):
        """Test that different static arguments compile separate NEFFs."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(1, 2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Execute operations with different static arguments
        result1 = kernel(a, 1, False)  # Shape: (4, 6)
        result2 = kernel(a, (1, 2), False)  # Shape: (4,)
        result3 = kernel(a, 1, True)  # Shape: (4, 1, 6)

        torch.neuron.synchronize()  # Ensure all operations complete

        # Should have 3 different compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        cache_count = cache_stats["total_entries"]
        assert (
            cache_count >= 3
        ), f"Expected at least 3 NEFFs for different static args, got {cache_count}"

        # Verify output shapes are correct
        assert result1.shape == (4, 6)
        assert result2.shape == (4,)
        assert result3.shape == (4, 1, 6)

    def test_non_static_arguments_type_only(self):
        """Test that non-static arguments only include type in cache key."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, scale):
            return x * scale

        # No static_argnums specified
        kernel = TorchNeuronXLAKernel(fn, "test_no_static")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with different scalar values - should reuse same NEFF
        kernel(a, 2.0)
        kernel(a, 3.0)
        torch.neuron.synchronize()

        # Should only have one cache entry since scalars are runtime values
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 1
        ), f"Expected 1 cache entry for runtime scalars, got {len(all_cached_keys)}"

        # Verify the cache key contains the operation name
        cache_key = next(iter(all_cached_keys))
        assert (
            "test_no_static" in cache_key
        ), f"Cache key should contain operation name: {cache_key}"

        # Verify cache stats show 1 compilation and 1 hit
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 1
        ), f"Expected 1 cache entry for runtime scalars, got {cache_stats['total_entries']}"
        assert cache_stats["cache_hits"] == 1, "Expected 1 cache hit for the second execution"

    def test_tensor_shapes_affect_cache_key(self):
        """Test that tensor shapes are included in cache key."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_shapes")

        a = torch.ones(2, 3, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 2, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel(a)
        kernel(b)
        torch.neuron.synchronize()

        # Different shapes should create separate cache entries
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 2
        ), f"Expected 2 cache entries for different shapes, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_shapes" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 2 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 2
        ), f"Expected 2 cache entries for different shapes, got {cache_stats['total_entries']}"

    def test_tensor_dtypes_affect_cache_key(self):
        """Test that tensor dtypes are included in cache key."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_dtypes")

        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(2, 2, dtype=torch.float16).to("neuron")

        # Execute operations to populate cache
        kernel(a)
        kernel(b)
        torch.neuron.synchronize()

        # Different dtypes should create separate cache entries
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 2
        ), f"Expected 2 cache entries for different dtypes, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_dtypes" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 2 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 2
        ), f"Expected 2 cache entries for different dtypes, got {cache_stats['total_entries']}"

    def test_mixed_static_and_runtime_arguments(self):
        """Test cache key with mix of static and runtime arguments."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, dim, scale):
            # dim is static, scale is runtime
            return jnp.sum(x * scale, axis=dim)

        # Only dim (arg 1) is static
        kernel = TorchNeuronXLAKernel(fn, "test_mixed", static_argnums=(1,))
        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel(a, 0, 2.0)
        kernel(a, 0, 3.0)  # Same dim, different scale - should reuse cache
        kernel(a, 1, 2.0)  # Different dim - should create new cache entry
        torch.neuron.synchronize()

        # Should have 2 cache entries (one for each unique static dim value)
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 2
        ), f"Expected 2 cache entries for different static args, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_mixed" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 2 compilations and 1 hit
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 2, (
            f"Expected 2 cache entries for different static args, "
            f"got {cache_stats['total_entries']}"
        )
        assert (
            cache_stats["cache_hits"] == 1
        ), "Expected 1 cache hit for the second execution with same dim"

    def test_different_op_names(self):
        """Test cache keys with differing op names."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x * 2

        kernel_a = TorchNeuronXLAKernel(fn, "test_a")
        kernel_b = TorchNeuronXLAKernel(fn, "test_b")

        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel_a(a)
        kernel_b(a)
        torch.neuron.synchronize()

        # Should have 2 cache entries for different op names
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 2
        ), f"Expected 2 cache entries for different op names, got {len(all_cached_keys)}"

        # Verify each cache key contains its respective operation name
        cache_keys_list = list(all_cached_keys)
        assert any(
            "test_a" in key for key in cache_keys_list
        ), f"Should have cache key containing 'test_a': {cache_keys_list}"
        assert any(
            "test_b" in key for key in cache_keys_list
        ), f"Should have cache key containing 'test_b': {cache_keys_list}"

        # Verify cache stats show 2 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 2
        ), f"Expected 2 cache entries for different op names, got {cache_stats['total_entries']}"

    def test_cache_statistics(self):
        """Test that cache statistics can be retrieved."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_stats")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        torch.neuron.synchronize()

        # Get initial stats
        initial_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert isinstance(initial_stats, dict), "Cache stats should be a dictionary"
        assert "total_entries" in initial_stats, "Stats should include total_entries"
        assert "cache_hits" in initial_stats, "Stats should include cache_hits"
        assert "cache_misses" in initial_stats, "Stats should include cache_misses"

        # Execute operation
        kernel(a)

        torch.neuron.synchronize()

        # Get stats after execution - this should not block
        final_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            final_stats["total_entries"] >= initial_stats["total_entries"]
        ), "Cache entries should not decrease"

    def test_has_cached_neff_functionality(self):
        """Test cache functionality by checking cache stats before and after execution."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x + 1

        kernel = TorchNeuronXLAKernel(fn, "test_has_cached")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        torch.neuron.synchronize()

        # Before execution, should have no cache entries
        stats_before = torch_neuronx._C._get_compilation_cache_stats()

        # Execute operation
        kernel(a)
        torch.neuron.synchronize()

        # After execution, should have cache entries
        stats_after = torch_neuronx._C._get_compilation_cache_stats()

        # Check that cache entries increased
        assert (
            stats_after["total_entries"] >= stats_before["total_entries"]
        ), "Cache entries should not decrease after execution"

    def test_cache_key_consistency(self):
        """Test that cache keys are consistent across multiple calls."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(fn, "test_consistency")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute multiple times with same inputs
        kernel(a, b)
        kernel(a, b)
        kernel(a, b)
        torch.neuron.synchronize()

        # Should only have one cache entry since inputs are identical
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert (
            len(all_cached_keys) == 1
        ), f"Expected 1 cache entry for identical calls, got {len(all_cached_keys)}"

        # Verify the cache key contains the operation name
        cache_key = next(iter(all_cached_keys))
        assert (
            "test_consistency" in cache_key
        ), f"Cache key should contain operation name: {cache_key}"

        # Verify cache stats show 1 compilation and 2 hits
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert (
            cache_stats["total_entries"] == 1
        ), f"Expected 1 cache entry for identical calls, got {cache_stats['total_entries']}"
        assert cache_stats["cache_hits"] == 2, "Expected 2 cache hits for repeated executions"

    def test_cache_key_with_static_and_tensor_args(self):
        """Test cache key generation with mixed tensor and static arguments."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, axis, keepdims):
            return jnp.sum(x, axis=axis, keepdims=keepdims)

        kernel = TorchNeuronXLAKernel(fn, "test_mixed_args", static_argnums=(1, 2))
        a = torch.ones(3, 4, 5, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel(a, 0, True)
        kernel(a, 1, True)
        kernel(a, 0, False)
        torch.neuron.synchronize()

        # Should have 3 cache entries for different static argument combinations
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 3, (
            f"Expected 3 cache entries for different static combinations, "
            f"got {len(all_cached_keys)}"
        )

        # Verify all cache keys contain the operation name
        assert all(
            "test_mixed_args" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 3 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 3, (
            f"Expected 3 cache entries for different static combinations, "
            f"got {cache_stats['total_entries']}"
        )

    def test_expected_cache_keys_are_present(self):
        """Test that cache keys are actually present in the compilation cache."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(fn, "test_key_presence")
        a = torch.ones(3, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(3, 4, dtype=torch.float32).to("neuron")

        # Execute operation
        kernel(a, b)
        torch.neuron.synchronize()

        # Get all actual cache keys
        actual_keys = torch_neuronx._C._get_all_cache_keys()

        # Should have at least one cache entry
        assert len(actual_keys) >= 1, f"Expected at least 1 cache entry, got {len(actual_keys)}"

        # Verify the cache key contains the operation name
        cache_key = next(iter(actual_keys))
        assert (
            "test_key_presence" in cache_key
        ), f"Cache key should contain operation name: {cache_key}"

        # Verify the cache contains entries
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] >= 1, "Should have at least 1 cache entry"

    def test_cache_keys_are_deterministic(self):
        """Test that cache keys are deterministic across multiple runs."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x):
            return x * 2

        kernel = TorchNeuronXLAKernel(fn, "test_deterministic")
        a = torch.ones(2, 3, dtype=torch.float32).to("neuron")

        # Execute multiple times
        kernel(a)
        torch.neuron.synchronize()
        keys_after_first = torch_neuronx._C._get_all_cache_keys()

        kernel(a)
        torch.neuron.synchronize()
        keys_after_second = torch_neuronx._C._get_all_cache_keys()

        kernel(a)
        torch.neuron.synchronize()
        keys_after_third = torch_neuronx._C._get_all_cache_keys()

        # Should have same cache keys (deterministic)
        assert (
            keys_after_first == keys_after_second == keys_after_third
        ), "Cache keys should be deterministic across multiple executions"

        # Should only have one cache entry
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 1, (
            f"Expected exactly 1 cache entry for identical operations, "
            f"got {cache_stats['total_entries']}"
        )

    def test_static_indices_negative(self):
        """ "Test cache_key with negative static_argnames."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, dim, keepdim):
            return jnp.sum(x, axis=dim, keepdims=keepdim)

        # Create kernel with static_argnums for dim and keepdim
        kernel = TorchNeuronXLAKernel(fn, "test_static_op", static_argnums=(-1, -2))
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron")

        # Execute operations to populate cache
        kernel(a, 1, False)
        kernel(a, (1, 2), False)
        kernel(a, 1, True)
        torch.neuron.synchronize()

        # Get actual cache keys - should have 3 entries for different static args
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 3, f"Expected 3 cache entries, got {len(all_cached_keys)}"

        # Verify all cache keys contain the operation name
        assert all(
            "test_static_op" in key for key in all_cached_keys
        ), f"All cache keys should contain operation name: {all_cached_keys}"

        # Verify cache stats show 3 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 3, (
            f"Expected 3 cache entries for different static args, "
            f"got {cache_stats['total_entries']}"
        )

    def test_donate_argnums_creates_separate_cache_entries(self):
        """Test that different donate_argnums create separate cache entries."""
        torch_neuronx._C._clear_compilation_cache()

        def fn(x, y):
            return x + y

        kernel = TorchNeuronXLAKernel(fn, "test_donate_async")
        a = torch.ones(2, 2, dtype=torch.float32).to("neuron")
        b = torch.ones(2, 2, dtype=torch.float32).to("neuron")

        # Execute with no donation
        output1 = torch.empty_like(a)
        result1 = kernel(a.clone(), b.clone(), output=output1)
        torch.neuron.synchronize()

        # Execute with donate_argnums=(0,)
        output2 = a.clone()
        result2 = kernel(output2, b.clone(), output=output2, donate_argnums=(0,))
        torch.neuron.synchronize()

        # Execute with donate_argnums=(0, 1)
        output3 = torch.empty_like(a)
        result3 = kernel(a.clone(), b.clone(), output=output3, donate_argnums=(0, 1))
        torch.neuron.synchronize()

        # Should have 3 different cache entries
        all_cached_keys = torch_neuronx._C._get_all_cache_keys()
        assert len(all_cached_keys) == 3, (
            f"Expected 3 cache entries for different donate_argnums, " f"got {len(all_cached_keys)}"
        )

        # Verify cache stats show 3 compilations
        cache_stats = torch_neuronx._C._get_compilation_cache_stats()
        assert cache_stats["total_entries"] == 3, (
            f"Expected 3 cache entries for different donate_argnums, "
            f"got {cache_stats['total_entries']}"
        )

        # Verify results are all correct
        expected = torch.ones(2, 2, dtype=torch.float32) * 2
        torch.testing.assert_close(result1.cpu(), expected)
        torch.testing.assert_close(result2.cpu(), expected)
        torch.testing.assert_close(result3.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
