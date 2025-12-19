# ruff: noqa: N806

"""
Unit tests for neuron_dynamo_backend decompositions module
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch._decomp import decomposition_table

from torch_neuronx.neuron_dynamo_backend import decompositions


class TestDecompositionTable:
    """Test decomposition table functionality"""

    def test_get_custom_decomposition_table(self):
        """Test that custom decomposition table contains expected entries"""
        custom_table = decompositions.get_custom_decomposition_table()

        # Check that it returns a dictionary
        assert isinstance(custom_table, dict)

        # Check that specific ops are in the table
        assert torch.ops.aten.as_strided.default in custom_table
        assert torch.ops.aten.scalar_tensor.default in custom_table

        # Verify the functions are callable
        assert callable(custom_table[torch.ops.aten.as_strided.default])
        assert callable(custom_table[torch.ops.aten.scalar_tensor.default])

    def test_get_decomposition_table_selective(self):
        """Test selective decomposition table includes only specified ops"""
        decomp_table = decompositions.get_decomposition_table(decompose_all=False)

        # Should contain our custom decompositions
        assert torch.ops.aten.as_strided.default in decomp_table

        # Should be smaller than the full table
        full_table = decomposition_table
        assert len(decomp_table) <= len(full_table)

    def test_get_decomposition_table_all(self):
        """Test that decompose_all=True includes standard decompositions"""
        decomp_table = decompositions.get_decomposition_table(decompose_all=True)

        # Should contain our custom decompositions
        assert torch.ops.aten.as_strided.default in decomp_table

        # Should be at least as large as custom table
        custom_table = decompositions.get_custom_decomposition_table()
        assert len(decomp_table) >= len(custom_table)


class TestCustomAsStridedDecomposition:
    """Test custom as_strided decomposition"""

    def test_as_strided_basic_functionality(self):
        """Test basic as_strided decomposition functionality"""
        # Create a simple test tensor
        input_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        # Test as_strided operation
        size = (2, 3)
        stride = (4, 1)  # Skip every 4 elements for first dim, contiguous for second
        storage_offset = 0

        result = decompositions.custom_as_strided_decomposition(
            input_tensor, size, stride, storage_offset
        )

        # Verify output shape
        assert result.shape == size

        # Compare with PyTorch's as_strided
        expected = torch.as_strided(input_tensor, size, stride, storage_offset)
        torch.testing.assert_close(result, expected)

    def test_as_strided_with_offset(self):
        """Test as_strided decomposition with storage offset"""
        input_tensor = torch.arange(12, dtype=torch.float32)

        size = (3, 2)
        stride = (2, 1)
        storage_offset = 1

        result = decompositions.custom_as_strided_decomposition(
            input_tensor, size, stride, storage_offset
        )

        expected = torch.as_strided(input_tensor, size, stride, storage_offset)
        torch.testing.assert_close(result, expected)

    def test_as_strided_multidimensional(self):
        """Test as_strided with complex multidimensional striding"""
        input_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        size = (2, 2, 2)
        stride = (12, 4, 2)  # Original strides: (12, 4, 1), skip every other element in last dim
        storage_offset = 0

        result = decompositions.custom_as_strided_decomposition(
            input_tensor, size, stride, storage_offset
        )

        expected = torch.as_strided(input_tensor, size, stride, storage_offset)
        torch.testing.assert_close(result, expected)


class TestFlexAttentionDecomposition:
    """Test flex attention decomposition"""

    def test_flex_attention_basic(self):
        """Test basic flex attention functionality"""
        B, H, L, S, E = 2, 4, 8, 8, 16

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        result = decompositions.flex_attention_decomposition(query, key, value)

        assert result.shape == (B, H, L, E)
        assert result.dtype == query.dtype

    def test_flex_attention_with_scale(self):
        """Test flex attention with custom scale"""
        B, H, L, S, E = 1, 2, 4, 4, 8

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)
        scale = 0.5

        result = decompositions.flex_attention_decomposition(query, key, value, scale=scale)

        assert result.shape == (B, H, L, E)

    def test_flex_attention_with_mask(self):
        """Test flex attention with block mask"""
        B, H, L, S, E = 1, 2, 4, 4, 8

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        # Create a causal mask
        block_mask = torch.tril(torch.ones(1, 1, L, S, dtype=torch.bool))

        result = decompositions.flex_attention_decomposition(
            query, key, value, block_mask=block_mask
        )

        assert result.shape == (B, H, L, E)

    def test_flex_attention_with_gqa(self):
        """Test flex attention with grouped query attention"""
        B, H_q, H_kv, L, S, E = 1, 8, 2, 4, 4, 16

        query = torch.randn(B, H_q, L, E)
        key = torch.randn(B, H_kv, S, E)
        value = torch.randn(B, H_kv, S, E)

        result = decompositions.flex_attention_decomposition(query, key, value, enable_gqa=True)

        assert result.shape == (B, H_q, L, E)

    def test_flex_attention_return_lse(self):
        """Test flex attention returning log-sum-exp"""
        B, H, L, S, E = 1, 2, 4, 4, 8

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        result, lse = decompositions.flex_attention_decomposition(
            query, key, value, return_lse=True
        )

        assert result.shape == (B, H, L, E)
        assert lse.shape == (B, H, L)

    def test_flex_attention_chunked_processing(self):
        """Test flex attention with different chunk sizes"""
        B, H, L, S, E = 1, 2, 10, 8, 16

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E)

        # Test with different chunk sizes
        result1 = decompositions.flex_attention_decomposition(query, key, value, chunk_size=4)
        result2 = decompositions.flex_attention_decomposition(query, key, value, chunk_size=8)

        # Results should be close (within numerical precision)
        torch.testing.assert_close(result1, result2, rtol=1e-4, atol=1e-5)


class TestIndexCopyDecomposition:
    """Exhaustive tests for index_copy decomposition"""

    # Parameterized tests for dtype/shape combinations
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
        ],
    )
    def test_index_copy_dtypes(self, dtype):
        """Test index_copy with various dtypes"""
        input_tensor = torch.zeros(4, 3, dtype=dtype)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3, dtype=dtype)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "shape,dim,idx_size",
        [
            ((5,), 0, 3),  # 1D
            ((4, 3), 0, 2),  # 2D dim0
            ((4, 3), 1, 2),  # 2D dim1
            ((2, 4, 3), 0, 1),  # 3D dim0
            ((2, 4, 3), 1, 2),  # 3D dim1
            ((2, 4, 3), 2, 2),  # 3D dim2
            ((2, 3, 4, 5), 0, 1),  # 4D dim0
            ((2, 3, 4, 5), 1, 2),  # 4D dim1
            ((2, 3, 4, 5), 2, 2),  # 4D dim2
            ((2, 3, 4, 5), 3, 3),  # 4D dim3
        ],
    )
    def test_index_copy_shapes(self, shape, dim, idx_size):
        """Test index_copy with various shape/dim combinations"""
        input_tensor = torch.zeros(shape)
        index = torch.randperm(shape[dim])[:idx_size]
        source_shape = list(shape)
        source_shape[dim] = idx_size
        source = torch.ones(source_shape)

        result = decompositions.index_copy_decomposition(input_tensor, dim, index, source)
        expected = input_tensor.index_copy(dim, index, source)

        torch.testing.assert_close(result, expected)

    # NaN/Inf edge cases
    def test_index_copy_with_nan(self):
        """Test index_copy with NaN values in source"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([0, 2])
        source = torch.tensor([[float("nan"), 1.0, 2.0], [3.0, float("nan"), 5.0]])

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        # Use equal for NaN comparison
        assert torch.equal(torch.isnan(result), torch.isnan(expected))
        # Compare non-NaN values
        mask = ~torch.isnan(expected)
        torch.testing.assert_close(result[mask], expected[mask])

    def test_index_copy_with_inf(self):
        """Test index_copy with Inf values in source"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([0, 2])
        source = torch.tensor([[float("inf"), 1.0, float("-inf")], [3.0, 4.0, 5.0]])

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_with_nan_in_input(self):
        """Test index_copy preserves NaN in non-indexed positions"""
        input_tensor = torch.tensor([[float("nan"), 1.0], [2.0, 3.0], [4.0, float("nan")]])
        index = torch.tensor([1])
        source = torch.tensor([[99.0, 99.0]])

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        assert torch.isnan(result[0, 0]) and torch.isnan(expected[0, 0])
        assert torch.isnan(result[2, 1]) and torch.isnan(expected[2, 1])
        torch.testing.assert_close(result[1], expected[1])

    # Device execution tests
    @pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
    def test_index_copy_neuron_device(self):
        """Test index_copy execution on Neuron device"""
        input_tensor = torch.zeros(4, 3).to("neuron")
        index = torch.tensor([0, 2]).to("neuron")
        source = torch.ones(2, 3).to("neuron")

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = torch.zeros(4, 3).index_copy(0, torch.tensor([0, 2]), torch.ones(2, 3))

        torch.testing.assert_close(result.to("cpu"), expected)

    @pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_index_copy_neuron_dtypes(self, dtype):
        """Test index_copy on Neuron with various dtypes"""
        input_tensor = torch.zeros(4, 3, dtype=dtype).to("neuron")
        index = torch.tensor([0, 2]).to("neuron")
        source = torch.ones(2, 3, dtype=dtype).to("neuron")

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = torch.zeros(4, 3, dtype=dtype).index_copy(
            0, torch.tensor([0, 2]), torch.ones(2, 3, dtype=dtype)
        )

        torch.testing.assert_close(result.to("cpu"), expected, rtol=1e-2, atol=1e-2)

    # Basic functionality tests
    def test_index_copy_1d(self):
        """Test index_copy decomposition on 1D tensor"""
        input_tensor = torch.zeros(5)
        index = torch.tensor([0, 2, 4])
        source = torch.tensor([1.0, 2.0, 3.0])

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_2d_dim0(self):
        """Test index_copy decomposition on 2D tensor along dim 0"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_2d_dim1(self):
        """Test index_copy decomposition on 2D tensor along dim 1"""
        input_tensor = torch.zeros(3, 5)
        index = torch.tensor([1, 3])
        source = torch.ones(3, 2)

        result = decompositions.index_copy_decomposition(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_3d(self):
        """Test index_copy decomposition on 3D tensor"""
        input_tensor = torch.zeros(2, 4, 3)
        index = torch.tensor([1, 3])
        source = torch.ones(2, 2, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    # Negative dimension tests
    def test_index_copy_negative_dim_last(self):
        """Test index_copy decomposition with dim=-1"""
        input_tensor = torch.zeros(3, 4)
        index = torch.tensor([0, 2])
        source = torch.ones(3, 2)

        result = decompositions.index_copy_decomposition(input_tensor, -1, index, source)
        expected = input_tensor.index_copy(-1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_negative_dim_first(self):
        """Test index_copy decomposition with dim=-2 on 2D tensor"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([1, 3])
        source = torch.ones(2, 3)

        result = decompositions.index_copy_decomposition(input_tensor, -2, index, source)
        expected = input_tensor.index_copy(-2, index, source)

        torch.testing.assert_close(result, expected)

    # Different dtypes
    def test_index_copy_float64(self):
        """Test index_copy with float64 dtype"""
        input_tensor = torch.zeros(4, 3, dtype=torch.float64)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3, dtype=torch.float64)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_int32(self):
        """Test index_copy with int32 dtype"""
        input_tensor = torch.zeros(4, 3, dtype=torch.int32)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3, dtype=torch.int32)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    # Edge cases
    def test_index_copy_single_index(self):
        """Test index_copy with single index"""
        input_tensor = torch.zeros(5, 3)
        index = torch.tensor([2])
        source = torch.ones(1, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_all_indices(self):
        """Test index_copy replacing all elements along dim"""
        input_tensor = torch.zeros(3, 4)
        index = torch.tensor([0, 1, 2])
        source = torch.ones(3, 4) * 5

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_non_contiguous_index(self):
        """Test index_copy with non-contiguous indices"""
        input_tensor = torch.zeros(6, 3)
        index = torch.tensor([5, 1, 3])  # Out of order
        source = torch.arange(9, dtype=torch.float32).reshape(3, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_4d_tensor(self):
        """Test index_copy on 4D tensor (batch, channel, height, width)"""
        input_tensor = torch.zeros(2, 4, 3, 3)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 2, 3, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_preserves_existing_values(self):
        """Test that index_copy preserves values at non-indexed positions"""
        input_tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        index = torch.tensor([1])
        source = torch.ones(1, 3) * 99

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)
        # Verify non-indexed rows are preserved
        assert result[0, 0] == 0.0
        assert result[2, 0] == 6.0
        assert result[3, 0] == 9.0

    def test_index_copy_with_random_values(self):
        """Test index_copy with random tensor values"""
        torch.manual_seed(42)
        input_tensor = torch.randn(5, 4, 3)
        index = torch.tensor([0, 2, 4])
        source = torch.randn(3, 4, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_middle_dim(self):
        """Test index_copy on middle dimension of 3D tensor"""
        input_tensor = torch.zeros(2, 5, 3)
        index = torch.tensor([1, 4])
        source = torch.ones(2, 2, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_last_dim_3d(self):
        """Test index_copy on last dimension of 3D tensor"""
        input_tensor = torch.zeros(2, 3, 6)
        index = torch.tensor([0, 3, 5])
        source = torch.ones(2, 3, 3)

        result = decompositions.index_copy_decomposition(input_tensor, 2, index, source)
        expected = input_tensor.index_copy(2, index, source)

        torch.testing.assert_close(result, expected)
