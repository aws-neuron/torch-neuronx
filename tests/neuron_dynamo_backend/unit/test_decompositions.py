# ruff: noqa: N806

"""
Unit tests for neuron_dynamo_backend decompositions module
"""

import itertools
import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch._decomp import decomposition_table

from torch_neuronx.neuron_dynamo_backend import decompositions


class TestDecompositionTable:
    """Test decomposition table functionality"""

    def test_neuron_decompositions_contains_expected_entries(self):
        """Test that neuron_decompositions contains expected entries"""
        # Check that it's a dictionary
        assert isinstance(decompositions.neuron_decompositions, dict)

        # Check that specific ops are in the table
        assert torch.ops.aten.as_strided.default in decompositions.neuron_decompositions
        assert torch.ops.aten.scalar_tensor.default in decompositions.neuron_decompositions
        assert torch.ops.aten.index_copy.default in decompositions.neuron_decompositions

        # Verify the functions are callable
        assert callable(decompositions.neuron_decompositions[torch.ops.aten.as_strided.default])
        assert callable(decompositions.neuron_decompositions[torch.ops.aten.scalar_tensor.default])

    def test_get_decomposition_table_selective(self):
        """Test selective decomposition table includes only specified ops"""
        decomp_table = decompositions.get_decomposition_table()

        # Should contain our custom decompositions
        assert torch.ops.aten.as_strided.default in decomp_table

        # Should be the same as neuron_decompositions
        assert decomp_table is decompositions.neuron_decompositions


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

        result = decompositions.as_strided(input_tensor, size, stride, storage_offset)

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

        result = decompositions.as_strided(input_tensor, size, stride, storage_offset)

        expected = torch.as_strided(input_tensor, size, stride, storage_offset)
        torch.testing.assert_close(result, expected)

    def test_as_strided_multidimensional(self):
        """Test as_strided with complex multidimensional striding"""
        input_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        size = (2, 2, 2)
        stride = (12, 4, 2)  # Original strides: (12, 4, 1), skip every other element in last dim
        storage_offset = 0

        result = decompositions.as_strided(input_tensor, size, stride, storage_offset)

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

        result = decompositions.index_copy(input_tensor, 0, index, source)
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

        result = decompositions.index_copy(input_tensor, dim, index, source)
        expected = input_tensor.index_copy(dim, index, source)

        torch.testing.assert_close(result, expected)

    # NaN/Inf edge cases
    def test_index_copy_with_nan(self):
        """Test index_copy with NaN values in source"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([0, 2])
        source = torch.tensor([[float("nan"), 1.0, 2.0], [3.0, float("nan"), 5.0]])

        result = decompositions.index_copy(input_tensor, 0, index, source)
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

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_with_nan_in_input(self):
        """Test index_copy preserves NaN in non-indexed positions"""
        input_tensor = torch.tensor([[float("nan"), 1.0], [2.0, 3.0], [4.0, float("nan")]])
        index = torch.tensor([1])
        source = torch.tensor([[99.0, 99.0]])

        result = decompositions.index_copy(input_tensor, 0, index, source)
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

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = torch.zeros(4, 3).index_copy(0, torch.tensor([0, 2]), torch.ones(2, 3))

        torch.testing.assert_close(result.to("cpu"), expected)

    @pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_index_copy_neuron_dtypes(self, dtype):
        """Test index_copy on Neuron with various dtypes"""
        input_tensor = torch.zeros(4, 3, dtype=dtype).to("neuron")
        index = torch.tensor([0, 2]).to("neuron")
        source = torch.ones(2, 3, dtype=dtype).to("neuron")

        result = decompositions.index_copy(input_tensor, 0, index, source)
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

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_2d_dim0(self):
        """Test index_copy decomposition on 2D tensor along dim 0"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3)

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_2d_dim1(self):
        """Test index_copy decomposition on 2D tensor along dim 1"""
        input_tensor = torch.zeros(3, 5)
        index = torch.tensor([1, 3])
        source = torch.ones(3, 2)

        result = decompositions.index_copy(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_3d(self):
        """Test index_copy decomposition on 3D tensor"""
        input_tensor = torch.zeros(2, 4, 3)
        index = torch.tensor([1, 3])
        source = torch.ones(2, 2, 3)

        result = decompositions.index_copy(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    # Negative dimension tests
    def test_index_copy_negative_dim_last(self):
        """Test index_copy decomposition with dim=-1"""
        input_tensor = torch.zeros(3, 4)
        index = torch.tensor([0, 2])
        source = torch.ones(3, 2)

        result = decompositions.index_copy(input_tensor, -1, index, source)
        expected = input_tensor.index_copy(-1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_negative_dim_first(self):
        """Test index_copy decomposition with dim=-2 on 2D tensor"""
        input_tensor = torch.zeros(4, 3)
        index = torch.tensor([1, 3])
        source = torch.ones(2, 3)

        result = decompositions.index_copy(input_tensor, -2, index, source)
        expected = input_tensor.index_copy(-2, index, source)

        torch.testing.assert_close(result, expected)

    # Different dtypes
    def test_index_copy_float64(self):
        """Test index_copy with float64 dtype"""
        input_tensor = torch.zeros(4, 3, dtype=torch.float64)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3, dtype=torch.float64)

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_int32(self):
        """Test index_copy with int32 dtype"""
        input_tensor = torch.zeros(4, 3, dtype=torch.int32)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 3, dtype=torch.int32)

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    # Edge cases
    def test_index_copy_single_index(self):
        """Test index_copy with single index"""
        input_tensor = torch.zeros(5, 3)
        index = torch.tensor([2])
        source = torch.ones(1, 3)

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_all_indices(self):
        """Test index_copy replacing all elements along dim"""
        input_tensor = torch.zeros(3, 4)
        index = torch.tensor([0, 1, 2])
        source = torch.ones(3, 4) * 5

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_non_contiguous_index(self):
        """Test index_copy with non-contiguous indices"""
        input_tensor = torch.zeros(6, 3)
        index = torch.tensor([5, 1, 3])  # Out of order
        source = torch.arange(9, dtype=torch.float32).reshape(3, 3)

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_4d_tensor(self):
        """Test index_copy on 4D tensor (batch, channel, height, width)"""
        input_tensor = torch.zeros(2, 4, 3, 3)
        index = torch.tensor([0, 2])
        source = torch.ones(2, 2, 3, 3)

        result = decompositions.index_copy(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_preserves_existing_values(self):
        """Test that index_copy preserves values at non-indexed positions"""
        input_tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        index = torch.tensor([1])
        source = torch.ones(1, 3) * 99

        result = decompositions.index_copy(input_tensor, 0, index, source)
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

        result = decompositions.index_copy(input_tensor, 0, index, source)
        expected = input_tensor.index_copy(0, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_middle_dim(self):
        """Test index_copy on middle dimension of 3D tensor"""
        input_tensor = torch.zeros(2, 5, 3)
        index = torch.tensor([1, 4])
        source = torch.ones(2, 2, 3)

        result = decompositions.index_copy(input_tensor, 1, index, source)
        expected = input_tensor.index_copy(1, index, source)

        torch.testing.assert_close(result, expected)

    def test_index_copy_last_dim_3d(self):
        """Test index_copy on last dimension of 3D tensor"""
        input_tensor = torch.zeros(2, 3, 6)
        index = torch.tensor([0, 3, 5])
        source = torch.ones(2, 3, 3)

        result = decompositions.index_copy(input_tensor, 2, index, source)
        expected = input_tensor.index_copy(2, index, source)

        torch.testing.assert_close(result, expected)


class TestLinearBackwardDecomposition:
    """Comprehensive tests for linear_backward decomposition"""

    def _compute_reference_gradients(self, input_, weight_, grad_output_, output_mask):
        """Compute reference gradients using PyTorch autograd for comparison"""
        input_copy = input_.detach().clone().requires_grad_(output_mask[0])
        weight_copy = weight_.detach().clone().requires_grad_(output_mask[1])
        if output_mask[2]:
            bias_copy = torch.zeros(
                weight_.shape[0], device=weight_.device, dtype=weight_.dtype
            ).requires_grad_(True)
            output = torch.nn.functional.linear(input_copy, weight_copy, bias_copy)
        else:
            bias_copy = None
            output = torch.nn.functional.linear(input_copy, weight_copy)
        output.backward(grad_output_)
        grad_input = input_copy.grad if output_mask[0] else None
        grad_weight = weight_copy.grad if output_mask[1] else None
        grad_bias = bias_copy.grad if output_mask[2] else None
        return grad_input, grad_weight, grad_bias

    def test_linear_backward_basic_functionality(self):
        """Test basic linear backward decomposition functionality"""
        torch.manual_seed(1)
        batch_size, in_features, out_features = 3, 4, 5
        input_ = torch.randn(batch_size, in_features)
        weight_ = torch.randn(out_features, in_features)
        grad_output_ = torch.randn(batch_size, out_features)
        output_mask = [True, True, True]
        result_grad_input, result_grad_weight, result_grad_bias = (
            decompositions.linear_backward_decomposition(input_, grad_output_, weight_, output_mask)
        )
        ref_grad_input, ref_grad_weight, ref_grad_bias = self._compute_reference_gradients(
            input_, weight_, grad_output_, output_mask
        )

        assert result_grad_input.shape == input_.shape
        assert result_grad_weight.shape == weight_.shape
        assert result_grad_bias.shape == (out_features,)
        torch.testing.assert_close(result_grad_input, ref_grad_input, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(result_grad_weight, ref_grad_weight, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(result_grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "output_mask",
        list(itertools.product([True, False], [True, False], [True, False])),
    )
    def test_linear_backward_output_mask_combinations(self, output_mask):
        """Test different output_mask combinations"""
        torch.manual_seed(1)
        batch_size, in_features, out_features = 2, 3, 4
        input_ = torch.randn(batch_size, in_features)
        weight_ = torch.randn(out_features, in_features)
        grad_output_ = torch.randn(batch_size, out_features)
        result_grad_input, result_grad_weight, result_grad_bias = (
            decompositions.linear_backward_decomposition(input_, grad_output_, weight_, output_mask)
        )
        if not output_mask[0]:
            assert result_grad_input is None
        else:
            assert result_grad_input is not None
            assert result_grad_input.shape == input_.shape
        if not output_mask[1]:
            assert result_grad_weight is None
        else:
            assert result_grad_weight is not None
            assert result_grad_weight.shape == weight_.shape
        if not output_mask[2]:
            assert result_grad_bias is None
        else:
            assert result_grad_bias is not None
            assert result_grad_bias.shape == (out_features,)

    @pytest.mark.parametrize(
        "dtype,rtol,atol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 1e-2, 1e-2),
            (torch.bfloat16, 1e-2, 1e-2),
        ],
    )
    @pytest.mark.parametrize(
        "input_shape,weight_shape,grad_output_shape",
        [
            ((4,), (5, 4), (5,)),  # 1D input
            ((3, 4), (5, 4), (3, 5)),  # Basic 2D
            ((2, 3, 4), (5, 4), (2, 3, 5)),  # 3D input
            ((1, 2, 3, 4), (5, 4), (1, 2, 3, 5)),  # 4D input
        ],
    )
    def test_linear_backward_dtypes_shapes(
        self, dtype, rtol, atol, input_shape, weight_shape, grad_output_shape
    ):
        """Test linear backward with various tensor shapes"""
        torch.manual_seed(1)
        input_ = torch.randn(input_shape, dtype=dtype)
        weight_ = torch.randn(weight_shape, dtype=dtype)
        grad_output_ = torch.randn(grad_output_shape, dtype=dtype)
        output_mask = [True, True, True]

        result_grad_input, result_grad_weight, result_grad_bias = (
            decompositions.linear_backward_decomposition(input_, grad_output_, weight_, output_mask)
        )
        ref_grad_input, ref_grad_weight, ref_grad_bias = self._compute_reference_gradients(
            input_, weight_, grad_output_, output_mask
        )

        assert result_grad_input.shape == input_shape
        assert result_grad_weight.shape == weight_shape
        assert result_grad_bias.shape == (grad_output_shape[-1],)
        torch.testing.assert_close(result_grad_input, ref_grad_input, rtol=rtol, atol=atol)
        torch.testing.assert_close(result_grad_weight, ref_grad_weight, rtol=rtol, atol=atol)
        torch.testing.assert_close(result_grad_bias, ref_grad_bias, rtol=rtol, atol=atol)
