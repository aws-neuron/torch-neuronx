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


class TestTupleHelper:
    """Test _tuple helper function"""

    @pytest.mark.parametrize(
        "x,ndim,expected",
        [
            # int input
            (1, 2, (1, 1)),
            (3, 3, (3, 3, 3)),
            (0, 1, (0,)),
            # tuple exact length
            ((1, 2), 2, (1, 2)),
            ((1, 2, 3), 3, (1, 2, 3)),
            # tuple shorter than ndim - repeats last element
            ((1,), 2, (1, 1)),
            ((1,), 3, (1, 1, 1)),
            ((1, 2), 4, (1, 2, 2, 2)),
            # tuple longer than ndim - keeps original
            ((1, 2, 3), 2, (1, 2, 3)),
            # empty tuple/list
            ((), 2, (0, 0)),
            ([], 3, (0, 0, 0)),
            # list input
            ([1, 2], 2, (1, 2)),
            ([1], 2, (1, 1)),
        ],
    )
    def test_tuple(self, x, ndim, expected):
        """Test _tuple converts input to tuple of length ndim"""
        assert decompositions._tuple(x, ndim) == expected


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


class TestValueSelectingReductionBackwardDecomposition:
    """Tests for value_selecting_reduction_backward decomposition"""

    def _reference(self, grad, dim, indices, sizes, keepdim):
        """Compute reference using PyTorch's native implementation."""
        return torch.ops.aten.value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim)

    def _make_test_input(self, shape):
        """Create a deterministic test tensor from arange."""
        numel = 1
        for s in shape:
            numel *= s
        return torch.linspace(-1.0, 1.0, numel).reshape(shape)

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((5,), 0),  # 1D
            ((4, 3), 0),  # 2D dim0
            ((4, 3), 1),  # 2D dim1
            ((2, 4, 3), 0),  # 3D dim0
            ((2, 4, 3), 1),  # 3D dim1
            ((2, 4, 3), 2),  # 3D dim2
            ((2, 3, 4, 5), 0),  # 4D dim0
            ((2, 3, 4, 5), 1),  # 4D dim1
            ((2, 3, 4, 5), 2),  # 4D dim2
            ((2, 3, 4, 5), 3),  # 4D dim3
        ],
    )
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_shapes_and_dims(self, shape, dim, keepdim):
        """Test with various shape/dim/keepdim combinations"""
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=dim, keepdim=keepdim)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, dim, indices, list(shape), keepdim
        )
        expected = self._reference(grad, dim, indices, list(shape), keepdim)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [-1, -2])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_negative_dim(self, dim, keepdim):
        """Test with negative dimension values"""
        shape = (3, 4, 5)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=dim, keepdim=keepdim)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, dim, indices, list(shape), keepdim
        )
        expected = self._reference(grad, dim, indices, list(shape), keepdim)

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_dtypes(self, dtype):
        """Test with various floating point dtypes"""
        shape = (4, 5)
        x = self._make_test_input(shape).to(dtype)
        values, indices = torch.max(x, dim=1, keepdim=False)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), False
        )
        expected = self._reference(grad, 1, indices, list(shape), False)

        torch.testing.assert_close(result, expected)

    def test_min_backward(self):
        """Test that decomposition works for min reduction backward"""
        shape = (3, 4)
        x = self._make_test_input(shape)
        values, indices = torch.min(x, dim=0, keepdim=False)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 0, indices, list(shape), False
        )
        expected = self._reference(grad, 0, indices, list(shape), False)

        torch.testing.assert_close(result, expected)

    def test_gradient_only_at_selected_positions(self):
        """Test that gradient is nonzero only at positions selected by indices"""
        shape = (3, 5)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=1, keepdim=False)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), False
        )

        # Each row should have exactly one nonzero entry at the index position
        for i in range(shape[0]):
            assert result[i, indices[i]] == 1.0
            assert result[i].sum() == 1.0

    def test_keepdim_true_shape(self):
        """Test output shape when keepdim=True"""
        shape = (3, 4, 5)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=1, keepdim=True)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), True
        )

        assert result.shape == torch.Size(shape)

    def test_keepdim_false_shape(self):
        """Test output shape when keepdim=False"""
        shape = (3, 4, 5)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=1, keepdim=False)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), False
        )

        assert result.shape == torch.Size(shape)

    def test_single_element_dim(self):
        """Test with a dimension of size 1"""
        shape = (3, 1, 5)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=1, keepdim=False)
        grad = torch.ones_like(values)

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), False
        )
        expected = self._reference(grad, 1, indices, list(shape), False)

        torch.testing.assert_close(result, expected)

    def test_nonuniform_grad_values(self):
        """Test with non-uniform gradient values"""
        shape = (3, 4)
        x = self._make_test_input(shape)
        values, indices = torch.max(x, dim=1, keepdim=False)
        grad = torch.tensor([2.0, 3.0, 5.0])

        result = decompositions.value_selecting_reduction_backward(
            grad, 1, indices, list(shape), False
        )
        expected = self._reference(grad, 1, indices, list(shape), False)

        torch.testing.assert_close(result, expected)

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.value_selecting_reduction_backward.default
            in decompositions.neuron_decompositions
        ), "aten.value_selecting_reduction_backward.default not found in neuron_decompositions"


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


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLogsumexpDecompositionE2E:
    """Test logsumexp decomposition"""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((4, 8), 0),
            ((4, 8), 1),
            ((4, 8), -1),
            ((3, 4, 5), 0),
            ((3, 4, 5), 1),
            ((3, 4, 5), 2),
            ((2, 3, 4, 5), -1),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_logsumexp_e2e(self, shape, dim, dtype):
        """Test logsumexp"""
        torch.manual_seed(42)

        def logsumexp_fn(x):
            return torch.logsumexp(x, dim=dim)

        compiled_fn = torch.compile(logsumexp_fn, backend="neuron")

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = logsumexp_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_logsumexp_keepdim(self, keepdim):
        """Test logsumexp with keepdim parameter"""
        torch.manual_seed(42)

        def logsumexp_fn(x):
            return torch.logsumexp(x, dim=1, keepdim=keepdim)

        compiled_fn = torch.compile(logsumexp_fn, backend="neuron")

        x_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = logsumexp_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_logsumexp_backward(self):
        """Test logsumexp backward pass"""
        torch.manual_seed(42)

        def logsumexp_and_sum(x):
            return torch.logsumexp(x, dim=1).sum()

        compiled_fn = torch.compile(logsumexp_and_sum, backend="neuron")

        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = logsumexp_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBinaryCrossEntropyDecompositionE2E:
    """Test binary_cross_entropy decomposition"""

    @pytest.mark.parametrize(
        "shape",
        [
            (8,),
            (4, 8),
            (2, 4, 8),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_binary_cross_entropy_e2e(self, shape, dtype):
        """Test binary_cross_entropy"""
        torch.manual_seed(42)

        def bce_fn(input, target):
            return torch.nn.functional.binary_cross_entropy(input, target)

        compiled_fn = torch.compile(bce_fn, backend="neuron")

        input_cpu = torch.sigmoid(torch.randn(shape, dtype=dtype))
        target_cpu = torch.rand(shape, dtype=dtype)
        input_neuron = input_cpu.detach().clone().to("neuron")
        target_neuron = target_cpu.detach().clone().to("neuron")

        result_cpu = bce_fn(input_cpu, target_cpu)
        result_neuron = compiled_fn(input_neuron, target_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_binary_cross_entropy_reduction(self, reduction):
        """Test binary_cross_entropy with different reductions"""
        torch.manual_seed(42)

        def bce_fn(input, target):
            return torch.nn.functional.binary_cross_entropy(input, target, reduction=reduction)

        compiled_fn = torch.compile(bce_fn, backend="neuron")

        input_cpu = torch.sigmoid(torch.randn(4, 8))
        target_cpu = torch.rand(4, 8)
        input_neuron = input_cpu.detach().clone().to("neuron")
        target_neuron = target_cpu.detach().clone().to("neuron")

        result_cpu = bce_fn(input_cpu, target_cpu)
        result_neuron = compiled_fn(input_neuron, target_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNativeBatchNormBackwardDecompositionE2E:
    """Test native_batch_norm_backward decomposition"""

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3, 4, 4),
            (4, 8, 2, 2),
            (1, 16, 8, 8),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_batch_norm_backward_e2e(self, shape, dtype):
        """Test batch_norm backward"""
        torch.manual_seed(42)
        N, C, H, W = shape

        class BNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(C)

            def forward(self, x):
                return self.bn(x).sum()

        model_cpu = BNModel()
        model_cpu.train()
        model_neuron = BNModel()
        model_neuron.load_state_dict(model_cpu.state_dict())
        model_neuron.to("neuron")
        model_neuron.train()

        compiled_model = torch.compile(model_neuron, backend="neuron")

        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = model_cpu(x_cpu)
        loss_neuron = compiled_model(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)

    def test_batch_norm_1d_backward(self):
        """Test BatchNorm1d backward"""
        torch.manual_seed(42)

        class BN1dModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(8)

            def forward(self, x):
                return self.bn(x).sum()

        model_cpu = BN1dModel()
        model_cpu.train()
        model_neuron = BN1dModel()
        model_neuron.load_state_dict(model_cpu.state_dict())
        model_neuron.to("neuron")
        model_neuron.train()

        compiled_model = torch.compile(model_neuron, backend="neuron")

        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = model_cpu(x_cpu)
        loss_neuron = compiled_model(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestForeachPowDecompositionE2E:
    """Test foreach_pow decomposition"""

    @pytest.mark.parametrize("exponent", [0.5, 2.0, 3.0])
    def test_foreach_pow_e2e(self, exponent):
        """Test foreach_pow"""
        torch.manual_seed(42)

        def foreach_pow_fn(t1, t2):
            tensors = [t1, t2]
            result = torch._foreach_pow(tensors, exponent)
            return result[0].sum() + result[1].sum()

        compiled_fn = torch.compile(foreach_pow_fn, backend="neuron")

        # use positive values for fractional exponents
        t1_cpu = torch.abs(torch.randn(4, 4)) + 0.1
        t2_cpu = torch.abs(torch.randn(3, 3)) + 0.1
        t1_neuron = t1_cpu.detach().clone().to("neuron")
        t2_neuron = t2_cpu.detach().clone().to("neuron")

        result_cpu = foreach_pow_fn(t1_cpu, t2_cpu)
        result_neuron = compiled_fn(t1_neuron, t2_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_foreach_pow_multiple_tensors(self):
        """Test foreach_pow with multiple tensors"""
        torch.manual_seed(42)

        def foreach_pow_fn(t1, t2, t3):
            tensors = [t1, t2, t3]
            result = torch._foreach_pow(tensors, 2.0)
            return sum(r.sum() for r in result)

        compiled_fn = torch.compile(foreach_pow_fn, backend="neuron")

        t1_cpu = torch.randn(2, 3)
        t2_cpu = torch.randn(4, 5)
        t3_cpu = torch.randn(
            6,
        )
        t1_neuron = t1_cpu.detach().clone().to("neuron")
        t2_neuron = t2_cpu.detach().clone().to("neuron")
        t3_neuron = t3_cpu.detach().clone().to("neuron")

        result_cpu = foreach_pow_fn(t1_cpu, t2_cpu, t3_cpu)
        result_neuron = compiled_fn(t1_neuron, t2_neuron, t3_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNansumDecomposition:
    """Test nansum decomposition - not E2E because NAN causes the following error:
    NRT model execution failed, Execution completed with numerical errors (NaN)"""

    def test_nansum_decomposition_registered(self):
        """Test that nansum decomposition is in the table"""
        assert (
            torch.ops.aten.nansum.default in decompositions.neuron_decompositions
        ), "aten.nansum.default not found in neuron_decompositions"

    def test_nansum_is_decomposed(self):
        """Test that nansum is decomposed (not passed through)"""
        captured_ops = []

        def capture_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    captured_ops.append(node.target)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=capture_backend, fullgraph=True)
        def fn(x):
            return torch.nansum(x)

        x = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        result = fn(x)

        # check nansum is decomposed
        assert (
            torch.ops.aten.nansum.default not in captured_ops
        ), f"nansum should be decomposed but was found in: {captured_ops}"

        expected = torch.tensor(7.0)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_nansum_with_dim(self, dim):
        """Test nansum with dim parameter"""
        captured_ops = []

        def capture_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    captured_ops.append(node.target)
            return gm.forward

        torch._dynamo.reset()

        def fn(x):
            return torch.nansum(x, dim=dim)

        compiled_fn = torch.compile(fn, backend=capture_backend, fullgraph=True)

        x = torch.randn(4, 8)
        x[0, 0] = float("nan")
        x[2, 3] = float("nan")

        result = compiled_fn(x)
        expected = fn(x)

        assert torch.ops.aten.nansum.default not in captured_ops
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAddrDecompositionE2E:
    """Test addr decomposition"""

    @pytest.mark.parametrize(
        "m,n",
        [
            (3, 4),
            (5, 5),
            (2, 8),
            (10, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_addr_e2e(self, m, n, dtype):
        """Test addr"""
        torch.manual_seed(42)

        def addr_fn(input, vec1, vec2):
            return torch.addr(input, vec1, vec2)

        compiled_fn = torch.compile(addr_fn, backend="neuron")

        input_cpu = torch.randn(m, n, dtype=dtype)
        vec1_cpu = torch.randn(m, dtype=dtype)
        vec2_cpu = torch.randn(n, dtype=dtype)
        input_neuron = input_cpu.detach().clone().to("neuron")
        vec1_neuron = vec1_cpu.detach().clone().to("neuron")
        vec2_neuron = vec2_cpu.detach().clone().to("neuron")

        result_cpu = addr_fn(input_cpu, vec1_cpu, vec2_cpu)
        result_neuron = compiled_fn(input_neuron, vec1_neuron, vec2_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "alpha,beta",
        [
            (1.0, 1.0),
            (2.0, 0.5),
            (0.0, 1.0),
            (1.0, 0.0),
        ],
    )
    def test_addr_alpha_beta(self, alpha, beta):
        """Test addr with alpha and beta parameters"""
        torch.manual_seed(42)

        def addr_fn(input, vec1, vec2):
            return torch.addr(input, vec1, vec2, alpha=alpha, beta=beta)

        compiled_fn = torch.compile(addr_fn, backend="neuron")

        input_cpu = torch.randn(3, 4)
        vec1_cpu = torch.randn(3)
        vec2_cpu = torch.randn(4)
        input_neuron = input_cpu.detach().clone().to("neuron")
        vec1_neuron = vec1_cpu.detach().clone().to("neuron")
        vec2_neuron = vec2_cpu.detach().clone().to("neuron")

        result_cpu = addr_fn(input_cpu, vec1_cpu, vec2_cpu)
        result_neuron = compiled_fn(input_neuron, vec1_neuron, vec2_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_addr_backward(self):
        """Test addr backward pass"""
        torch.manual_seed(42)

        def addr_and_sum(input, vec1, vec2):
            return torch.addr(input, vec1, vec2).sum()

        compiled_fn = torch.compile(addr_and_sum, backend="neuron")

        input_cpu = torch.randn(3, 4, requires_grad=True)
        vec1_cpu = torch.randn(3, requires_grad=True)
        vec2_cpu = torch.randn(4, requires_grad=True)
        input_neuron = input_cpu.detach().clone().to("neuron").requires_grad_(True)
        vec1_neuron = vec1_cpu.detach().clone().to("neuron").requires_grad_(True)
        vec2_neuron = vec2_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = addr_and_sum(input_cpu, vec1_cpu, vec2_cpu)
        loss_neuron = compiled_fn(input_neuron, vec1_neuron, vec2_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(vec1_neuron.grad.cpu(), vec1_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(vec2_neuron.grad.cpu(), vec2_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRshiftScalarDecompositionE2E:
    """Test right shift decomposition"""

    @pytest.mark.parametrize("shift", [1, 2, 4, 8])
    def test_rshift_e2e(self, shift):
        """Test right shift"""
        torch.manual_seed(42)

        def rshift_fn(x):
            return x >> shift

        compiled_fn = torch.compile(rshift_fn, backend="neuron")

        x_cpu = torch.randint(0, 1000, (4, 8), dtype=torch.int32)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = rshift_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_rshift_dtypes(self, dtype):
        """Test right shift with different dtypes"""
        torch.manual_seed(42)

        def rshift_fn(x):
            return x >> 2

        compiled_fn = torch.compile(rshift_fn, backend="neuron")

        x_cpu = torch.randint(0, 1000, (4, 8), dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = rshift_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    def test_rshift_2d(self):
        """Test right shift on 2D tensor"""
        torch.manual_seed(42)

        def rshift_fn(x):
            return x >> 3

        compiled_fn = torch.compile(rshift_fn, backend="neuron")

        x_cpu = torch.tensor([[8, 16, 32], [64, 128, 256]], dtype=torch.int32)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = rshift_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAsStridedScatterDecompositionE2E:
    """Test as_strided_scatter decomposition"""

    def test_as_strided_scatter_decomposition_registered(self):
        """Test that as_strided_scatter decomposition is in the table"""
        assert (
            torch.ops.aten.as_strided_scatter.default in decompositions.neuron_decompositions
        ), "aten.as_strided_scatter.default not found in neuron_decompositions"

    def test_as_strided_scatter_is_decomposed(self):
        """Test that as_strided_scatter is decomposed (not passed through)"""
        from torch._decomp import get_decompositions

        captured_ops = []

        def capture_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    captured_ops.append(node.target)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend=capture_backend, fullgraph=True)
        def fn(x, src):
            return torch.as_strided_scatter(x, src, (2, 2), (2, 1), 0)

        x = torch.arange(8, dtype=torch.float32)
        src = torch.full((2, 2), 99.0)
        result = fn(x, src)

        # Verify as_strided_scatter is NOT in captured ops (i.e., it was decomposed)
        assert (
            torch.ops.aten.as_strided_scatter.default not in captured_ops
        ), f"as_strided_scatter should be decomposed but was found in: {captured_ops}"

        # Verify result is correct
        expected = torch.as_strided_scatter(x, src, (2, 2), (2, 1), 0)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "x_size,src_shape,size,stride,offset",
        [
            (8, (2, 2), (2, 2), (2, 1), 0),
            (10, (2,), (2,), (1,), 4),
            (12, (2, 3), (2, 3), (3, 1), 0),
        ],
    )
    def test_as_strided_scatter_decomposition_correctness(
        self, x_size, src_shape, size, stride, offset
    ):
        """Test decomposed as_strided_scatter matches native PyTorch"""
        captured_ops = []

        def capture_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    captured_ops.append(node.target)
            return gm.forward

        torch._dynamo.reset()

        def fn(x, src):
            return torch.as_strided_scatter(x, src, size, stride, offset)

        compiled_fn = torch.compile(fn, backend=capture_backend, fullgraph=True)

        x = torch.arange(x_size, dtype=torch.float32)
        src = torch.full(src_shape, 99.0)

        result = compiled_fn(x, src)
        expected = fn(x, src)

        assert torch.ops.aten.as_strided_scatter.default not in captured_ops
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestFoldDecompositionE2E:
    """E2E tests for torch.nn.functional.fold using torch.compile with neuron backend"""

    @pytest.mark.parametrize(
        "batch_size,channels,output_size,kernel_size",
        [
            (1, 1, (4, 4), (2, 2)),
            (2, 3, (6, 6), (3, 3)),
            (16, 1, (5, 5), (2, 2)),  # large batch
            (1, 8, (4, 4), (2, 2)),  # multi-channel
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_fold_basic(self, batch_size, channels, output_size, kernel_size, dtype):
        """Test fold with various batch/channel/output/kernel combinations"""
        torch.manual_seed(42)

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        # Input shape for fold: (N, C * prod(kernel_size), L)
        L = 1
        for i in range(2):
            L *= (output_size[i] - kernel_size[i]) + 1
        C_times_K = channels * kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(batch_size, C_times_K, L, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-1 if dtype == torch.bfloat16 else rtol
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 3e-1 if dtype == torch.bfloat16 else atol
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "output_size,kernel_size,stride,padding,dilation",
        [
            ((8, 8), (3, 3), 1, 0, 1),  # default params
            ((8, 8), (3, 3), 2, 0, 1),  # stride
            ((6, 6), (3, 3), 1, 1, 1),  # padding
            ((8, 8), (3, 3), 1, 0, 2),  # dilation
            ((10, 10), (3, 3), 2, 1, 1),  # all combined
        ],
    )
    def test_fold_parameters(self, output_size, kernel_size, stride, padding, dilation):
        """Test fold with various parameter combinations"""
        torch.manual_seed(42)

        def fold_fn(x):
            return torch.nn.functional.fold(
                x, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride
            )

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        effective_k0 = (kernel_size[0] - 1) * dilation + 1
        effective_k1 = (kernel_size[1] - 1) * dilation + 1
        L = ((output_size[0] + 2 * padding - effective_k0) // stride + 1) * (
            (output_size[1] + 2 * padding - effective_k1) // stride + 1
        )
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "output_size,kernel_size",
        [
            ((4, 4), (1, 1)),  # 1x1 kernel edge case
            ((6, 8), (2, 3)),  # non-square output and kernel
            ((10, 10), (7, 7)),  # large kernel
        ],
    )
    def test_fold_edge_cases(self, output_size, kernel_size):
        """Test fold with edge cases: 1x1 kernel, non-square dimensions, large kernel"""
        torch.manual_seed(42)

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_fold_backward(self):
        """Test fold backward pass"""
        torch.manual_seed(42)
        output_size = (4, 4)
        kernel_size = (2, 2)

        def fold_and_sum(x):
            return torch.nn.functional.fold(x, output_size, kernel_size).sum()

        compiled_fn = torch.compile(fold_and_sum, backend="neuron")

        L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = fold_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_fold_sequential_values(self):
        """Test fold with sequential input values"""
        ipt = torch.arange(36).reshape(1, 9, 4).float()
        ipt_neuron = ipt.detach().to("neuron")

        def fold_fn(x):
            return torch.nn.functional.fold(x, (4, 4), (3, 3))

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        out_cpu = fold_fn(ipt)
        out_neuron = compiled_fn(ipt_neuron).to("cpu")

        torch.testing.assert_close(out_neuron, out_cpu, rtol=1e-4, atol=1e-4)

    def test_fold_unfold_inverse(self):
        """Test that fold and unfold are inverse operations"""
        torch.manual_seed(42)
        output_size = (6, 6)
        kernel_size = (3, 3)

        img = torch.randn(1, 2, 6, 6)

        def unfold_fold_fn(x):
            unfolded = torch.nn.functional.unfold(x, kernel_size)
            folded = torch.nn.functional.fold(unfolded, output_size, kernel_size)
            return folded

        compiled_fn = torch.compile(unfold_fold_fn, backend="neuron")

        img_neuron = img.detach().to("neuron")

        result_cpu = unfold_fold_fn(img)
        result_neuron = compiled_fn(img_neuron).to("cpu")

        torch.testing.assert_close(result_neuron, result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.bfloat16]
    )
    def test_fold_single_location(self, dtype):
        """Test fold when only 1 spatial location (kernel size = output size)"""
        torch.manual_seed(42)
        output_size = (3, 3)
        kernel_size = (3, 3)

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        # Only 1 location: L = 1
        L = 1
        C_times_K = 2 * kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.bfloat16]
    )
    def test_fold_with_negative_values(self, dtype):
        """Test fold with negative input values"""
        torch.manual_seed(42)
        output_size = (4, 4)
        kernel_size = (2, 2)

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = -torch.abs(torch.randn(1, C_times_K, L, dtype=dtype))  # All negative
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_fold_mixed_stride_padding(self, dtype):
        """Test fold with non-overlapping windows (stride >= kernel_size)"""
        torch.manual_seed(42)
        output_size = (8, 8)
        kernel_size = (2, 2)
        stride = 2  # Non-overlapping

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size, stride=stride)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        L = ((output_size[0] - kernel_size[0]) // stride + 1) * (
            (output_size[1] - kernel_size[1]) // stride + 1
        )
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(2, C_times_K, L, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_fold_high_dilation(self, dtype):
        """Test fold with high dilation value"""
        torch.manual_seed(42)
        output_size = (12, 12)
        kernel_size = (3, 3)
        dilation = 3

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size, dilation=dilation)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        effective_k0 = (kernel_size[0] - 1) * dilation + 1
        effective_k1 = (kernel_size[1] - 1) * dilation + 1
        L = (output_size[0] - effective_k0 + 1) * (output_size[1] - effective_k1 + 1)
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-1 if dtype == torch.bfloat16 else rtol
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 5e-2 if dtype == torch.bfloat16 else atol
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_fold_extreme_aspect_ratio(self, dtype):
        """Test fold with extreme aspect ratio and higher precision dtypes"""
        torch.manual_seed(42)
        output_size = (2, 16)  # 1:8 aspect ratio
        kernel_size = (2, 3)

        def fold_fn(x):
            return torch.nn.functional.fold(x, output_size, kernel_size)

        compiled_fn = torch.compile(fold_fn, backend="neuron")

        L = (output_size[0] - kernel_size[0] + 1) * (output_size[1] - kernel_size[1] + 1)
        C_times_K = kernel_size[0] * kernel_size[1]

        x_cpu = torch.randn(1, C_times_K, L, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = fold_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 2e-2 if dtype == torch.bfloat16 else rtol
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 2e-2 if dtype == torch.bfloat16 else atol
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSelectBackwardDecompositionE2E:
    """End-to-end tests for select_backward decomposition using torch.compile with neuron backend"""

    @pytest.mark.parametrize(
        "shape,dim,index",
        [
            ((4, 5), 0, 0),  # 2D, select first row
            ((4, 5), 0, 2),  # 2D, select middle row
            ((4, 5), 1, 3),  # 2D, select column
            ((3, 4, 5), 0, 1),  # 3D, select along dim 0
            ((3, 4, 5), 1, 2),  # 3D, select along dim 1
            ((3, 4, 5), 2, 0),  # 3D, select along dim 2
            ((2, 3, 4, 5), 0, 0),  # 4D, select along dim 0
            ((2, 3, 4, 5), 2, 1),  # 4D, select along dim 2
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_select_backward_e2e(self, shape, dim, index, dtype):
        """Test select_backward via torch.compile with neuron backend"""
        torch.manual_seed(42)

        # Define a function that uses select and computes loss
        def select_and_sum(x):
            selected = torch.select(x, dim, index)
            return selected.sum()

        # Compile with neuron backend
        compiled_fn = torch.compile(select_and_sum, backend="neuron")

        # Create input tensors
        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        # Forward pass on CPU
        loss_cpu = select_and_sum(x_cpu)
        # Forward pass on Neuron (compiled)
        loss_neuron = compiled_fn(x_neuron)

        # Backward pass
        loss_cpu.backward()
        loss_neuron.backward()

        # Compare gradients
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])
    def test_select_backward_negative_dim(self, dim):
        """Test select_backward with negative dimensions"""
        torch.manual_seed(42)
        shape = (3, 4, 5)
        index = 1

        def select_and_sum(x):
            selected = torch.select(x, dim, index)
            return selected.sum()

        compiled_fn = torch.compile(select_and_sum, backend="neuron")

        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = select_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_select_backward_with_computation(self):
        """Test select_backward with additional computation after select"""
        torch.manual_seed(42)

        def compute_with_select(x):
            # Select a row and do some computation
            selected = torch.select(x, 0, 1)
            result = selected * 2 + 1
            return result.sum()

        compiled_fn = torch.compile(compute_with_select, backend="neuron")

        x_cpu = torch.randn(4, 5, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = compute_with_select(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_select_backward_multiple_selects(self):
        """Test select_backward with multiple select operations"""
        torch.manual_seed(42)

        def multiple_selects(x):
            s1 = torch.select(x, 0, 0)
            s2 = torch.select(x, 0, 2)
            return (s1 + s2).sum()

        compiled_fn = torch.compile(multiple_selects, backend="neuron")

        x_cpu = torch.randn(4, 5, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = multiple_selects(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSliceBackwardDecompositionE2E:
    """End-to-end tests for slice_backward decomposition using torch.compile with neuron backend"""

    @pytest.mark.parametrize(
        "shape,slices",
        [
            ((8,), (slice(2, 6),)),  # 1D slice
            ((4, 5), (slice(1, 3),)),  # 2D, slice first dim
            ((4, 5), (slice(None), slice(1, 4))),  # 2D, slice second dim
            ((4, 5), (slice(0, 2), slice(1, 4))),  # 2D, slice both dims
            ((3, 4, 5), (slice(1, 3),)),  # 3D, slice first dim
            ((3, 4, 5), (slice(None), slice(1, 3), slice(2, 4))),  # 3D, multiple slices
            ((2, 3, 4, 5), (slice(None), slice(1, 2), slice(None), slice(2, 4))),  # 4D
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_slice_backward_e2e(self, shape, slices, dtype):
        """Test slice_backward via torch.compile with neuron backend"""
        torch.manual_seed(42)

        # Define a function that uses slicing and computes loss
        def slice_and_sum(x):
            sliced = x[slices]
            return sliced.sum()

        # Compile with neuron backend
        compiled_fn = torch.compile(slice_and_sum, backend="neuron")

        # Create input tensors
        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        # Forward pass on CPU
        loss_cpu = slice_and_sum(x_cpu)
        # Forward pass on Neuron (compiled)
        loss_neuron = compiled_fn(x_neuron)

        # Backward pass
        loss_cpu.backward()
        loss_neuron.backward()

        # Compare gradients
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "start,end,step",
        [
            (0, 4, 1),  # Basic slice
            (1, 5, 1),  # Offset start
            (0, 6, 2),  # Strided slice
            (1, 7, 2),  # Strided with offset
            (None, 4, 1),  # None start
            (2, None, 1),  # None end
            (None, None, 2),  # None start and end with stride
        ],
    )
    def test_slice_backward_step_variations(self, start, end, step):
        """Test slice_backward with various step/stride values"""
        torch.manual_seed(42)
        shape = (8, 6)

        def slice_and_sum(x):
            sliced = x[start:end:step, :]
            return sliced.sum()

        compiled_fn = torch.compile(slice_and_sum, backend="neuron")

        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = slice_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_negative_indices(self):
        """Test slice_backward with negative indices"""
        torch.manual_seed(42)

        def slice_and_sum(x):
            # Use negative indices
            sliced = x[-3:, :-1]
            return sliced.sum()

        compiled_fn = torch.compile(slice_and_sum, backend="neuron")

        x_cpu = torch.randn(5, 4, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = slice_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_with_computation(self):
        """Test slice_backward with additional computation after slice"""
        torch.manual_seed(42)

        def compute_with_slice(x):
            sliced = x[1:3, 2:5]
            result = sliced * 3 - 2
            return result.sum()

        compiled_fn = torch.compile(compute_with_slice, backend="neuron")

        x_cpu = torch.randn(4, 6, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = compute_with_slice(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_multiple_slices(self):
        """Test slice_backward with multiple slice operations"""
        torch.manual_seed(42)

        def multiple_slices(x):
            s1 = x[:2, :3]
            s2 = x[2:, 3:]
            return s1.sum() + s2.sum()

        compiled_fn = torch.compile(multiple_slices, backend="neuron")

        x_cpu = torch.randn(4, 6, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = multiple_slices(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_overlapping_slices(self):
        """Test slice_backward with overlapping slice regions"""
        torch.manual_seed(42)

        def overlapping_slices(x):
            s1 = x[0:3, :]
            s2 = x[1:4, :]
            return s1.sum() + s2.sum()

        compiled_fn = torch.compile(overlapping_slices, backend="neuron")

        x_cpu = torch.randn(5, 4, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = overlapping_slices(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

    def test_slice_backward_combined_with_select(self):
        """Test slice_backward combined with select operation"""
        torch.manual_seed(42)

        def slice_and_select(x):
            sliced = x[1:3, :]
            selected = torch.select(sliced, 0, 0)
            return selected.sum()

        compiled_fn = torch.compile(slice_and_select, backend="neuron")

        x_cpu = torch.randn(4, 5, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = slice_and_select(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestEluDecompositionE2E:
    """End-to-end tests for elu decomposition using torch.compile with neuron backend"""

    @pytest.mark.parametrize(
        "shape",
        [
            (8,),
            (4, 8),
            (2, 4, 8),
            (2, 3, 4, 5),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_elu_e2e(self, shape, dtype):
        """Test elu via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def elu_fn(x):
            return torch.nn.functional.elu(x)

        compiled_fn = torch.compile(elu_fn, backend="neuron")

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = elu_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_elu_alpha(self, alpha):
        """Test elu with different alpha values"""
        torch.manual_seed(42)

        def elu_fn(x):
            return torch.nn.functional.elu(x, alpha=alpha)

        compiled_fn = torch.compile(elu_fn, backend="neuron")

        x_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = elu_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_elu_backward(self):
        """Test elu backward pass"""
        torch.manual_seed(42)

        def elu_and_sum(x):
            return torch.nn.functional.elu(x).sum()

        compiled_fn = torch.compile(elu_and_sum, backend="neuron")

        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = elu_and_sum(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestArgsortDecompositionE2E:
    """Tests for argsort decomposition"""

    def test_argsort_basic_accuracy(self):
        """Test argsort decomposition accuracy"""
        input_tensor = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
        result = decompositions.argsort_decomposition(input_tensor)
        expected = torch.argsort(input_tensor)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("descending", [False, True])
    def test_argsort_dim_descending(self, dim, descending):
        """Test argsort with different dimensions and descending parameter"""
        input_tensor = torch.randn(4, 5)
        result = decompositions.argsort_decomposition(input_tensor, dim=dim, descending=descending)
        expected = torch.argsort(input_tensor, dim=dim, descending=descending)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    )
    def test_argsort_dtypes(self, dtype):
        """Test argsort with various dtypes"""
        torch.manual_seed(42)
        if dtype in [torch.int32, torch.int64]:
            input_tensor = torch.randint(-100, 100, (10,), dtype=dtype)
        else:
            input_tensor = torch.randn(10, dtype=dtype)
        result = decompositions.argsort_decomposition(input_tensor)
        expected = torch.argsort(input_tensor)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("stable", [False, True])
    def test_argsort_stable(self, stable):
        """Test argsort with stable parameter"""
        input_tensor = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        result = decompositions.argsort_decomposition(input_tensor, stable=stable)
        expected = torch.argsort(input_tensor, stable=stable)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (8,),
            (4, 8),
            (2, 4, 8),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_argsort_e2e(self, shape, dtype):
        """Test argsort accuracy via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def argsort_fn(x):
            return torch.argsort(x)

        compiled_fn = torch.compile(argsort_fn, backend="neuron")

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = argsort_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("descending", [False, True])
    def test_argsort_dim_descending_e2e(self, dim, descending):
        """Test argsort with different dimensions and descending parameter via torch.compile"""
        torch.manual_seed(42)

        def argsort_fn(x):
            return torch.argsort(x, dim=dim, descending=descending)

        compiled_fn = torch.compile(argsort_fn, backend="neuron")

        x_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = argsort_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestHistcDecompositionE2E:
    """Tests for histc decomposition"""

    def test_histc_basic_accuracy(self):
        """Test histc decomposition accuracy"""
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = decompositions.histc_decomposition(input_tensor, bins=5, min=0, max=6)
        expected = torch.histc(input_tensor, bins=5, min=0, max=6)
        torch.testing.assert_close(result, expected)

    def test_histc_auto_range(self):
        """Test histc with automatic range detection (min=max triggers auto-detect)"""
        torch.manual_seed(42)
        input_tensor = torch.randn(50)
        result = decompositions.histc_decomposition(input_tensor, bins=10, min=0, max=0)
        expected = torch.histc(input_tensor, bins=10, min=0, max=0)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_histc_edge_and_out_of_range(self):
        """Test histc with edge values and out-of-range values"""
        input_tensor = torch.tensor([-1.0, 0.0, 2.5, 5.0, 6.0])
        result = decompositions.histc_decomposition(input_tensor, bins=5, min=0, max=5)
        expected = torch.histc(input_tensor, bins=5, min=0, max=5)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_histc_identical_values(self):
        """Test histc when all input values are identical (division by zero case)"""
        input_tensor = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
        result = decompositions.histc_decomposition(input_tensor, bins=5, min=0, max=0)
        expected = torch.histc(input_tensor, bins=5, min=0, max=0)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_histc_dtypes(self, dtype):
        """Test histc with various dtypes"""
        torch.manual_seed(42)
        input_tensor = torch.randn(50, dtype=dtype)
        result = decompositions.histc_decomposition(input_tensor, bins=10, min=-3, max=3)
        expected = torch.histc(input_tensor, bins=10, min=-3, max=3)
        # histc returns the same dtype as the input
        assert result.dtype == dtype
        assert expected.dtype == dtype
        rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("bins", [1, 5, 10, 20, 100])
    def test_histc_various_bins(self, bins):
        """Test histc with various bin counts"""
        torch.manual_seed(42)
        input_tensor = torch.randn(100)
        result = decompositions.histc_decomposition(input_tensor, bins=bins, min=-3, max=3)
        expected = torch.histc(input_tensor, bins=bins, min=-3, max=3)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_histc_empty_tensor(self):
        """Test histc with empty tensor"""
        input_tensor = torch.tensor([])
        result = decompositions.histc_decomposition(input_tensor, bins=5, min=0, max=10)
        expected = torch.histc(input_tensor, bins=5, min=0, max=10)
        torch.testing.assert_close(result, expected)

    def test_histc_single_value(self):
        """Test histc with single value"""
        input_tensor = torch.tensor([5.0])
        result = decompositions.histc_decomposition(input_tensor, bins=10, min=0, max=10)
        expected = torch.histc(input_tensor, bins=10, min=0, max=10)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (0, 10),
            (-5, 5),
            (-10, 0),
            (100, 200),
        ],
    )
    def test_histc_various_ranges(self, min_val, max_val):
        """Test histc with various min/max ranges"""
        torch.manual_seed(42)
        input_tensor = torch.rand(100) * (max_val - min_val) + min_val
        result = decompositions.histc_decomposition(input_tensor, bins=10, min=min_val, max=max_val)
        expected = torch.histc(input_tensor, bins=10, min=min_val, max=max_val)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),
            (4, 8),
            (2, 4, 8),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_histc_e2e(self, shape, dtype):
        """Test histc accuracy via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def histc_fn(x):
            return torch.histc(x, bins=10)

        compiled_fn = torch.compile(histc_fn, backend="neuron")

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = histc_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_histc_bins(self, bins):
        """Test histc with different number of bins via torch.compile"""
        torch.manual_seed(42)

        def histc_fn(x):
            return torch.histc(x, bins=bins)

        compiled_fn = torch.compile(histc_fn, backend="neuron")

        x_cpu = torch.randn(100)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = histc_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        assert result_cpu.shape == (bins,)
        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (0, 10),
            (-5, 5),
        ],
    )
    def test_histc_min_max(self, min_val, max_val):
        """Test histc with different min and max values via torch.compile"""
        torch.manual_seed(42)

        def histc_fn(x):
            return torch.histc(x, bins=10, min=min_val, max=max_val)

        compiled_fn = torch.compile(histc_fn, backend="neuron")

        x_cpu = torch.rand(100) * (max_val - min_val) + min_val
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = histc_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_histc_auto_range_e2e(self):
        """Test histc with automatic min/max detection via torch.compile"""
        torch.manual_seed(42)

        def histc_fn(x):
            return torch.histc(x, bins=10, min=0, max=0)

        compiled_fn = torch.compile(histc_fn, backend="neuron")

        x_cpu = torch.randn(50)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = histc_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)


class TestIndexPutNegativeIndicesE2E:
    """End-to-end tests for index_put negative indices handling"""

    def _compare_with_cpu(self, fn, *args):
        """Helper function to compare Neuron result with CPU result"""
        # CPU run
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        cpu_result = fn(*cpu_args)

        # neuron run
        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_args = [a.to("neuron") if isinstance(a, torch.Tensor) else a for a in args]
        neuron_result = compiled_fn(*neuron_args)

        torch.testing.assert_close(neuron_result.cpu(), cpu_result)
        return neuron_result

    @pytest.mark.parametrize(
        "idx_list,description",
        [
            ([-1], "single_negative"),
            ([-1, -2, -3], "multiple_negative"),
            ([0, -1, 5, -2], "mixed_positive_negative"),
            ([-1, -2, -3, -4, -5], "all_negative"),
            ([0, 5, 9], "all_positive"),
        ],
    )
    def test_index_patterns(self, idx_list, description):
        """Test various index patterns"""

        def fn(x, idx, val):
            return x.index_put((idx,), val)

        x = torch.zeros(10, 4)
        idx = torch.tensor(idx_list)
        val = torch.ones(len(idx_list), 4)
        self._compare_with_cpu(fn, x, idx, val)

    @pytest.mark.parametrize("shape", [(10,), (10, 4), (5, 4, 3), (2, 3, 4, 5)])
    def test_negative_indices_various_shapes(self, shape):
        """Test negative indices with various tensor shapes"""

        def fn(x, idx, val):
            return x.index_put((idx,), val)

        x = torch.zeros(shape)
        idx = torch.tensor([-1, -2])
        val_shape = (2,) + shape[1:]
        val = torch.ones(val_shape)

        self._compare_with_cpu(fn, x, idx, val)

    @pytest.mark.parametrize(
        "x_init,idx_list,val_list,check_idx,expected_val",
        [
            (torch.ones(10, 4), [-1, -1, -2], None, None, None),
            (torch.zeros(5), [-1, -1, -1], [1.0, 2.0, 3.0], -1, 6.0),
        ],
    )
    def test_accumulate(self, x_init, idx_list, val_list, check_idx, expected_val):
        """Test negative indices with accumulate=True"""

        def fn(x, idx, val):
            return x.index_put((idx,), val, accumulate=True)

        x = x_init.clone()
        idx = torch.tensor(idx_list)
        val = torch.tensor(val_list) if val_list else torch.ones(len(idx_list), *x.shape[1:])

        result = self._compare_with_cpu(fn, x, idx, val)

        if check_idx is not None:
            assert result.cpu()[check_idx].item() == expected_val

    def test_negative_indices_multi_dim(self):
        """Test negative indices in multi-dimensional indexing"""

        def fn(x, idx1, idx2, val):
            return x.index_put((idx1, idx2), val)

        x = torch.zeros(5, 6)
        idx1 = torch.tensor([-1, -2])
        idx2 = torch.tensor([-1, -2])
        val = torch.tensor([1.0, 2.0])

        self._compare_with_cpu(fn, x, idx1, idx2, val)

    @pytest.mark.parametrize(
        "x_shape,idx_list,val,check_idx,expected_val",
        [
            ((5,), [-5], [1.0], 0, 1.0),  # boundary (-size)
            ((10, 4), [-1, -2, -3], 1.0, None, None),  # scalar broadcasts
            ((10, 4), [], None, None, None),  # empty indices
            ((1,), [-1], [5.0], 0, 5.0),  # single element tensor
        ],
    )
    def test_edge_cases(self, x_shape, idx_list, val, check_idx, expected_val):
        """Test edge cases: boundary, scalar, empty, single element"""

        def fn(x, idx, val):
            return x.index_put((idx,), val)

        x = torch.zeros(x_shape)
        idx = torch.tensor(idx_list, dtype=torch.long)

        if val is None:
            val = torch.ones((0,) + x_shape[1:])
        elif isinstance(val, float):
            val = torch.tensor(val)
        else:
            val = torch.tensor(val)

        result = self._compare_with_cpu(fn, x, idx, val)

        if check_idx is not None:
            assert result.cpu()[check_idx].item() == expected_val

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.bfloat16])
    def test_negative_indices_various_dtypes(self, dtype):
        """Test negative indices with various data types"""

        def fn(x, idx, val):
            return x.index_put((idx,), val)

        x = torch.zeros(10, dtype=dtype)
        idx = torch.tensor([-1, -2])
        val = torch.ones(2, dtype=dtype)

        self._compare_with_cpu(fn, x, idx, val)

    def test_negative_indices_inplace(self):
        """Test index_put_ (inplace) with negative indices"""
        x_cpu = torch.zeros(10, 4)
        x_neuron = x_cpu.clone().to("neuron")

        idx_cpu = torch.tensor([-1, -2])
        idx_neuron = idx_cpu.clone().to("neuron")

        val_cpu = torch.ones(2, 4)
        val_neuron = val_cpu.clone().to("neuron")

        x_cpu.index_put_((idx_cpu,), val_cpu)
        x_neuron.index_put_((idx_neuron,), val_neuron)

        torch.testing.assert_close(x_neuron.cpu(), x_cpu)
