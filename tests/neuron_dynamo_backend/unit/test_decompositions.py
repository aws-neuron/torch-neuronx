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
        assert torch.ops.aten.native_batch_norm.default in decompositions.neuron_decompositions
        assert (
            torch.ops.aten._native_batch_norm_legit.default in decompositions.neuron_decompositions
        )
        assert (
            torch.ops.aten._native_batch_norm_legit_functional.default
            in decompositions.neuron_decompositions
        )
        assert (
            torch.ops.aten._native_batch_norm_legit_no_training.default
            in decompositions.neuron_decompositions
        )

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

    def test_bmm_dtype_decomposition_registered(self):
        """bmm.dtype decomp is registered."""
        assert torch.ops.aten.bmm.dtype in decompositions.neuron_decompositions
        assert callable(decompositions.neuron_decompositions[torch.ops.aten.bmm.dtype])

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.aminmax.default,
            torch.ops.aten.cumprod.default,
            torch.ops.aten.diag_embed.default,
            torch.ops.aten.hardtanh_backward.default,
            torch.ops.aten.binary_cross_entropy_backward.default,
            torch.ops.aten.nll_loss2d_forward.default,
            torch.ops.aten.max_unpool2d.default,
            torch.ops.aten.unfold_copy.default,
            torch.ops.aten.split_with_sizes_copy.default,
            torch.ops.aten._adaptive_avg_pool2d.default,
            # Upsample forward ops covered by PyTorch builtins.
            # Registering the functional packet registers .default and .vec.
            # NOTE: the linear family (bilinear2d/linear1d/trilinear3d) is NOT
            # here — those builtins are wrong on Neuron and are replaced by
            # custom decompositions (see TestUpsampleLinearFamilyDecomposition).
            torch.ops.aten.upsample_nearest2d.default,
            torch.ops.aten.upsample_nearest2d.vec,
            torch.ops.aten.upsample_bicubic2d.default,
        ],
    )
    def test_pytorch_builtin_decompositions_registered(self, op):
        """The PyTorch built-in decomps pulled in via get_decompositions([...])
        are present in the Neuron table."""
        assert op in decompositions.neuron_decompositions

    def test_bmm_dtype_out_fake_registered(self):
        """bmm.dtype_out has a fake/meta impl so Dynamo's get_fake_value can
        trace it instead of falling back to a real (device) kernel. Exercising
        the op on meta tensors must not raise NotImplementedError."""
        self_t = torch.empty(4, 8, 16, device="meta")
        mat2 = torch.empty(4, 16, 8, device="meta")
        out = torch.empty(4, 8, 8, device="meta")
        res = torch.ops.aten.bmm.dtype_out(self_t, mat2, torch.float32, out=out)
        assert tuple(res.shape) == (4, 8, 8)
        assert res.device.type == "meta"


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

    @pytest.mark.parametrize(
        "input_shape,weight_shape,grad_output_shape",
        [
            ((3, 4), (5, 4), (3, 5)),
            ((2, 3, 4), (5, 4), (2, 3, 5)),
            ((1, 2, 3, 4), (5, 4), (1, 2, 3, 5)),
        ],
    )
    def test_linear_backward_grad_weight_uses_2d_mm(
        self, input_shape, weight_shape, grad_output_shape
    ):
        """grad_weight must lower to a 2D mm on flattened operands.

        Regression guard for the einsum-based formulation, which lowered to a
        4D-transpose-chain in StableHLO which is slower than 2D dot_general
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(0)
        input_ = torch.randn(input_shape)
        weight_ = torch.randn(weight_shape)
        grad_output_ = torch.randn(grad_output_shape)
        # output_mask = [False, True, False] isolates the grad_weight branch
        output_mask = [False, True, False]

        gm = make_fx(decompositions.linear_backward_decomposition)(
            input_, grad_output_, weight_, output_mask
        )
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]

        assert (
            torch.ops.aten.einsum.default not in targets
        ), f"linear_backward grad_weight regressed to einsum: {targets}"
        mm_targets = {torch.ops.aten.mm.default, torch.ops.aten.mm.out}
        assert any(
            t in mm_targets for t in targets
        ), f"linear_backward grad_weight should use aten.mm; got: {targets}"

        mm_nodes = [n for n in gm.graph.nodes if n.target in mm_targets]
        for n in mm_nodes:
            for arg in n.args[:2]:
                if hasattr(arg, "meta") and "val" in arg.meta:
                    assert (
                        arg.meta["val"].ndim == 2
                    ), f"mm operand should be 2D, got shape {arg.meta['val'].shape}"

    @pytest.mark.parametrize(
        "input_shape,weight_shape,grad_output_shape",
        [
            ((3, 4), (5, 4), (3, 5)),  # 2D
            ((2, 3, 4), (5, 4), (2, 3, 5)),  # 3D (FSDP transformer activation shape)
        ],
    )
    def test_linear_backward_mixed_precision(self, input_shape, weight_shape, grad_output_shape):
        """FSDP2 mixed precision: fp32 (master) weight with bf16 grad_output/input.

        PT2.12's bmm/baddbmm meta enforces matching operand dtypes, so a bf16
        grad x fp32 weight matmul fails during compile-mode sharding propagation.
        The decomposition casts the weight to the grad dtype; verify it (a) does
        not raise on the mixed-dtype inputs and (b) matches an eager mixed-precision
        reference. Guards the compile-mode mixed-precision fix.
        """
        torch.manual_seed(1)
        input_ = torch.randn(input_shape, dtype=torch.bfloat16)
        weight_ = torch.randn(weight_shape, dtype=torch.float32)  # fp32 master weight
        grad_output_ = torch.randn(grad_output_shape, dtype=torch.bfloat16)
        output_mask = [True, True, True]

        result_grad_input, result_grad_weight, result_grad_bias = (
            decompositions.linear_backward_decomposition(input_, grad_output_, weight_, output_mask)
        )

        # Reference: eager backward with the weight cast to the compute (bf16) dtype,
        # matching what the decomposition (and eager mixed precision) does.
        ref_grad_input, ref_grad_weight, ref_grad_bias = self._compute_reference_gradients(
            input_.to(torch.float32),
            weight_.to(torch.bfloat16).to(torch.float32),
            grad_output_.to(torch.float32),
            output_mask,
        )

        assert result_grad_input.shape == input_shape
        assert result_grad_weight.shape == weight_shape
        assert result_grad_bias.shape == (grad_output_shape[-1],)
        # grad_input follows grad_output/weight-compute dtype (bf16); others follow inputs
        torch.testing.assert_close(
            result_grad_input.to(torch.float32), ref_grad_input, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            result_grad_weight.to(torch.float32), ref_grad_weight, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            result_grad_bias.to(torch.float32), ref_grad_bias, rtol=1e-2, atol=1e-2
        )


class TestBmmDtypeDecomposition:
    """Test the bmm.dtype decomposition (cast + plain bmm)."""

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 8, 16), (4, 16, 8)),
            ((2, 3, 5), (2, 5, 7)),
            ((1, 1, 1), (1, 1, 1)),
        ],
    )
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
    def test_bmm_dtype_decomp_matches_reference(self, shape_a, shape_b, out_dtype):
        torch.manual_seed(0)
        a = torch.randn(shape_a)
        b = torch.randn(shape_b)
        result = decompositions.bmm_dtype(a, b, out_dtype)
        expected = torch.bmm(a.to(out_dtype), b.to(out_dtype))
        assert result.dtype == out_dtype
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBmmDtypeOutDecompositionE2E:
    """Test bmm.dtype_out compiles end-to-end on Neuron via the fake impl +
    bmm.dtype decomposition."""

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 8, 16), (4, 16, 8)),
            ((2, 32, 64), (2, 64, 16)),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_dtype_out_e2e(self, shape_a, shape_b, dtype):
        torch.manual_seed(42)

        def bmm_dtype_out_fn(a, b, out):
            return torch.ops.aten.bmm.dtype_out(a, b, torch.float32, out=out)

        compiled_fn = torch.compile(bmm_dtype_out_fn, backend="neuron", fullgraph=True)

        a_cpu = torch.randn(shape_a, dtype=dtype)
        b_cpu = torch.randn(shape_b, dtype=dtype)
        out_shape = (shape_a[0], shape_a[1], shape_b[2])

        # bmm.dtype_out has no CPU kernel — reference via plain bmm. out_dtype is
        # float32 so the result is float32 regardless of input dtype.
        expected = torch.bmm(a_cpu.float(), b_cpu.float())

        a_n = a_cpu.detach().clone().to("neuron")
        b_n = b_cpu.detach().clone().to("neuron")
        out_n = torch.empty(out_shape, device="neuron")
        result = compiled_fn(a_n, b_n, out_n)

        rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-3)
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)


class TestClampPromotionDecomposition:
    """Test the clamp.default decomposition (maximum/minimum + type promotion)."""

    @pytest.mark.parametrize(
        "in_dtype,min,max",
        [
            (torch.float32, None, 1.0),
            (torch.float32, -1.0, 1.0),
            (torch.int64, None, 1.0),  # float bound promotes int -> float
            (torch.int64, None, 1),  # int bound keeps int
            (torch.int32, None, 2.0),
            (torch.bfloat16, None, 1.0),  # float input stays bf16 (no upcast)
            (torch.float16, -2.0, 2.0),
        ],
    )
    def test_clamp_decomp_matches_reference(self, in_dtype, min, max):
        torch.manual_seed(0)
        if in_dtype in (torch.int64, torch.int32):
            x = torch.randint(-5, 5, (4, 4), dtype=in_dtype)
        else:
            x = (torch.randn(4, 4) * 4).to(in_dtype)
        result = decompositions.clamp_decomp(x, min, max)
        expected = torch.clamp(x, min=min, max=max)
        assert result.dtype == expected.dtype, f"{result.dtype} != {expected.dtype}"
        torch.testing.assert_close(result.float(), expected.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestClampPromotionDecompositionE2E:
    """Test clamp.default compiles end-to-end on Neuron via the decomposition,
    incl. the int->float promotion case that raw torch-mlir clamp rejects."""

    @pytest.mark.parametrize(
        "in_dtype,min,max",
        [
            (torch.float32, -1.0, 1.0),
            (torch.int64, None, 1.0),
            (torch.int32, None, 2.0),
            (torch.bfloat16, None, 1.0),
        ],
    )
    def test_clamp_e2e(self, in_dtype, min, max):
        torch.manual_seed(42)
        if in_dtype in (torch.int64, torch.int32):
            x_cpu = torch.randint(-5, 5, (4, 8), dtype=in_dtype)
        else:
            x_cpu = (torch.randn(4, 8) * 4).to(in_dtype)

        def fn(x):
            return torch.clamp(x, min=min, max=max)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled_fn(x_cpu.to("neuron"))
        expected = fn(x_cpu)
        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu().float(), expected.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAminmaxDecompositionE2E:
    """E2E tests for aminmax decomposition"""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((4, 8), None),
            ((4, 8), 0),
            ((3, 4, 5), 2),
            pytest.param(
                (),
                None,
                marks=pytest.mark.xfail(
                    reason="pre-existing PT 2.11->2.12 bump failure; tracked for fix"
                ),
            ),
            pytest.param(
                (),
                0,
                marks=pytest.mark.xfail(
                    reason="pre-existing PT 2.11->2.12 bump failure; tracked for fix"
                ),
            ),
        ],
    )
    def test_aminmax_e2e(self, shape, dim):
        torch.manual_seed(42)

        def fn(x):
            return torch.aminmax(x, dim=dim) if dim is not None else torch.aminmax(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        amin_n, amax_n = compiled_fn(x_cpu.to("neuron"))
        amin_c, amax_c = fn(x_cpu)
        torch.testing.assert_close(amin_n.cpu(), amin_c, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(amax_n.cpu(), amax_c, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestCumprodDecompositionE2E:
    """E2E tests for cumprod decomposition"""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((6,), 0),
            ((4, 8), 1),
            ((3, 4, 5), -1),
            pytest.param(
                (),
                0,
                marks=pytest.mark.xfail(
                    reason="pre-existing PT 2.11->2.12 bump failure; tracked for fix"
                ),
            ),
        ],
    )
    def test_cumprod_e2e(self, shape, dim):
        torch.manual_seed(42)

        def fn(x):
            return torch.cumprod(x, dim=dim)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestDiagEmbedDecompositionE2E:
    """E2E tests for diag_embed decomposition"""

    @pytest.mark.parametrize("shape,offset", [((4,), 0), ((3, 5), 0), ((3, 5), 1), ((2, 3, 4), -1)])
    def test_diag_embed_e2e(self, shape, offset):
        torch.manual_seed(42)

        def fn(x):
            return torch.diag_embed(x, offset=offset)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestHardtanhBackwardDecompositionE2E:
    """E2E tests for hardtanh_backward decomposition (via hardtanh fwd+bwd)"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_hardtanh_backward_e2e(self, shape, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.hardtanh(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        # hardtanh_backward is a 0/1 gradient mask — exact, so the dtype-aware
        # defaults (1e-2 bf16 / 1.3e-6 fp32) are plenty.
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBinaryCrossEntropyBackwardDecompositionE2E:
    """E2E tests for binary_cross_entropy_backward decomposition (via BCE fwd+bwd)"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bce_backward_e2e(self, shape, dtype):
        torch.manual_seed(42)

        def fn(x, target):
            return torch.nn.functional.binary_cross_entropy(x, target)

        compiled_fn = torch.compile(fn, backend="neuron")
        # Keep inputs away from 0/1 so the BCE grad (1/(x(1-x))-scale) is well
        # conditioned and the bf16 comparison isn't dominated by a few huge grads.
        x_cpu = (torch.rand(shape) * 0.6 + 0.2).to(dtype).requires_grad_(True)
        target = (torch.rand(shape) * 0.6 + 0.2).to(dtype)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        target_neuron = target.to("neuron")

        fn(x_cpu, target).backward()
        compiled_fn(x_neuron, target_neuron).backward()
        # bf16 BCE backward accumulates more error than the fp32 default allows.
        rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNllLoss2dForwardDecompositionE2E:
    """E2E tests for nll_loss2d_forward decomposition (2D spatial NLL loss)"""

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_nll_loss2d_e2e(self, reduction):
        torch.manual_seed(42)
        N, C, H, W = 2, 4, 5, 5

        def fn(logits, target):
            return torch.nn.functional.nll_loss(logits, target, reduction=reduction)

        compiled_fn = torch.compile(fn, backend="neuron")
        logits_cpu = torch.log_softmax(torch.randn(N, C, H, W), dim=1)
        target_cpu = torch.randint(0, C, (N, H, W))
        result = compiled_fn(logits_cpu.to("neuron"), target_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(logits_cpu, target_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestMaxUnpool2dDecompositionE2E:
    """E2E tests for max_unpool2d decomposition"""

    def test_max_unpool2d_e2e(self):
        torch.manual_seed(42)
        x = torch.randn(1, 2, 4, 4)
        pooled, indices = torch.nn.functional.max_pool2d(x, 2, return_indices=True)

        def fn(p, idx):
            return torch.nn.functional.max_unpool2d(p, idx, kernel_size=2)

        compiled_fn = torch.compile(fn, backend="neuron")
        result = compiled_fn(pooled.to("neuron"), indices.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(pooled, indices), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestUnfoldCopyDecompositionE2E:
    """E2E tests for unfold_copy decomposition"""

    @pytest.mark.parametrize(
        "shape,dim,size,step", [((8,), 0, 3, 1), ((4, 8), 1, 4, 2), ((), 0, 1, 1)]
    )
    def test_unfold_copy_e2e(self, shape, dim, size, step):
        torch.manual_seed(42)

        def fn(x):
            return torch.ops.aten.unfold_copy(x, dim, size, step)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSplitWithSizesCopyDecompositionE2E:
    """E2E tests for split_with_sizes_copy decomposition"""

    def test_split_with_sizes_copy_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            parts = torch.ops.aten.split_with_sizes_copy(x, [2, 3, 3], dim=1)
            return parts[0].sum() + parts[1].sum() + parts[2].sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAdaptiveAvgPool2dDecompositionE2E:
    """E2E tests for _adaptive_avg_pool2d decomposition"""

    @pytest.mark.parametrize("output_size", [(1, 1), (2, 2), (3, 3)])
    def test_adaptive_avg_pool2d_e2e(self, output_size):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, output_size)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(2, 3, 6, 6)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


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


class TestSliceBackwardDecomposition:
    """Direct tests for the slice_backward decomposition function.

    The fast path replaces zeros + slice_scatter (lowers to stablehlo.scatter)
    with constant_pad_nd (lowers to stablehlo.pad). RoPE backward hits this
    path 4x per layer.
    """

    @pytest.mark.parametrize(
        "input_sizes,dim,start,end",
        [
            ([1, 8, 4096, 128], 3, 0, 64),  # RoPE backward shape: pad upper half
            ([1, 8, 4096, 128], 3, 64, 128),  # RoPE backward shape: pad lower half
            ([4, 6], 0, 1, 3),  # 2D, slice first dim
            ([4, 6], 1, 2, 5),  # 2D, slice last dim
            ([8], 0, 2, 6),  # 1D
            ([3, 4, 5], 1, 1, 3),  # 3D, middle dim
        ],
    )
    def test_fast_path_matches_pytorch(self, input_sizes, dim, start, end):
        """Fast path output must match PyTorch's slice_backward for step=1."""
        torch.manual_seed(42)
        sliced_shape = list(input_sizes)
        sliced_shape[dim] = end - start
        grad_output = torch.randn(sliced_shape)

        ours = decompositions.slice_backward_decomposition(
            grad_output, input_sizes, dim, start, end, 1
        )
        # PyTorch's reference C++ implementation
        reference = torch.ops.aten.slice_backward(grad_output, input_sizes, dim, start, end, 1)

        assert ours is not NotImplemented
        torch.testing.assert_close(ours, reference)

    def test_fast_path_handles_negative_indices(self):
        """Negative start follows Python slice semantics: -3 -> 5+(-3)=2."""
        torch.manual_seed(42)
        input_sizes = [5, 4]
        # x[-3:, :] selects rows 2..5 along dim 0; grad has shape (3, 4)
        grad_output = torch.randn(3, 4)
        ours = decompositions.slice_backward_decomposition(grad_output, input_sizes, 0, -3, 5, 1)
        reference = torch.ops.aten.slice_backward(grad_output, input_sizes, 0, -3, 5, 1)
        assert ours is not NotImplemented
        torch.testing.assert_close(ours, reference)

    def test_fallback_for_non_unit_step(self):
        """Strided slices use zeros + slice_scatter and must match PyTorch's reference."""
        torch.manual_seed(42)
        # x[0:6:2, :] selects rows 0, 2, 4 -> grad shape (3, 6)
        grad_output = torch.randn(3, 6)
        ours = decompositions.slice_backward_decomposition(grad_output, [6, 6], 0, 0, 6, 2)
        reference = torch.ops.aten.slice_backward(grad_output, [6, 6], 0, 0, 6, 2)
        torch.testing.assert_close(ours, reference)

    def test_not_in_default_decomposition_table(self):
        """slice_backward must NOT use PyTorch's default — that produces stablehlo.scatter."""
        # The base table delegates to PyTorch defaults for any op listed at module load time.
        # Our custom decomposition must be present and registered.
        assert torch.ops.aten.slice_backward.default in decompositions.neuron_decompositions
        assert (
            decompositions.neuron_decompositions[torch.ops.aten.slice_backward.default]
            is decompositions.slice_backward_decomposition
        )


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


class TestAvgPool3dBackwardDecomposition:
    """Tests for avg_pool3d_backward decomposition.

    Matches aten's avg_pool3d_backward for the count_include_pad=True / no-pad
    cases the decomposition supports.
    """

    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 4, 4, 4), 2, 2, 0),  # non-overlapping
            ((2, 3, 6, 6, 6), 3, 1, 0),  # overlapping windows
            ((1, 1, 5, 5, 5), 2, 1, 0),  # odd input, unit stride
            ((1, 2, 4, 4, 4), 2, 2, 1),  # with padding
        ],
    )
    def test_matches_aten(self, shape, kernel, stride, padding):
        F = torch.nn.functional
        x = torch.randn(*shape, dtype=torch.float64)
        k, s, p = [kernel] * 3, [stride] * 3, [padding] * 3
        out = F.avg_pool3d(x, k, s, p)
        grad_output = torch.randn_like(out)

        result = decompositions.avg_pool3d_backward_decomposition(
            grad_output, x, k, s, p, False, True, None
        )
        expected = torch.ops.aten.avg_pool3d_backward.default(
            grad_output, x, k, s, p, False, True, None
        )
        torch.testing.assert_close(result, expected)

    def test_divisor_override(self):
        """divisor_override replaces the window-size divisor."""
        F = torch.nn.functional
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64)
        k, s, p = [2, 2, 2], [2, 2, 2], [0, 0, 0]
        out = F.avg_pool3d(x, k, s, p, divisor_override=5)
        grad_output = torch.randn_like(out)

        result = decompositions.avg_pool3d_backward_decomposition(
            grad_output, x, k, s, p, False, True, 5
        )
        expected = torch.ops.aten.avg_pool3d_backward.default(
            grad_output, x, k, s, p, False, True, 5
        )
        torch.testing.assert_close(result, expected)

    def test_registered(self):
        assert torch.ops.aten.avg_pool3d_backward.default in decompositions.neuron_decompositions


class TestMaxPool3dWithIndicesDecomposition:
    """Tests for max_pool3d_with_indices forward + backward decompositions."""

    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 4, 4, 4), 2, 2, 0),  # non-overlapping
            ((2, 3, 6, 6, 6), 3, 1, 0),  # overlapping windows
            ((1, 1, 5, 5, 5), 2, 1, 1),  # with padding
        ],
    )
    def test_forward_matches_aten(self, shape, kernel, stride, padding):
        """Forward returns values and flat (D*H*W) indices matching aten."""
        x = torch.randn(*shape, dtype=torch.float64)
        k, s, p, d = [kernel] * 3, [stride] * 3, [padding] * 3, [1, 1, 1]

        values, indices = decompositions.max_pool3d_with_indices_decomposition(x, k, s, p, d, False)
        exp_values, exp_indices = torch.ops.aten.max_pool3d_with_indices.default(
            x, k, s, p, d, False
        )
        torch.testing.assert_close(values, exp_values)
        assert torch.equal(indices, exp_indices)

    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 4, 4, 4), 2, 2, 0),
            ((2, 3, 6, 6, 6), 3, 1, 0),
        ],
    )
    def test_backward_matches_aten(self, shape, kernel, stride, padding):
        """Backward scatter-adds gradients to the winning positions."""
        x = torch.randn(*shape, dtype=torch.float64)
        k, s, p, d = [kernel] * 3, [stride] * 3, [padding] * 3, [1, 1, 1]
        _, indices = torch.ops.aten.max_pool3d_with_indices.default(x, k, s, p, d, False)
        grad_output = torch.randn(*indices.shape, dtype=torch.float64)

        result = decompositions.max_pool3d_with_indices_backward_decomposition(
            grad_output, x, k, s, p, d, False, indices
        )
        expected = torch.ops.aten.max_pool3d_with_indices_backward.default(
            grad_output, x, k, s, p, d, False, indices
        )
        torch.testing.assert_close(result, expected)

    def test_dilation_falls_through(self):
        """dilation != 1 is unsupported; decomposition returns NotImplemented."""
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64)
        result = decompositions.max_pool3d_with_indices_decomposition(
            x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [2, 2, 2], False
        )
        assert result is NotImplemented

    def test_ceil_mode_falls_through(self):
        """ceil_mode=True is unsupported; decomposition returns NotImplemented."""
        x = torch.randn(1, 2, 5, 5, 5, dtype=torch.float64)
        result = decompositions.max_pool3d_with_indices_decomposition(
            x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], True
        )
        assert result is NotImplemented

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices_backward.default,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestPool3dDecompositionsE2E:
    """End-to-end tests for the 3d pooling decompositions via torch.compile

    The forward pass exercises ``max_pool3d_with_indices`` and ``avg_pool3d``;
    the backward pass fires ``max_pool3d_with_indices_backward`` and
    ``avg_pool3d_backward`` through autograd. Grads are compared against the
    eager CPU reference.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 4, 4, 4), 2, 2, 0),  # non-overlapping
            ((1, 2, 4, 4, 4), 2, 2, 1),  # with padding
        ],
    )
    def test_avg_pool3d_backward_e2e(self, shape, kernel, stride, padding, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.avg_pool3d(x, kernel, stride, padding).sum()

        compiled_fn = torch.compile(fn, backend="neuron")

        # CPU avg_pool3d has no bfloat16 kernel, so the reference is always fp32;
        # the Neuron path runs in the target dtype and is compared against it.
        x_cpu = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron").requires_grad_(True)

        loss_cpu = fn(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(loss_neuron.cpu().float(), loss_cpu, rtol=tol, atol=tol)
        torch.testing.assert_close(x_neuron.grad.cpu().float(), x_cpu.grad, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 4, 4, 4), 2, 2, 0),  # non-overlapping
            ((2, 3, 6, 6, 6), 3, 1, 0),  # overlapping windows
        ],
    )
    def test_max_pool3d_backward_e2e(self, shape, kernel, stride, padding, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.max_pool3d(x, kernel, stride, padding).sum()

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(*shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = fn(x_cpu)
        loss_neuron = compiled_fn(x_neuron)

        loss_cpu.backward()
        loss_neuron.backward()

        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(loss_neuron.cpu(), loss_cpu, rtol=tol, atol=tol)
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=tol, atol=tol)


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

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
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
class TestArgmaxDecompositionE2E:
    """Tests for argmax decomposition"""

    def test_argmax_basic_accuracy(self):
        """Test argmax decomposition accuracy"""
        input_tensor = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
        result = decompositions.argmax_decomposition(input_tensor)
        expected = torch.argmax(input_tensor)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_argmax_dim_keepdim(self, dim, keepdim):
        """Test argmax with different dimensions and keepdim parameter"""
        input_tensor = torch.randn(4, 5)
        result = decompositions.argmax_decomposition(input_tensor, dim=dim, keepdim=keepdim)
        expected = torch.argmax(input_tensor, dim=dim, keepdim=keepdim)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    )
    def test_argmax_dtypes(self, dtype):
        """Test argmax with various dtypes"""
        torch.manual_seed(42)
        if dtype in [torch.int32, torch.int64]:
            input_tensor = torch.randint(-100, 100, (10,), dtype=dtype)
        else:
            input_tensor = torch.randn(10, dtype=dtype)
        result = decompositions.argmax_decomposition(input_tensor)
        expected = torch.argmax(input_tensor)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (8,),
            (4, 8),
            (2, 4, 8),
            (1, 1, 32000),
            (1, 1, 128256),
            (8, 1, 32000),
            (32, 50257),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32, torch.int64])
    def test_argmax_e2e(self, shape, dtype):
        """Test argmax accuracy via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def argmax_fn(x):
            return torch.argmax(x, dim=-1, keepdim=True)

        compiled_fn = torch.compile(argmax_fn, backend="neuron")

        if dtype in (torch.int32, torch.int64):
            x_cpu = torch.randint(-100, 100, shape, dtype=dtype)
        else:
            x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = argmax_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_argmax_dim_keepdim_e2e(self, dim, keepdim):
        """Test argmax with different dimensions and keepdim via torch.compile"""
        torch.manual_seed(42)

        def argmax_fn(x):
            return torch.argmax(x, dim=dim, keepdim=keepdim)

        compiled_fn = torch.compile(argmax_fn, backend="neuron")

        x_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron")

        result_cpu = argmax_fn(x_cpu)
        result_neuron = compiled_fn(x_neuron)

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    def test_argmax_llm_decode_pattern(self):
        """Test the exact pattern used in LLM greedy decode: logits[:, -1, :].argmax(dim=-1)"""
        torch.manual_seed(42)

        def greedy_decode(logits):
            return logits[:, -1, :].argmax(dim=-1, keepdim=True)

        compiled_fn = torch.compile(greedy_decode, backend="neuron")

        logits_cpu = torch.randn(1, 1, 32000, dtype=torch.bfloat16)
        logits_neuron = logits_cpu.detach().clone().to("neuron")

        result_cpu = greedy_decode(logits_cpu)
        result_neuron = compiled_fn(logits_neuron)

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

    # bfloat16 omitted: bf16's ~7-bit mantissa can quantize values near a bin
    # boundary differently between this decomposition's
    # `(input - min) / bin_width` arithmetic and torch.histc's reference,
    # producing off-by-one bin counts that are impossible to align with any
    # tolerance.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_histc_dtypes(self, dtype):
        """Test histc with various dtypes"""
        torch.manual_seed(42)
        input_tensor = torch.randn(50, dtype=dtype)
        result = decompositions.histc_decomposition(input_tensor, bins=10, min=-3, max=3)
        expected = torch.histc(input_tensor, bins=10, min=-3, max=3)
        # histc returns the same dtype as the input
        assert result.dtype == dtype
        assert expected.dtype == dtype
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
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


class TestMseLossBackwardDecomposition:
    """Tests for mse_loss_backward decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.mse_loss_backward.default in decompositions.neuron_decompositions
        ), "aten.mse_loss_backward.default not found in neuron_decompositions"

    @pytest.mark.parametrize("reduction", [0, 1, 2])  # NONE, MEAN, SUM
    def test_mse_loss_backward_reductions(self, reduction):
        """Test mse_loss_backward with all reduction modes"""
        torch.manual_seed(42)
        input_ = torch.randn(4, 5, requires_grad=True)
        target = torch.randn(4, 5)
        output = torch.nn.functional.mse_loss(
            input_, target, reduction=["none", "mean", "sum"][reduction]
        )
        grad_output = torch.ones_like(output)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.mse_loss_backward.default]
        result = decomp_fn(grad_output, input_.detach(), target, reduction)

        output.backward(grad_output)
        torch.testing.assert_close(result, input_.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 5), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_mse_loss_backward_shapes(self, shape):
        """Test mse_loss_backward with various tensor shapes"""
        torch.manual_seed(42)
        input_ = torch.randn(shape, requires_grad=True)
        target = torch.randn(shape)
        output = torch.nn.functional.mse_loss(input_, target, reduction="mean")
        grad_output = torch.ones_like(output)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.mse_loss_backward.default]
        result = decomp_fn(grad_output, input_.detach(), target, 1)  # MEAN=1

        output.backward(grad_output)
        torch.testing.assert_close(result, input_.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_mse_loss_backward_dtypes(self, dtype):
        """Test mse_loss_backward with various dtypes"""
        torch.manual_seed(42)
        input_ = torch.randn(4, 5, dtype=dtype, requires_grad=True)
        target = torch.randn(4, 5, dtype=dtype)
        output = torch.nn.functional.mse_loss(input_, target, reduction="sum")
        grad_output = torch.ones_like(output)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.mse_loss_backward.default]
        result = decomp_fn(grad_output, input_.detach(), target, 2)  # SUM=2

        output.backward(grad_output)
        torch.testing.assert_close(result, input_.grad, rtol=1e-4, atol=1e-4)


class TestNativeGroupNormBackwardDecomposition:
    """Tests for native_group_norm_backward decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.native_group_norm_backward.default
            in decompositions.neuron_decompositions
        ), "aten.native_group_norm_backward.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "n,c,h_x_w,group",
        [
            (2, 4, 9, 2),
            (1, 6, 16, 3),
            (4, 8, 4, 4),
        ],
    )
    def test_group_norm_backward_shapes(self, n, c, h_x_w, group):
        """Test native_group_norm_backward with various n, c, h_x_w, group combos"""
        torch.manual_seed(42)
        input_ = torch.randn(n, c, h_x_w, requires_grad=True)
        gn = nn.GroupNorm(group, c)
        output = gn(input_).sum()
        output.backward()
        expected_grad = input_.grad.clone()

        # Now call the decomposition directly
        input2 = input_.detach().clone().requires_grad_(True)
        fwd_out = torch.ops.aten.native_group_norm(
            input2, gn.weight, gn.bias, n, c, h_x_w, group, 1e-5
        )
        mean, rstd = fwd_out[1], fwd_out[2]
        grad_output = torch.ones_like(input2)

        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.native_group_norm_backward.default
        ]
        d_input, d_gamma, d_bias = decomp_fn(
            grad_output, input2, mean, rstd, gn.weight, n, c, h_x_w, group, [True, True, True]
        )

        torch.testing.assert_close(d_input, expected_grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "output_mask",
        [
            [True, True, True],
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
    )
    def test_group_norm_backward_output_mask(self, output_mask):
        """Test native_group_norm_backward with various output masks"""
        torch.manual_seed(42)
        n, c, h_x_w, group = 2, 4, 9, 2
        input_ = torch.randn(n, c, h_x_w)
        weight = torch.randn(c)
        fwd_out = torch.ops.aten.native_group_norm(
            input_, weight, torch.zeros(c), n, c, h_x_w, group, 1e-5
        )
        mean, rstd = fwd_out[1], fwd_out[2]
        grad_output = torch.randn(n, c, h_x_w)

        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.native_group_norm_backward.default
        ]
        d_input, d_gamma, d_bias = decomp_fn(
            grad_output, input_, mean, rstd, weight, n, c, h_x_w, group, output_mask
        )

        if output_mask[0]:
            assert d_input is not None and d_input.shape == input_.shape
        else:
            assert d_input is None
        if output_mask[1]:
            assert d_gamma is not None and d_gamma.shape == (c,)
        else:
            assert d_gamma is None
        if output_mask[2]:
            assert d_bias is not None and d_bias.shape == (c,)
        else:
            assert d_bias is None

    def test_group_norm_backward_no_weight(self):
        """Test native_group_norm_backward with gamma=None"""
        torch.manual_seed(42)
        n, c, h_x_w, group = 2, 4, 9, 2
        input_ = torch.randn(n, c, h_x_w)
        fwd_out = torch.ops.aten.native_group_norm(input_, None, None, n, c, h_x_w, group, 1e-5)
        mean, rstd = fwd_out[1], fwd_out[2]
        grad_output = torch.randn(n, c, h_x_w)

        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.native_group_norm_backward.default
        ]
        d_input, d_gamma, d_bias = decomp_fn(
            grad_output, input_, mean, rstd, None, n, c, h_x_w, group, [True, True, True]
        )

        assert d_input is not None and d_input.shape == input_.shape


class TestIndexSelectBackwardDecomposition:
    """Tests for index_select_backward decomposition"""

    @pytest.mark.parametrize("dim", [0, 1])
    def test_basic_accuracy(self, dim):
        """Test index_select_backward decomposition accuracy"""
        input_sizes = [3, 4]
        grad_output = torch.ones(2, 4) if dim == 0 else torch.ones(3, 2)
        index = torch.tensor([0, 2])

        result = decompositions.index_select_backward_decomposition(
            grad_output, input_sizes, dim, index
        )

        x = torch.zeros(input_sizes, requires_grad=True)
        selected = torch.index_select(x, dim, index)
        selected.backward(grad_output)
        expected = x.grad

        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [1])
    @pytest.mark.parametrize("compile_mode", [None, "neuron"])
    def test_neuron_device(self, dim, compile_mode):
        """Test index_select_backward on neuron device"""

        input_sizes = [3, 4]
        grad_output_cpu = torch.ones(2, 4) if dim == 0 else torch.ones(3, 2)
        index_cpu = torch.tensor([0, 2], dtype=torch.int32)

        grad_output = grad_output_cpu.to("neuron")
        index = index_cpu.to("neuron")

        fn = decompositions.index_select_backward_decomposition
        if compile_mode:
            fn = torch.compile(fn, backend=compile_mode)

        result = fn(grad_output, input_sizes, dim, index)

        x = torch.zeros(input_sizes, requires_grad=True)
        selected = torch.index_select(x, dim, index_cpu)
        selected.backward(grad_output_cpu)
        expected = x.grad

        torch.testing.assert_close(result.cpu(), expected)


class TestRepeatInterleaveDecomposition:
    """Tests for repeat_interleave.Tensor decomposition"""

    @pytest.mark.parametrize(
        "repeats,expected",
        [
            ([2, 1, 3], [0, 0, 1, 2, 2, 2]),
            ([1, 1, 1], [0, 1, 2]),
            ([3, 0, 2], [0, 0, 0, 2, 2]),
        ],
    )
    def test_basic_accuracy(self, repeats, expected):
        """Test repeat_interleave decomposition accuracy"""
        repeats_t = torch.tensor(repeats)
        output_size = sum(repeats)
        result = decompositions.repeat_interleave_tensor_decomposition(repeats_t, output_size)
        expected_t = torch.tensor(expected)
        torch.testing.assert_close(result, expected_t)

    @pytest.mark.parametrize(
        "repeats,expected",
        [
            ([2, 1, 3], [0, 0, 1, 2, 2, 2]),
            ([1, 1, 1], [0, 1, 2]),
            ([3, 0, 2], [0, 0, 0, 2, 2]),
        ],
    )
    @pytest.mark.parametrize("compile_mode", [None, "neuron"])
    def test_neuron_device(self, repeats, expected, compile_mode):
        """Test repeat_interleave on neuron device"""

        repeats_t = torch.tensor(repeats, device="neuron")
        output_size = sum(repeats)

        fn = decompositions.repeat_interleave_tensor_decomposition
        if compile_mode:
            fn = torch.compile(fn, backend=compile_mode)

        result = fn(repeats_t, output_size)
        expected_t = torch.tensor(expected)
        torch.testing.assert_close(result.cpu(), expected_t)


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


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSiluBackwardDecompositionE2E:
    """E2E tests for silu_backward decomposition via torch.compile with neuron backend.

    Uses the same three-way precision methodology as TestSiluBackwardPrecision:
    fp32 CPU ground truth vs CPU bf16 baseline vs Neuron bf16 target. The ratio
    of Neuron error to CPU error must stay <= 1.1, catching regressions where the
    decomposition computes in bf16 instead of upcasting to fp32.
    """

    MAX_RATIO = 1.1
    D, H = 4096, 12288

    def _run_compiled(self, model_fn, dt, device):
        """Run forward+backward through torch.compile, return input grad."""
        m, x = model_fn(dt, device)
        x = x.detach().requires_grad_(True)

        def fwd(x):
            return m(x).sum()

        if device == "neuron":
            fwd = torch.compile(fwd, backend="neuron")

        loss = fwd(x)
        loss.backward()
        return x.grad.detach().cpu().float()

    def _assert_parity(self, model_fn):
        """Assert compiled Neuron bf16 is no worse than CPU bf16 vs fp32 ground truth."""
        ref = self._run_compiled(model_fn, torch.float32, "cpu")
        cpu_bf16 = self._run_compiled(model_fn, torch.bfloat16, "cpu")
        trn_bf16 = self._run_compiled(model_fn, torch.bfloat16, "neuron")

        err_cpu = (cpu_bf16 - ref).norm().item()
        err_trn = (trn_bf16 - ref).norm().item()
        ratio = err_trn / err_cpu if err_cpu > 0 else float("inf")
        assert ratio <= self.MAX_RATIO, (
            f"Neuron bf16 error is {ratio:.4f}x CPU bf16 error "
            f"(max allowed {self.MAX_RATIO}x). "
            f"silu_backward decomposition may not be upcasting to fp32."
        )

    def test_silu_backward_precision_compiled(self):
        """silu(linear(x)) backward through torch.compile — core regression test."""
        torch.manual_seed(42)
        x_base = torch.normal(1e-6, 1e-6, (1, 2048, self.D))
        w_base = torch.normal(-0.81, 0.47, (self.H, self.D))

        def model_fn(dt, device):
            m = nn.Sequential(nn.Linear(self.D, self.H, bias=False), nn.SiLU()).to(dt).to(device)
            m[0].weight.data.copy_(w_base.to(dt))
            return m, x_base.to(dt).to(device)

        self._assert_parity(model_fn)


class TestAddbmmDecomposition:
    """Tests for addbmm decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.addbmm.default in decompositions.neuron_decompositions
        ), "aten.addbmm.default not found in neuron_decompositions"

    def test_basic(self):
        """Test basic addbmm: self + (batch1 @ batch2).sum(dim=0)"""
        torch.manual_seed(42)
        self_tensor = torch.randn(3, 4)
        batch1 = torch.randn(5, 3, 2)
        batch2 = torch.randn(5, 2, 4)

        result = decompositions.addbmm(self_tensor, batch1, batch2)
        expected = torch.addbmm(self_tensor, batch1, batch2)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "alpha,beta",
        [
            (1, 1),
            (2.0, 0.5),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (-1.0, 2.0),
        ],
    )
    def test_alpha_beta(self, alpha, beta):
        """Test addbmm with various alpha and beta values"""
        torch.manual_seed(42)
        self_tensor = torch.randn(3, 4)
        batch1 = torch.randn(5, 3, 2)
        batch2 = torch.randn(5, 2, 4)

        result = decompositions.addbmm(self_tensor, batch1, batch2, beta=beta, alpha=alpha)
        expected = torch.addbmm(self_tensor, batch1, batch2, beta=beta, alpha=alpha)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16],
    )
    def test_dtypes(self, dtype):
        """Test addbmm with various dtypes"""
        torch.manual_seed(42)
        self_tensor = torch.randn(3, 4, dtype=dtype)
        batch1 = torch.randn(2, 3, 5, dtype=dtype)
        batch2 = torch.randn(2, 5, 4, dtype=dtype)

        result = decompositions.addbmm(self_tensor, batch1, batch2)
        expected = torch.addbmm(self_tensor, batch1, batch2)

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_single_batch(self):
        """Test addbmm with batch size 1"""
        torch.manual_seed(42)
        self_tensor = torch.randn(2, 3)
        batch1 = torch.randn(1, 2, 4)
        batch2 = torch.randn(1, 4, 3)

        result = decompositions.addbmm(self_tensor, batch1, batch2)
        expected = torch.addbmm(self_tensor, batch1, batch2)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_beta_zero_ignores_self(self):
        """Test that beta=0 ignores self (even if self contains NaN)"""
        self_tensor = torch.full((2, 3), float("nan"))
        batch1 = torch.ones(1, 2, 4)
        batch2 = torch.ones(1, 4, 3)

        result = decompositions.addbmm(self_tensor, batch1, batch2, beta=0, alpha=1)
        expected = torch.addbmm(self_tensor, batch1, batch2, beta=0, alpha=1)

        torch.testing.assert_close(result, expected)


class TestHypotDecomposition:
    """Tests for hypot decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.hypot.default in decompositions.neuron_decompositions
        ), "aten.hypot.default not found in neuron_decompositions"

    def test_basic(self):
        """Test basic hypot: sqrt(a^2 + b^2)"""
        a = torch.tensor([3.0, 5.0, 0.0])
        b = torch.tensor([4.0, 12.0, 0.0])

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16],
    )
    def test_dtypes(self, dtype):
        """Test hypot with various dtypes"""
        torch.manual_seed(42)
        a = torch.randn(10, dtype=dtype)
        b = torch.randn(10, dtype=dtype)

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_both_zeros(self):
        """Test hypot when both inputs are zero"""
        a = torch.zeros(5)
        b = torch.zeros(5)

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        torch.testing.assert_close(result, expected)

    def test_negative_values(self):
        """Test hypot with negative values"""
        a = torch.tensor([-3.0, -5.0, 3.0])
        b = torch.tensor([4.0, -12.0, -4.0])

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 5), (2, 3, 4)],
    )
    def test_shapes(self, shape):
        """Test hypot with various tensor shapes"""
        torch.manual_seed(42)
        a = torch.randn(shape)
        b = torch.randn(shape)

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_large_values(self):
        """Test numerical stability with large values that would overflow naive sqrt(a^2+b^2)"""
        a = torch.tensor([1e30, 1e38])
        b = torch.tensor([1e30, 1e38])

        result = decompositions.hypot(a, b)
        expected = torch.hypot(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestAddmvDecomposition:
    """Tests for addmv decomposition (registered via PyTorch get_decompositions)"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.addmv.default in decompositions.neuron_decompositions
        ), "aten.addmv.default not found in neuron_decompositions"

    def test_basic(self):
        """Test basic addmv: self + (mat @ vec)"""
        torch.manual_seed(42)
        self_tensor = torch.randn(3)
        mat = torch.randn(3, 4)
        vec = torch.randn(4)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.addmv.default]
        result = decomp_fn(self_tensor, mat, vec)
        expected = torch.addmv(self_tensor, mat, vec)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "alpha,beta",
        [
            (1, 1),
            (2.0, 0.5),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (-1.0, 2.0),
        ],
    )
    def test_alpha_beta(self, alpha, beta):
        """Test addmv with various alpha and beta values"""
        torch.manual_seed(42)
        self_tensor = torch.randn(3)
        mat = torch.randn(3, 4)
        vec = torch.randn(4)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.addmv.default]
        result = decomp_fn(self_tensor, mat, vec, beta=beta, alpha=alpha)
        expected = torch.addmv(self_tensor, mat, vec, beta=beta, alpha=alpha)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.float16, torch.bfloat16],
    )
    def test_dtypes(self, dtype):
        """Test addmv with various dtypes"""
        torch.manual_seed(42)
        self_tensor = torch.randn(5, dtype=dtype)
        mat = torch.randn(5, 3, dtype=dtype)
        vec = torch.randn(3, dtype=dtype)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.addmv.default]
        result = decomp_fn(self_tensor, mat, vec)
        expected = torch.addmv(self_tensor, mat, vec)

        rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "m,n",
        [
            (1, 1),
            (2, 8),
            (10, 3),
            (64, 64),
        ],
    )
    def test_shapes(self, m, n):
        """Test addmv with various matrix shapes"""
        torch.manual_seed(42)
        self_tensor = torch.randn(m)
        mat = torch.randn(m, n)
        vec = torch.randn(n)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.addmv.default]
        result = decomp_fn(self_tensor, mat, vec)
        expected = torch.addmv(self_tensor, mat, vec)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_beta_zero_ignores_self(self):
        """Test that beta=0 ignores self (even if self contains NaN)"""
        self_tensor = torch.full((3,), float("nan"))
        mat = torch.ones(3, 4)
        vec = torch.ones(4)

        decomp_fn = decompositions.neuron_decompositions[torch.ops.aten.addmv.default]
        result = decomp_fn(self_tensor, mat, vec, beta=0, alpha=1)
        expected = torch.addmv(self_tensor, mat, vec, beta=0, alpha=1)

        torch.testing.assert_close(result, expected)


class TestSearchsortedDecomposition:
    """Tests for searchsorted decomposition"""

    def test_searchsorted_tensor_basic(self):
        """Test basic searchsorted with left (default) insertion"""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        values = torch.tensor([2.0, 4.0, 6.0])
        result = decompositions.searchsorted_tensor_decomposition(sorted_seq, values)
        expected = torch.searchsorted(sorted_seq, values)
        torch.testing.assert_close(result, expected)

    def test_searchsorted_tensor_right(self):
        """Test searchsorted with right insertion on duplicate boundaries"""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0])
        values = torch.tensor([2.0])
        result_left = decompositions.searchsorted_tensor_decomposition(sorted_seq, values)
        result_right = decompositions.searchsorted_tensor_decomposition(
            sorted_seq, values, right=True
        )
        expected_left = torch.searchsorted(sorted_seq, values)
        expected_right = torch.searchsorted(sorted_seq, values, right=True)
        torch.testing.assert_close(result_left, expected_left)
        torch.testing.assert_close(result_right, expected_right)

    def test_searchsorted_tensor_side_param(self):
        """Test that side='right' behaves like right=True"""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0])
        values = torch.tensor([2.0])
        result = decompositions.searchsorted_tensor_decomposition(sorted_seq, values, side="right")
        expected = torch.searchsorted(sorted_seq, values, right=True)
        torch.testing.assert_close(result, expected)

    def test_searchsorted_tensor_sorter(self):
        """Test searchsorted with a sorter index"""
        unsorted_seq = torch.tensor([3.0, 1.0, 4.0, 2.0])
        sorter = torch.argsort(unsorted_seq)
        values = torch.tensor([2.5])
        result = decompositions.searchsorted_tensor_decomposition(
            unsorted_seq, values, sorter=sorter
        )
        expected = torch.searchsorted(unsorted_seq, values, sorter=sorter)
        torch.testing.assert_close(result, expected)

    def test_searchsorted_tensor_out_int32(self):
        """Test out_int32 flag returns int32 dtype"""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0])
        values = torch.tensor([2.0])
        result = decompositions.searchsorted_tensor_decomposition(
            sorted_seq, values, out_int32=True
        )
        assert result.dtype == torch.int32

    def test_searchsorted_tensor_2d(self):
        """Test searchsorted with batched 2D inputs"""
        sorted_seq = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        values = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
        result = decompositions.searchsorted_tensor_decomposition(sorted_seq, values)
        expected = torch.searchsorted(sorted_seq, values)
        torch.testing.assert_close(result, expected)

    def test_searchsorted_scalar(self):
        """Test scalar variant wraps correctly"""
        sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0])
        result = decompositions.searchsorted_scalar_decomposition(sorted_seq, 4.0)
        expected = torch.searchsorted(sorted_seq, 4.0)
        torch.testing.assert_close(result, expected)

    def test_searchsorted_scalar_right(self):
        """Test scalar variant with right insertion"""
        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0])
        result = decompositions.searchsorted_scalar_decomposition(sorted_seq, 2.0, right=True)
        expected = torch.searchsorted(sorted_seq, 2.0, right=True)
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSearchsortedDecompositionCompileE2E:
    """E2E tests for searchsorted decomposition via torch.compile with neuron backend"""

    def test_searchsorted_basic_e2e(self):
        """Test basic searchsorted via torch.compile"""

        def fn(sorted_seq, values):
            return torch.searchsorted(sorted_seq, values)

        compiled_fn = torch.compile(fn, backend="neuron")

        sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device="neuron")
        values = torch.tensor([2.0, 4.0, 6.0], device="neuron")

        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        result = compiled_fn(sorted_seq, values)

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("right", [False, True])
    def test_searchsorted_right_e2e(self, right):
        """Test searchsorted left/right via torch.compile"""

        def fn(sorted_seq, values):
            return torch.searchsorted(sorted_seq, values, right=right)

        compiled_fn = torch.compile(fn, backend="neuron")

        sorted_seq = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0], device="neuron")
        values = torch.tensor([2.0, 3.0], device="neuron")

        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu(), right=right)
        result = compiled_fn(sorted_seq, values)

        torch.testing.assert_close(result.cpu(), expected)

    def test_searchsorted_2d_e2e(self):
        """Test batched 2D searchsorted via torch.compile"""

        def fn(sorted_seq, values):
            return torch.searchsorted(sorted_seq, values)

        compiled_fn = torch.compile(fn, backend="neuron")

        sorted_seq = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], device="neuron")
        values = torch.tensor([[2.0, 4.0], [3.0, 5.0]], device="neuron")

        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        result = compiled_fn(sorted_seq, values)

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_searchsorted_dtypes_e2e(self, dtype):
        """Test searchsorted with various dtypes via torch.compile"""
        torch.manual_seed(42)

        def fn(sorted_seq, values):
            return torch.searchsorted(sorted_seq, values)

        compiled_fn = torch.compile(fn, backend="neuron")

        sorted_seq = torch.sort(torch.randn(20, dtype=dtype))[0].to("neuron")
        values = torch.randn(5, dtype=dtype).to("neuron")

        expected = torch.searchsorted(sorted_seq.cpu(), values.cpu())
        result = compiled_fn(sorted_seq, values)

        torch.testing.assert_close(result.cpu(), expected)


class TestNativeBatchNormDecomposition:
    """Tests for native_batch_norm decomposition numerical correctness.

    Verifies that the PyTorch-provided decomposition produces outputs matching
    the monolithic aten.native_batch_norm op.
    """

    def test_native_batch_norm_accuracy_bf16(self):
        """native_batch_norm decomposition handles bf16 inputs in training mode."""
        torch.manual_seed(0)
        shape = (2, 8, 16, 16)
        num_features = 8
        x = torch.randn(shape, dtype=torch.bfloat16)
        weight = torch.randn(num_features)
        bias = torch.randn(num_features)
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)

        ref_out, _, _ = torch.ops.aten.native_batch_norm(
            x.float(),
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            0.1,
            1e-5,
        )

        decomp = decompositions.neuron_decompositions[torch.ops.aten.native_batch_norm.default]
        decomp_out, _, _ = decomp(
            x,
            weight,
            bias,
            running_mean.clone(),
            running_var.clone(),
            True,
            0.1,
            1e-5,
        )

        assert decomp_out.dtype == torch.bfloat16
        torch.testing.assert_close(decomp_out.float(), ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormStatsDecomposition:
    """Tests for batch_norm_stats decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.batch_norm_stats.default in decompositions.neuron_decompositions
        ), "aten.batch_norm_stats.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 16, 32, 32),
            (8, 32, 128),
            (2, 8, 4, 16, 16),
            (1, 8, 16, 16),
            (8, 64, 32, 32),
        ],
    )
    def test_accuracy_e2e(self, shape):
        """Test batch_norm_stats via torch.compile with neuron backend"""
        torch.manual_seed(42)
        eps = 1e-5

        def fn(x):
            return torch.ops.aten.batch_norm_stats(x, eps)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(shape)
        x_neuron = x_cpu.detach().clone().to("neuron")

        expected_mean, expected_invstd = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_stats.default
        ](x_cpu, eps)
        result_mean, result_invstd = compiled_fn(x_neuron)

        torch.testing.assert_close(result_mean.cpu(), expected_mean, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result_invstd.cpu(), expected_invstd, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7])
    def test_different_eps(self, eps):
        """Test batch_norm_stats with different epsilon values"""
        torch.manual_seed(42)

        def fn(x):
            return torch.ops.aten.batch_norm_stats(x, eps)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(4, 8, 16, 16)
        x_neuron = x_cpu.detach().clone().to("neuron")

        expected_mean, expected_invstd = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_stats.default
        ](x_cpu, eps)
        result_mean, result_invstd = compiled_fn(x_neuron)

        torch.testing.assert_close(result_mean.cpu(), expected_mean, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result_invstd.cpu(), expected_invstd, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormElemtDecomposition:
    """Tests for batch_norm_elemt decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.batch_norm_elemt.default in decompositions.neuron_decompositions
        ), "aten.batch_norm_elemt.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [(4, 8, 16, 16), (2, 16, 32, 32), (8, 32, 64)],
    )
    def test_accuracy_e2e(self, shape):
        """Test batch_norm_elemt via torch.compile with neuron backend"""
        torch.manual_seed(42)
        C = shape[1]
        weight = torch.randn(C)
        bias = torch.randn(C)
        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1
        eps = 1e-5

        def fn(x, w, b, m, inv):
            return torch.ops.aten.batch_norm_elemt(x, w, b, m, inv, eps)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(shape)
        x_neuron = x_cpu.detach().clone().to("neuron")

        expected = decompositions.neuron_decompositions[torch.ops.aten.batch_norm_elemt.default](
            x_cpu, weight, bias, mean, invstd, eps
        )
        result = compiled_fn(
            x_neuron,
            weight.to("neuron"),
            bias.to("neuron"),
            mean.to("neuron"),
            invstd.to("neuron"),
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_no_weight_no_bias(self):
        """Test batch_norm_elemt without weight and bias"""
        torch.manual_seed(42)
        shape = (2, 16, 32, 32)
        C = shape[1]
        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1
        eps = 1e-5

        def fn(x, m, inv):
            return torch.ops.aten.batch_norm_elemt(x, None, None, m, inv, eps)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(shape)
        x_neuron = x_cpu.detach().clone().to("neuron")

        expected = decompositions.neuron_decompositions[torch.ops.aten.batch_norm_elemt.default](
            x_cpu, None, None, mean, invstd, eps
        )
        result = compiled_fn(x_neuron, mean.to("neuron"), invstd.to("neuron"))

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormBackwardElemtDecomposition:
    """Tests for batch_norm_backward_elemt decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.batch_norm_backward_elemt.default in decompositions.neuron_decompositions
        ), "aten.batch_norm_backward_elemt.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape,count_val",
        [
            ((4, 8, 16, 16), 1024),
            ((2, 16, 32, 32), 2048),
            ((8, 32, 64), 512),
            ((1, 8, 16, 16), 256),
        ],
    )
    def test_accuracy_e2e(self, shape, count_val):
        """Test batch_norm_backward_elemt via torch.compile with neuron backend"""
        torch.manual_seed(42)
        C = shape[1]

        weight = torch.randn(C)
        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1
        sum_dy = torch.randn(C)
        sum_dy_xmu = torch.randn(C)
        count = torch.tensor(count_val, dtype=torch.int32)

        def fn(grad_out, inp, m, inv, w, sdy, sdyxmu, cnt):
            return torch.ops.aten.batch_norm_backward_elemt(
                grad_out, inp, m, inv, w, sdy, sdyxmu, cnt
            )

        compiled_fn = torch.compile(fn, backend="neuron")

        grad_output = torch.randn(shape)
        input_ = torch.randn(shape)

        # No CPU kernel for this op — use decomposition as reference
        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_backward_elemt.default
        ]
        expected = decomp_fn(grad_output, input_, mean, invstd, weight, sum_dy, sum_dy_xmu, count)
        result = compiled_fn(
            grad_output.to("neuron"),
            input_.to("neuron"),
            mean.to("neuron"),
            invstd.to("neuron"),
            weight.to("neuron"),
            sum_dy.to("neuron"),
            sum_dy_xmu.to("neuron"),
            count.to("neuron"),
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_no_weight(self):
        """Test batch_norm_backward_elemt without weight"""
        torch.manual_seed(42)
        shape = (2, 16, 32, 32)
        C = shape[1]

        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1
        sum_dy = torch.randn(C)
        sum_dy_xmu = torch.randn(C)
        count = torch.tensor(2048, dtype=torch.int32)

        def fn(grad_out, inp, m, inv, sdy, sdyxmu, cnt):
            return torch.ops.aten.batch_norm_backward_elemt(
                grad_out, inp, m, inv, None, sdy, sdyxmu, cnt
            )

        compiled_fn = torch.compile(fn, backend="neuron")

        grad_output = torch.randn(shape)
        input_ = torch.randn(shape)

        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_backward_elemt.default
        ]
        expected = decomp_fn(grad_output, input_, mean, invstd, None, sum_dy, sum_dy_xmu, count)
        result = compiled_fn(
            grad_output.to("neuron"),
            input_.to("neuron"),
            mean.to("neuron"),
            invstd.to("neuron"),
            sum_dy.to("neuron"),
            sum_dy_xmu.to("neuron"),
            count.to("neuron"),
        )

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormBackwardReduceDecomposition:
    """Tests for batch_norm_backward_reduce decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.batch_norm_backward_reduce.default
            in decompositions.neuron_decompositions
        ), "aten.batch_norm_backward_reduce.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [(4, 8, 16, 16), (2, 16, 32, 32), (8, 32, 64), (2, 8, 16, 16)],
    )
    def test_accuracy_e2e(self, shape):
        """Test batch_norm_backward_reduce via torch.compile with neuron backend"""
        torch.manual_seed(42)
        C = shape[1]

        weight = torch.randn(C)
        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1

        def fn(grad_out, inp, m, inv, w):
            return torch.ops.aten.batch_norm_backward_reduce(
                grad_out, inp, m, inv, w, True, True, True
            )

        compiled_fn = torch.compile(fn, backend="neuron")

        grad_output = torch.randn(shape)
        input_ = torch.randn(shape)

        # No CPU kernel — use decomposition as reference
        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_backward_reduce.default
        ]
        expected = decomp_fn(grad_output, input_, mean, invstd, weight, True, True, True)
        result = compiled_fn(
            grad_output.to("neuron"),
            input_.to("neuron"),
            mean.to("neuron"),
            invstd.to("neuron"),
            weight.to("neuron"),
        )

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "input_g,weight_g,bias_g",
        [
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
        ],
    )
    def test_output_flags(self, input_g, weight_g, bias_g):
        """Test batch_norm_backward_reduce with various output flags"""
        torch.manual_seed(42)
        shape = (2, 8, 16, 16)
        C = shape[1]

        weight = torch.randn(C)
        mean = torch.randn(C)
        invstd = torch.rand(C) + 0.1

        def fn(grad_out, inp, m, inv, w):
            return torch.ops.aten.batch_norm_backward_reduce(
                grad_out, inp, m, inv, w, input_g, weight_g, bias_g
            )

        compiled_fn = torch.compile(fn, backend="neuron")

        grad_output = torch.randn(shape)
        input_ = torch.randn(shape)

        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_backward_reduce.default
        ]
        expected = decomp_fn(grad_output, input_, mean, invstd, weight, input_g, weight_g, bias_g)
        result = compiled_fn(
            grad_output.to("neuron"),
            input_.to("neuron"),
            mean.to("neuron"),
            invstd.to("neuron"),
            weight.to("neuron"),
        )

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestErfinvDecompositionE2E:
    """E2E tests for erfinv decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.erfinv.default in decompositions.neuron_decompositions
        ), "aten.erfinv.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_accuracy_e2e(self, shape, dtype):
        """Test erfinv via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def fn(x):
            return torch.erfinv(x)

        compiled_fn = torch.compile(fn, backend="neuron")

        # erfinv input must be in (-1, 1)
        x_cpu = torch.rand(shape, dtype=dtype) * 1.8 - 0.9
        x_neuron = x_cpu.detach().clone().to("neuron")

        expected = fn(x_cpu)
        result = compiled_fn(x_neuron)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_boundary_values(self):
        """Test erfinv near boundaries"""
        torch.manual_seed(42)

        def fn(x):
            return torch.erfinv(x)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.tensor([-0.99, -0.5, 0.0, 0.5, 0.99])
        x_neuron = x_cpu.to("neuron")

        expected = fn(x_cpu)
        result = compiled_fn(x_neuron)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAddcmulDecompositionE2E:
    """E2E tests for addcmul decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.addcmul.default in decompositions.neuron_decompositions
        ), "aten.addcmul.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("value", [1, 0.5, 2.0, -1.0])
    def test_accuracy_e2e(self, shape, value):
        """Test addcmul via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def fn(input, t1, t2):
            return torch.addcmul(input, t1, t2, value=value)

        compiled_fn = torch.compile(fn, backend="neuron")

        input_cpu = torch.randn(shape)
        t1_cpu = torch.randn(shape)
        t2_cpu = torch.randn(shape)

        expected = fn(input_cpu, t1_cpu, t2_cpu)
        result = compiled_fn(input_cpu.to("neuron"), t1_cpu.to("neuron"), t2_cpu.to("neuron"))

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAddcdivDecompositionE2E:
    """E2E tests for addcdiv decomposition"""

    def test_registered_in_decomposition_table(self):
        """Test that the decomposition is registered in neuron_decompositions"""
        assert (
            torch.ops.aten.addcdiv.default in decompositions.neuron_decompositions
        ), "aten.addcdiv.default not found in neuron_decompositions"

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("value", [1, 0.5, 2.0, -1.0])
    def test_accuracy_e2e(self, shape, value):
        """Test addcdiv via torch.compile with neuron backend"""
        torch.manual_seed(42)

        def fn(input, t1, t2):
            return torch.addcdiv(input, t1, t2, value=value)

        compiled_fn = torch.compile(fn, backend="neuron")

        input_cpu = torch.randn(shape)
        t1_cpu = torch.randn(shape)
        # Avoid division by zero
        t2_cpu = torch.randn(shape).abs() + 0.1

        expected = fn(input_cpu, t1_cpu, t2_cpu)
        result = compiled_fn(input_cpu.to("neuron"), t1_cpu.to("neuron"), t2_cpu.to("neuron"))

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


class TestForeachDecompositionCompileE2E:
    """E2E tests for foreach op decompositions via torch.compile with neuron backend.

    Verifies that foreach ops (used by optimizers) compile and produce
    correct results through the Neuron backend.
    """

    @pytest.fixture(autouse=True)
    def reset_dynamo(self):
        yield
        torch._dynamo.reset()

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_foreach_list(self, op_name, dtype):
        """Test foreach .List variants compile correctly."""
        torch.manual_seed(42)
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").List

        has_alpha = op_name in ("add", "sub")
        kwargs = {"alpha": 0.5} if has_alpha else {}

        def fn(t1, t2):
            return aten_op(t1, t2, **kwargs)

        t1 = [torch.randn(64, 128, dtype=dtype, device="neuron") for _ in range(3)]
        # Avoid near-zero divisors for div in bf16
        t2 = (
            [torch.randn(64, 128, dtype=dtype, device="neuron").clamp(min=0.1) for _ in range(3)]
            if op_name == "div"
            else [torch.randn(64, 128, dtype=dtype, device="neuron") for _ in range(3)]
        )

        expected = aten_op([x.cpu() for x in t1], [x.cpu() for x in t2], **kwargs)
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    def test_foreach_scalar(self, op_name):
        """Test foreach .Scalar variants compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").Scalar

        def fn(tensors):
            return aten_op(tensors, 2.5)

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        expected = aten_op([t.cpu() for t in tensors], 2.5)
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["add", "mul", "div"])
    def test_foreach_tensor(self, op_name):
        """Test foreach .Tensor variants compile correctly (sub has no .Tensor overload)."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").Tensor

        has_alpha = op_name == "add"
        kwargs = {"alpha": 0.5} if has_alpha else {}

        def fn(tensors, scalar_t):
            return aten_op(tensors, scalar_t, **kwargs)

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        scalar_t = torch.tensor(1.5, device="neuron")

        expected = aten_op([t.cpu() for t in tensors], scalar_t.cpu(), **kwargs)
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors, scalar_t)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    def test_foreach_list_inplace(self, op_name):
        """Test foreach inplace .List variants compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}_").List

        has_alpha = op_name in ("add", "sub")
        kwargs = {"alpha": 0.5} if has_alpha else {}

        def fn(t1, t2):
            aten_op(t1, t2, **kwargs)
            return t1

        t1 = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t2 = [torch.randn(64, 128, device="neuron") for _ in range(3)]

        t1_cpu = [x.cpu().clone() for x in t1]
        t2_cpu = [x.cpu() for x in t2]
        aten_op(t1_cpu, t2_cpu, **kwargs)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, t1_cpu, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    def test_foreach_scalar_inplace(self, op_name):
        """Test foreach inplace .Scalar variants compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}_").Scalar

        def fn(tensors):
            aten_op(tensors, 2.0)
            return tensors

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        tensors_cpu = [t.cpu().clone() for t in tensors]
        aten_op(tensors_cpu, 2.0)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, tensors_cpu, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_sqrt(self):
        """Test foreach sqrt compiles correctly."""

        def fn(tensors):
            return torch.ops.aten._foreach_sqrt.default(tensors)

        tensors = [torch.rand(64, 128, device="neuron") + 0.1 for _ in range(3)]
        expected = torch.ops.aten._foreach_sqrt.default([t.cpu() for t in tensors])

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_addcmul(self):
        """Test foreach addcmul compiles correctly."""

        def fn(s, t1, t2):
            return torch.ops.aten._foreach_addcmul.Scalar(s, t1, t2, value=0.5)

        s = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t1 = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t2 = [torch.randn(64, 128, device="neuron") for _ in range(3)]

        expected = torch.ops.aten._foreach_addcmul.Scalar(
            [x.cpu() for x in s], [x.cpu() for x in t1], [x.cpu() for x in t2], value=0.5
        )

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(s, t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_addcdiv(self):
        """Test foreach addcdiv compiles correctly."""

        def fn(s, t1, t2):
            return torch.ops.aten._foreach_addcdiv.Scalar(s, t1, t2, value=0.5)

        s = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t1 = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t2 = [torch.randn(64, 128, device="neuron") + 1.0 for _ in range(3)]

        expected = torch.ops.aten._foreach_addcdiv.Scalar(
            [x.cpu() for x in s], [x.cpu() for x in t1], [x.cpu() for x in t2], value=0.5
        )

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(s, t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_list_mixed_empty(self):
        """Test foreach .List with mix of empty and non-empty tensors."""

        def fn(t1, t2):
            return torch.ops.aten._foreach_add.List(t1, t2)

        t1 = [
            torch.randn(64, device="neuron"),
            torch.empty(0, device="neuron"),
            torch.randn(32, device="neuron"),
        ]
        t2 = [
            torch.randn(64, device="neuron"),
            torch.empty(0, device="neuron"),
            torch.randn(32, device="neuron"),
        ]

        expected = torch.ops.aten._foreach_add.List([x.cpu() for x in t1], [x.cpu() for x in t2])
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            assert r.shape == e.shape
            if r.numel() > 0:
                torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["add", "mul", "div"])
    def test_foreach_scalar_dtype_promotion(self, op_name):
        """Test foreach .Scalar with integer tensors (dtype promotion to float for div)."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").Scalar

        def fn(tensors):
            return aten_op(tensors, 2)

        tensors = [torch.tensor([1, 2, 3], dtype=torch.int32, device="neuron")]
        expected = aten_op([t.cpu() for t in tensors], 2)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            assert r.dtype == e.dtype
            torch.testing.assert_close(r.cpu(), e)

    def test_foreach_list_same_object(self):
        """Test foreach .List where self and other are the same list (x + x = 2x)."""

        def fn(t):
            return torch.ops.aten._foreach_add.List(t, t)

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        expected = [t.cpu() * 2 for t in tensors]

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    def test_foreach_scalarlist(self, op_name):
        """Test foreach .ScalarList variants compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").ScalarList

        def fn(tensors):
            return aten_op(tensors, [1.5, 2.0, 0.5])

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        expected = aten_op([t.cpu() for t in tensors], [1.5, 2.0, 0.5])

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
    def test_foreach_scalarlist_inplace(self, op_name):
        """Test foreach inplace .ScalarList variants compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}_").ScalarList

        def fn(tensors):
            aten_op(tensors, [1.5, 2.0, 0.5])
            return tensors

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        tensors_cpu = [t.cpu().clone() for t in tensors]
        aten_op(tensors_cpu, [1.5, 2.0, 0.5])

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, tensors_cpu, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "op_name", ["exp", "log", "sin", "cos", "tanh", "sigmoid", "neg", "reciprocal", "abs"]
    )
    def test_foreach_unary(self, op_name):
        """Test foreach unary ops compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}").default

        def fn(tensors):
            return aten_op(tensors)

        tensors = [torch.rand(64, 128, device="neuron") + 0.1 for _ in range(3)]
        expected = aten_op([t.cpu() for t in tensors])

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-2, atol=1e-2)

    def test_foreach_clamp_max(self):
        """Test foreach clamp_max compiles correctly."""

        def fn(tensors, others):
            return torch.ops.aten._foreach_clamp_max.List(tensors, others)

        tensors = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        others = [torch.randn(64, 128, device="neuron") for _ in range(3)]

        expected = torch.ops.aten._foreach_clamp_max.List(
            [t.cpu() for t in tensors], [o.cpu() for o in others]
        )

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors, others)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_lerp_scalar(self):
        """Test foreach lerp with scalar weight compiles correctly."""

        def fn(self, other):
            return torch.ops.aten._foreach_lerp.Scalar(self, other, 0.3)

        t1 = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t2 = [torch.randn(64, 128, device="neuron") for _ in range(3)]

        expected = torch.ops.aten._foreach_lerp.Scalar(
            [t.cpu() for t in t1], [t.cpu() for t in t2], 0.3
        )

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_list_small_tensors(self):
        """Test foreach .List fallback path with numel=1 tensors (no NKI)."""
        aten_op = torch.ops.aten._foreach_add.List

        def fn(t1, t2):
            return aten_op(t1, t2)

        # numel=1 tensors should skip NKI and use torch.add fallback
        t1 = [torch.randn(1, device="neuron") for _ in range(3)]
        t2 = [torch.randn(1, device="neuron") for _ in range(3)]

        expected = aten_op([x.cpu() for x in t1], [x.cpu() for x in t2])
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_list_mismatched_shapes(self):
        """Test foreach .List fallback with different shapes (no NKI)."""
        aten_op = torch.ops.aten._foreach_add.List

        def fn(t1, t2):
            return aten_op(t1, t2)

        # Different shapes → NKI skipped, uses torch.add with broadcasting
        t1 = [torch.randn(4, 128, device="neuron"), torch.randn(64, 1, device="neuron")]
        t2 = [torch.randn(4, 128, device="neuron"), torch.randn(1, 128, device="neuron")]

        expected = aten_op([x.cpu() for x in t1], [x.cpu() for x in t2])
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_foreach_scalar_small_tensor(self):
        """Test foreach .Scalar fallback with numel=1 tensor."""
        aten_op = torch.ops.aten._foreach_mul.Scalar

        def fn(tensors):
            return aten_op(tensors, 3.0)

        tensors = [torch.randn(1, device="neuron"), torch.randn(1, device="neuron")]
        expected = aten_op([t.cpu() for t in tensors], 3.0)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("op_name", ["exp", "sin", "tanh"])
    def test_foreach_unary_inplace(self, op_name):
        """Test foreach unary inplace ops compile correctly."""
        aten_op = getattr(torch.ops.aten, f"_foreach_{op_name}_").default

        def fn(tensors):
            aten_op(tensors)
            return tensors

        tensors = [torch.rand(64, 128, device="neuron") + 0.1 for _ in range(3)]
        tensors_cpu = [t.cpu().clone() for t in tensors]
        getattr(torch.ops.aten, f"_foreach_{op_name}_").default(tensors_cpu)

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(tensors)

        for r, e in zip(result, tensors_cpu, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-2, atol=1e-2)

    def test_foreach_nki_kernel_used(self):
        """Verify NKI kernel is actually called for large tensors."""

        from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import _add_tensor_kernel

        call_count = {"n": 0}
        original_call = _add_tensor_kernel.__class__.__call__

        def counting_call(self, *args, **kwargs):
            call_count["n"] += 1
            return original_call(self, *args, **kwargs)

        with patch.object(_add_tensor_kernel.__class__, "__call__", counting_call):

            def fn(t1, t2):
                return torch.ops.aten._foreach_add.List(t1, t2)

            t1 = [torch.randn(256, 256, device="neuron") for _ in range(3)]
            t2 = [torch.randn(256, 256, device="neuron") for _ in range(3)]
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            compiled(t1, t2)
            import torch_neuronx as tnx

            tnx.synchronize()

        assert call_count["n"] == 3, f"Expected 3 NKI calls, got {call_count['n']}"

    def test_foreach_nki_fallback(self, monkeypatch):
        """Test that NKI kernel failure falls back to torch op."""
        import torch_neuronx.neuron_dynamo_backend.decompositions as decomps

        class FailingKernel:
            def __getitem__(self, key):
                return self

            def __call__(self, *args, **kwargs):
                raise RuntimeError("NKI kernel failed intentionally")

        # Patch the tensor kernel for add to always fail
        monkeypatch.setitem(decomps._FOREACH_TENSOR_KERNELS, "add", FailingKernel())

        def fn(t1, t2):
            return torch.ops.aten._foreach_add.List(t1, t2)

        t1 = [torch.randn(64, 128, device="neuron") for _ in range(3)]
        t2 = [torch.randn(64, 128, device="neuron") for _ in range(3)]

        expected = [a.cpu() + b.cpu() for a, b in zip(t1, t2, strict=False)]
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        result = compiled(t1, t2)

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r.cpu(), e, rtol=1e-3, atol=1e-3)

    def test_adam_real_optimizer(self):
        """Test real torch.optim.Adam compiles and runs."""
        model = torch.nn.Linear(256, 128).to("neuron")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def train_step(x):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            return loss

        x = torch.randn(32, 256, device="neuron")

        compiled_step = torch.compile(train_step, backend="neuron")
        loss = compiled_step(x)

        assert loss.device.type == "neuron"
        assert loss.numel() == 1


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestClampDecompositionE2E:
    """E2E tests for clamp_min and clamp_max decompositions"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_clamp_min_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.clamp_min(x, 0.0)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_clamp_max_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.clamp_max(x, 1.0)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLerpDecompositionE2E:
    """E2E tests for lerp decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
    def test_lerp_e2e(self, shape, weight):
        torch.manual_seed(42)

        def fn(start, end):
            return torch.lerp(start, end, weight)

        compiled_fn = torch.compile(fn, backend="neuron")
        start = torch.randn(shape)
        end = torch.randn(shape)
        result = compiled_fn(start.to("neuron"), end.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(start, end), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSiluDecompositionE2E:
    """E2E tests for silu decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_silu_e2e(self, shape, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.silu(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLogSigmoidDecompositionE2E:
    """E2E tests for log_sigmoid_forward decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_log_sigmoid_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.logsigmoid(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_log_sigmoid_backward_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.logsigmoid(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSincDecompositionE2E:
    """E2E tests for sinc decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_sinc_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.sinc(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestXlogyDecompositionE2E:
    """E2E tests for xlogy decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_xlogy_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.xlogy(x, y)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        y_cpu = torch.rand(shape) + 0.1
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_xlogy_scalar_self_e2e(self, shape):
        """xlogy.Scalar_Self: scalar * log(tensor). Routes through
        prims.scalar_tensor, which needs the prims_scalar_tensor decomp to lower."""
        torch.manual_seed(42)

        def fn(x):
            return torch.xlogy(2.0, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(shape) + 0.1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_xlogy_scalar_other_e2e(self, shape):
        """xlogy.Scalar_Other: tensor * log(scalar)."""
        torch.manual_seed(42)

        def fn(x):
            return torch.xlogy(x, 2.0)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(shape) + 0.1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestPrimsScalarTensorDecompositionE2E:
    """E2E tests for the prims.scalar_tensor decomp (prims_scalar_tensor)."""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_special_xlog1py_self_scalar_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.special.xlog1py(2.0, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(shape) + 0.1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_special_xlog1py_other_scalar_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.special.xlog1py(x, 2.0)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(shape) + 0.1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestVdotDecompositionE2E:
    """E2E tests for vdot decomposition"""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_vdot_e2e(self, n):
        torch.manual_seed(42)

        def fn(a, b):
            return torch.vdot(a, b)

        compiled_fn = torch.compile(fn, backend="neuron")
        a = torch.randn(n)
        b = torch.randn(n)
        result = compiled_fn(a.to("neuron"), b.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(a, b), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestTakeDecompositionE2E:
    """E2E tests for take decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_take_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x, idx):
            return torch.take(x, idx)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape)
        numel = x_cpu.numel()
        idx = torch.randint(0, numel, (5,))
        result = compiled_fn(x_cpu.to("neuron"), idx.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, idx), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestHuberLossDecompositionE2E:
    """E2E tests for huber_loss decomposition"""

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_huber_loss_e2e(self, reduction):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.huber_loss(x, y, reduction=reduction, delta=1.0)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        y_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=1e-4, atol=1e-4)

    def test_huber_loss_backward_e2e(self):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.huber_loss(x, y).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        y_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu, y_cpu).backward()
        compiled_fn(x_neuron, y_cpu.to("neuron")).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSmoothL1LossDecompositionE2E:
    """E2E tests for smooth_l1_loss decomposition"""

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_loss_e2e(self, reduction):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.smooth_l1_loss(x, y, reduction=reduction)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        y_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=1e-4, atol=1e-4)

    def test_smooth_l1_loss_backward_e2e(self):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.smooth_l1_loss(x, y).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        y_cpu = torch.randn(4, 8)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu, y_cpu).backward()
        compiled_fn(x_neuron, y_cpu.to("neuron")).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


class TestWindowFunctionDecompositions:
    """Tests for windowing-function decompositions.

    bartlett/blackman/hamming/hann/kaiser have no PyTorch decomposition and
    fail to legalize in torch-mlir. Each is lowered to arange + analytic ops;
    these check the decompositions match aten across periodic/symmetric modes,
    edge cases (length 0/1), and the parameterized overloads.
    """

    @pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 17])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_hann_matches_aten(self, window_length, periodic):
        result = decompositions.hann_window_decomposition(window_length, periodic)
        expected = torch.ops.aten.hann_window.periodic(window_length, periodic)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 17])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_hamming_matches_aten(self, window_length, periodic):
        result = decompositions.hamming_window_decomposition(window_length, periodic)
        expected = torch.ops.aten.hamming_window.periodic(window_length, periodic)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("alpha,beta", [(0.54, 0.46), (0.6, 0.4), (0.5, 0.5)])
    def test_hamming_alpha_beta_matches_aten(self, alpha, beta):
        result = decompositions.hamming_window_decomposition(16, True, alpha, beta)
        expected = torch.ops.aten.hamming_window.periodic_alpha_beta(16, True, alpha, beta)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 17])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_bartlett_matches_aten(self, window_length, periodic):
        result = decompositions.bartlett_window_decomposition(window_length, periodic)
        expected = torch.ops.aten.bartlett_window.periodic(window_length, periodic)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 17])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_blackman_matches_aten(self, window_length, periodic):
        result = decompositions.blackman_window_decomposition(window_length, periodic)
        expected = torch.ops.aten.blackman_window.periodic(window_length, periodic)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 17])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_kaiser_matches_aten(self, window_length, periodic):
        result = decompositions.kaiser_window_decomposition(window_length, periodic)
        expected = torch.ops.aten.kaiser_window.periodic(window_length, periodic)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("beta", [12.0, 8.0, 5.0, 0.5])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_kaiser_beta_matches_aten(self, beta, periodic):
        result = decompositions.kaiser_window_decomposition(14, periodic, beta)
        expected = torch.ops.aten.kaiser_window.beta(14, periodic, beta)
        torch.testing.assert_close(result, expected)

    def test_default_is_periodic(self):
        """The base overload (no periodic arg) defaults to periodic=True."""
        result = decompositions.hann_window_decomposition(10)
        expected = torch.ops.aten.hann_window.default(10)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.hann_window.default,
            torch.ops.aten.hann_window.periodic,
            torch.ops.aten.hamming_window.default,
            torch.ops.aten.hamming_window.periodic_alpha,
            torch.ops.aten.hamming_window.periodic_alpha_beta,
            torch.ops.aten.bartlett_window.default,
            torch.ops.aten.blackman_window.default,
            torch.ops.aten.kaiser_window.default,
            torch.ops.aten.kaiser_window.beta,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestWindowFunctionDecompositionsE2E:
    """E2E tests driving the window functions through torch.compile(backend='neuron').

    Exercises the full interception -> trace -> torch-mlir lowering -> execute
    path on device (the direct-call tests above only check decomposition math),
    and compares against the CPU aten reference. Windows are factory ops, so the
    compiled fn constructs the window directly on the neuron device.
    """

    @pytest.mark.parametrize("periodic", [True, False])
    def test_hann_e2e(self, periodic):
        def fn():
            return torch.ops.aten.hann_window.periodic(16, periodic, device="neuron")

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.hann_window.periodic(16, periodic)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_hamming_e2e(self, periodic):
        def fn():
            return torch.ops.aten.hamming_window.periodic(16, periodic, device="neuron")

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.hamming_window.periodic(16, periodic)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_hamming_alpha_beta_e2e(self):
        def fn():
            return torch.ops.aten.hamming_window.periodic_alpha_beta(
                16, True, 0.6, 0.4, device="neuron"
            )

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.hamming_window.periodic_alpha_beta(16, True, 0.6, 0.4)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_bartlett_e2e(self, periodic):
        def fn():
            return torch.ops.aten.bartlett_window.periodic(16, periodic, device="neuron")

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.bartlett_window.periodic(16, periodic)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_blackman_e2e(self, periodic):
        def fn():
            return torch.ops.aten.blackman_window.periodic(16, periodic, device="neuron")

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.blackman_window.periodic(16, periodic)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("beta", [12.0, 8.0, 5.0])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_kaiser_e2e(self, periodic, beta):
        def fn():
            return torch.ops.aten.kaiser_window.beta(16, periodic, beta, device="neuron")

        result = torch.compile(fn, backend="neuron")()
        expected = torch.ops.aten.kaiser_window.beta(16, periodic, beta)
        # Kaiser uses a polynomial I0 approximation (aten.i0 does not lower);
        # accurate to ~1e-6, so a slightly looser tolerance than the others.
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSoftMarginLossDecompositionE2E:
    """E2E tests for soft_margin_loss decomposition"""

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_soft_margin_loss_e2e(self, reduction):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.soft_margin_loss(x, y, reduction=reduction)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        y_cpu = torch.sign(torch.randn(4, 8))  # targets must be +1/-1
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=1e-4, atol=1e-4)

    def test_soft_margin_loss_backward_e2e(self):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.nn.functional.soft_margin_loss(x, y).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        y_cpu = torch.sign(torch.randn(4, 8))
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu, y_cpu).backward()
        compiled_fn(x_neuron, y_cpu.to("neuron")).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestEmbeddingDecompositionE2E:
    """E2E tests for embedding decomposition"""

    @pytest.mark.parametrize("num_embeddings,dim", [(10, 4), (100, 32)])
    def test_embedding_e2e(self, num_embeddings, dim):
        torch.manual_seed(42)

        def fn(weight, indices):
            return torch.nn.functional.embedding(indices, weight)

        compiled_fn = torch.compile(fn, backend="neuron")
        weight = torch.randn(num_embeddings, dim)
        indices = torch.randint(0, num_embeddings, (4, 8))
        result = compiled_fn(weight.to("neuron"), indices.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(weight, indices), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestReplicationPadDecompositionE2E:
    """E2E tests for replication_pad decompositions"""

    def test_replication_pad1d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [2, 2], mode="replicate")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_replication_pad2d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [1, 1, 1, 1], mode="replicate")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 4, 4)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_replication_pad3d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [1, 1, 1, 1, 1, 1], mode="replicate")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 4, 4, 4)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestReflectionPadDecompositionE2E:
    """E2E tests for reflection_pad decompositions"""

    def test_reflection_pad1d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [2, 2], mode="reflect")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_reflection_pad2d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [1, 1, 1, 1], mode="reflect")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 4, 4)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_reflection_pad3d_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.pad(x, [1, 1, 1, 1, 1, 1], mode="reflect")

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(1, 3, 4, 4, 4)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestGeluBackwardDecompositionE2E:
    """E2E tests for gelu_backward decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_gelu_backward_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.gelu(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestThresholdBackwardDecompositionE2E:
    """E2E tests for threshold_backward decomposition (relu backward)"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_relu_backward_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.relu(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSoftplusBackwardDecompositionE2E:
    """E2E tests for softplus_backward decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_softplus_backward_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.softplus(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNativeLayerNormBackwardDecompositionE2E:
    """E2E tests for native_layer_norm_backward decomposition"""

    @pytest.mark.parametrize(
        "shape,normalized_shape",
        [((4, 8), [8]), ((2, 4, 8), [8]), ((2, 4, 8), [4, 8])],
    )
    def test_layer_norm_backward_e2e(self, shape, normalized_shape):
        torch.manual_seed(42)
        ln = nn.LayerNorm(normalized_shape)

        def fn(x):
            return ln(x).sum()

        ln_neuron = nn.LayerNorm(normalized_shape)
        ln_neuron.load_state_dict(ln.state_dict())
        ln_neuron.to("neuron")

        def fn_neuron(x):
            return ln_neuron(x).sum()

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNativeDropoutBackwardDecompositionE2E:
    """E2E tests for native_dropout_backward decomposition"""

    def test_dropout_backward_e2e(self):
        torch.manual_seed(42)

        # Dropout with p=0 to get deterministic mask (all ones)
        def fn(x):
            return torch.nn.functional.dropout(x, p=0.0, training=True).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestIndexAddDecompositionE2E:
    """E2E tests for index_add decomposition"""

    @pytest.mark.parametrize("dim", [0, 1])
    def test_index_add_e2e(self, dim):
        torch.manual_seed(42)

        def fn(x, idx, src):
            return x.index_add(dim, idx, src)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.zeros(5, 4)
        idx = torch.tensor([0, 2, 4]) if dim == 0 else torch.tensor([1, 3])
        src_shape = [3, 4] if dim == 0 else [5, 2]
        src = torch.randn(src_shape)

        result = compiled_fn(x_cpu.to("neuron"), idx.to("neuron"), src.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, idx, src), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestIndexFillDecompositionE2E:
    """E2E tests for index_fill decomposition"""

    @pytest.mark.parametrize("dim", [0, 1])
    def test_index_fill_e2e(self, dim):
        torch.manual_seed(42)

        def fn(x, idx):
            return x.index_fill(dim, idx, 99.0)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 5)
        idx = torch.tensor([0, 2])

        result = compiled_fn(x_cpu.to("neuron"), idx.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, idx), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestStdMeanDecompositionE2E:
    """E2E tests for std_mean.correction decomposition"""

    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_std_mean_e2e(self, dim, keepdim):
        torch.manual_seed(42)

        def fn(x):
            return torch.std_mean(x, dim=dim, keepdim=keepdim)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        result_std, result_mean = compiled_fn(x_cpu.to("neuron"))
        expected_std, expected_mean = fn(x_cpu)
        torch.testing.assert_close(result_std.cpu(), expected_std, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result_mean.cpu(), expected_mean, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestIsinDecompositionE2E:
    """E2E tests for isin decomposition"""

    def test_isin_e2e(self):
        torch.manual_seed(42)

        def fn(elements, test_elements):
            return torch.isin(elements, test_elements)

        compiled_fn = torch.compile(fn, backend="neuron")
        elements = torch.tensor([1, 2, 3, 4, 5])
        test_elements = torch.tensor([2, 4, 6])
        result = compiled_fn(elements.to("neuron"), test_elements.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(elements, test_elements))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestGridSampler2dDecompositionE2E:
    """E2E tests for grid_sampler_2d decomposition"""

    def test_grid_sampler_2d_e2e(self):
        torch.manual_seed(42)

        def fn(input, grid):
            return torch.nn.functional.grid_sample(input, grid, align_corners=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        input_cpu = torch.randn(1, 1, 4, 4)
        grid_cpu = torch.randn(1, 3, 3, 2).clamp(-1, 1)

        result = compiled_fn(input_cpu.to("neuron"), grid_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(input_cpu, grid_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestPreluDecompositionE2E:
    """E2E tests for _prelu_kernel decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8), (2, 4, 8)])
    def test_prelu_e2e(self, shape):
        torch.manual_seed(42)

        weight = torch.tensor([0.25])

        def fn(x):
            return torch.nn.functional.prelu(x, weight)

        weight_neuron = weight.to("neuron")

        def fn_neuron(x):
            return torch.nn.functional.prelu(x, weight_neuron)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        x_cpu = torch.randn(shape)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_prelu_backward_e2e(self):
        torch.manual_seed(42)

        prelu_cpu = nn.PReLU()
        prelu_neuron = nn.PReLU()
        prelu_neuron.load_state_dict(prelu_cpu.state_dict())
        prelu_neuron.to("neuron")

        def fn_cpu(x):
            return prelu_cpu(x).sum()

        def fn_neuron(x):
            return prelu_neuron(x).sum()

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn_cpu(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestGluBackwardDecompositionE2E:
    """E2E tests for glu_backward decomposition"""

    @pytest.mark.parametrize("dim", [-1, 1])
    def test_glu_backward_e2e(self, dim):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.glu(x, dim=dim).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestHardsigmoidBackwardDecompositionE2E:
    """E2E tests for hardsigmoid_backward decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_hardsigmoid_backward_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.hardsigmoid(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestHardswishBackwardDecompositionE2E:
    """E2E tests for hardswish_backward decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_hardswish_backward_e2e(self, shape):
        torch.manual_seed(42)

        def fn(x):
            return torch.nn.functional.hardswish(x).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLogitBackwardDecompositionE2E:
    """E2E tests for logit_backward decomposition"""

    def test_logit_backward_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.logit(x, eps=1e-6).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.rand(4, 8, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)


# Integer dtypes the bitwise/shift bridges must handle. The bridge wraps the
# scalar operand as a 0-d tensor of the input's dtype, so each dtype exercises a
# distinct StableHLO path.
_BITWISE_SHIFT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBitwiseShiftDecompositionE2E:
    """E2E tests for bitwise shift decompositions."""

    @pytest.mark.parametrize("dtype", _BITWISE_SHIFT_DTYPES)
    def test_right_shift_tensor_scalar_e2e(self, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.bitwise_right_shift(x, 1)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randint(0, 100, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    @pytest.mark.parametrize("dtype", _BITWISE_SHIFT_DTYPES)
    def test_left_shift_tensor_scalar_e2e(self, dtype):
        torch.manual_seed(42)

        def fn(x):
            return torch.bitwise_left_shift(x, 2)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randint(0, 8, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    @pytest.mark.parametrize("dtype", _BITWISE_SHIFT_DTYPES)
    def test_left_shift_scalar_tensor_e2e(self, dtype):
        """Scalar on the left: bitwise_left_shift(2, tensor)."""
        torch.manual_seed(42)

        def fn(x):
            return torch.bitwise_left_shift(2, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randint(0, 5, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBitwiseLogicalDecompositionE2E:
    """E2E tests for bitwise_and/or/xor Scalar_Tensor decompositions."""

    # Logical bitwise ops support integer dtypes AND bool.
    _DTYPES = (*_BITWISE_SHIFT_DTYPES, torch.bool)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_bitwise_and_scalar_tensor_e2e(self, dtype):
        torch.manual_seed(42)
        scalar = True if dtype is torch.bool else 0xF

        def fn(x):
            return torch.bitwise_and(scalar, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        hi = 2 if dtype is torch.bool else 100
        x_cpu = torch.randint(0, hi, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_bitwise_or_scalar_tensor_e2e(self, dtype):
        torch.manual_seed(42)
        scalar = True if dtype is torch.bool else 0x0F

        def fn(x):
            return torch.bitwise_or(scalar, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        hi = 2 if dtype is torch.bool else 100
        x_cpu = torch.randint(0, hi, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_bitwise_xor_scalar_tensor_e2e(self, dtype):
        torch.manual_seed(42)
        scalar = True if dtype is torch.bool else 0xF

        def fn(x):
            return torch.bitwise_xor(scalar, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        hi = 2 if dtype is torch.bool else 100
        x_cpu = torch.randint(0, hi, (4, 8), dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestFillDecompositionE2E:
    """E2E tests for fill.Scalar and fill.Tensor decompositions"""

    def test_fill_scalar_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.fill(x, 5.0)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    def test_fill_tensor_e2e(self):
        torch.manual_seed(42)

        val = torch.tensor(7.0)

        def fn(x):
            return torch.fill(x, val)

        val_neuron = val.to("neuron")

        def fn_neuron(x):
            return torch.fill(x, val_neuron)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        x_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRsubDecompositionE2E:
    """E2E tests for rsub.Tensor decomposition"""

    @pytest.mark.parametrize("shape", [(8,), (4, 8)])
    def test_rsub_e2e(self, shape):
        torch.manual_seed(42)

        def fn(a, b):
            return torch.rsub(a, b)

        compiled_fn = torch.compile(fn, backend="neuron")
        a = torch.randn(shape)
        b = torch.randn(shape)
        result = compiled_fn(a.to("neuron"), b.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(a, b), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRemainderDecompositionE2E:
    """E2E tests for remainder.Scalar_Tensor decomposition."""

    def test_remainder_scalar_tensor_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.remainder(10.0, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.tensor([3.0, 4.0, 5.0, 7.0])
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestCopysignScalarDecompositionE2E:
    """E2E tests for copysign.Scalar decomposition:q."""

    @pytest.mark.parametrize("scalar", [2.0, -3.0])
    def test_copysign_scalar_e2e(self, scalar):
        torch.manual_seed(42)

        def fn(x):
            return torch.copysign(x, scalar)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSpecialFunctionsDecompositionE2E:
    """E2E tests for special_* decompositions"""

    def test_special_xlog1py_e2e(self):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.special.xlog1py(x, y)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.randn(8)
        y = torch.rand(8) + 0.1
        result = compiled_fn(x.to("neuron"), y.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_special_entr_e2e(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.special.entr(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.rand(8) + 0.1
        result = compiled_fn(x.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRandomDistributionDecompositionE2E:
    """E2E tests for random distribution decompositions (cauchy, geometric, log_normal).
    These are non-deterministic so we only check shape and dtype."""

    def test_cauchy_e2e(self):
        def fn(x):
            return torch.ops.aten.cauchy.default(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.empty(4, 8, device="neuron")
        result = compiled_fn(x)
        assert result.shape == (4, 8)

    def test_geometric_e2e(self):
        def fn(x):
            return torch.ops.aten.geometric.default(x, 0.5)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.empty(4, 8, device="neuron")
        result = compiled_fn(x)
        assert result.shape == (4, 8)

    def test_log_normal_e2e(self):
        def fn(x):
            return torch.ops.aten.log_normal.default(x)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.empty(4, 8, device="neuron")
        result = compiled_fn(x)
        assert result.shape == (4, 8)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLinspaceDecompositionUnit:
    """Unit tests for linspace_decomposition edge cases"""

    @pytest.mark.parametrize("steps,expected_len", [(0, 0), (1, 1), (2, 2)])
    def test_edge_cases(self, steps, expected_len):
        result = decompositions.linspace_decomposition(0.0, 10.0, steps)
        assert result.shape == (expected_len,)
        if steps == 1:
            assert result[0] == 0.0
        if steps == 2:
            torch.testing.assert_close(result, torch.tensor([0.0, 10.0]))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_output_dtype(self, dtype):
        result = decompositions.linspace_decomposition(0.0, 10.0, 5, dtype=dtype)
        assert result.dtype == dtype
        assert result.shape == (5,)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLinspaceLogspaceDecompositionE2E:
    """E2E tests for linspace and logspace decompositions"""

    def test_linspace_scalar_tensor_e2e(self):
        torch.manual_seed(42)

        end = torch.tensor(10.0)

        def fn():
            return torch.ops.aten.linspace.Scalar_Tensor(0.0, end, 5)

        end_neuron = end.to("neuron")

        def fn_neuron():
            return torch.ops.aten.linspace.Scalar_Tensor(0.0, end_neuron, 5)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        result = compiled_fn()
        torch.testing.assert_close(result.cpu(), fn(), rtol=1e-4, atol=1e-4)

    def test_linspace_tensor_scalar_e2e(self):
        torch.manual_seed(42)

        start = torch.tensor(0.0)

        def fn():
            return torch.ops.aten.linspace.Tensor_Scalar(start, 10.0, 5)

        start_neuron = start.to("neuron")

        def fn_neuron():
            return torch.ops.aten.linspace.Tensor_Scalar(start_neuron, 10.0, 5)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        result = compiled_fn()
        torch.testing.assert_close(result.cpu(), fn(), rtol=1e-4, atol=1e-4)

    def test_linspace_tensor_tensor_e2e(self):
        torch.manual_seed(42)

        start = torch.tensor(0.0)
        end = torch.tensor(10.0)

        def fn():
            return torch.ops.aten.linspace.Tensor_Tensor(start, end, 5)

        start_neuron = start.to("neuron")
        end_neuron = end.to("neuron")

        def fn_neuron():
            return torch.ops.aten.linspace.Tensor_Tensor(start_neuron, end_neuron, 5)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        result = compiled_fn()
        torch.testing.assert_close(result.cpu(), fn(), rtol=1e-4, atol=1e-4)

    def test_logspace_e2e(self):
        torch.manual_seed(42)

        def fn():
            return torch.logspace(0, 2, 5)

        compiled_fn = torch.compile(fn, backend="neuron")
        result = compiled_fn()
        torch.testing.assert_close(result.cpu(), fn(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_linspace_dtype_e2e(self, dtype):
        start = torch.tensor(0.0)
        end = torch.tensor(10.0)

        def fn():
            return torch.ops.aten.linspace.Tensor_Tensor(start, end, 5, dtype=dtype)

        start_neuron = start.to("neuron")
        end_neuron = end.to("neuron")

        def fn_neuron():
            return torch.ops.aten.linspace.Tensor_Tensor(start_neuron, end_neuron, 5, dtype=dtype)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        result = compiled_fn()
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), fn(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestUnsafeMaskedIndexDecompositionE2E:
    """E2E tests for _unsafe_masked_index and _unsafe_masked_index_put_accumulate"""

    def test_unsafe_masked_index_via_gather(self):
        """Test _unsafe_masked_index indirectly via gather which may use it."""
        torch.manual_seed(42)

        def fn(x, idx):
            return torch.gather(x, 1, idx)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.randn(4, 8)
        idx = torch.randint(0, 8, (4, 3))
        result = compiled_fn(x_cpu.to("neuron"), idx.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, idx), rtol=1e-4, atol=1e-4)

    def test_unsafe_masked_index_put_accumulate_via_scatter_add(self):
        """Test _unsafe_masked_index_put_accumulate indirectly via scatter_add."""
        torch.manual_seed(42)

        def fn(x, idx, src):
            return x.scatter_add(1, idx, src)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.zeros(4, 8)
        idx = torch.randint(0, 8, (4, 3))
        src = torch.randn(4, 3)
        result = compiled_fn(x_cpu.to("neuron"), idx.to("neuron"), src.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, idx, src), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBucketizeDecompositionE2E:
    """E2E tests for bucketize.Scalar decomposition"""

    def test_bucketize_scalar_e2e(self):
        torch.manual_seed(42)

        boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        def fn():
            return torch.bucketize(4.5, boundaries)

        boundaries_neuron = boundaries.to("neuron")

        def fn_neuron():
            return torch.bucketize(4.5, boundaries_neuron)

        compiled_fn = torch.compile(fn_neuron, backend="neuron")
        result = compiled_fn()
        torch.testing.assert_close(result.cpu(), fn())


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAddmmActivationDecompositionE2E:
    """E2E tests for _addmm_activation decomposition"""

    def test_addmm_activation_relu_e2e(self):
        torch.manual_seed(42)

        def fn(input, m1, m2):
            # _addmm_activation is addmm + activation fused
            return torch.ops.aten._addmm_activation.default(input, m1, m2, use_gelu=False)

        compiled_fn = torch.compile(fn, backend="neuron")
        input_cpu = torch.randn(4)
        m1 = torch.randn(3, 5)
        m2 = torch.randn(5, 4)
        result = compiled_fn(input_cpu.to("neuron"), m1.to("neuron"), m2.to("neuron"))
        expected = fn(input_cpu, m1, m2)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_activation_gelu_e2e(self):
        torch.manual_seed(42)

        def fn(input, m1, m2):
            return torch.ops.aten._addmm_activation.default(input, m1, m2, use_gelu=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        input_cpu = torch.randn(4)
        m1 = torch.randn(3, 5)
        m2 = torch.randn(5, 4)
        result = compiled_fn(input_cpu.to("neuron"), m1.to("neuron"), m2.to("neuron"))
        expected = fn(input_cpu, m1, m2)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestChunkCatDecompositionE2E:
    """E2E tests for _chunk_cat decomposition"""

    def test_chunk_cat_e2e(self):
        torch.manual_seed(42)

        def fn(x, y):
            return torch.ops.aten._chunk_cat.default([x, y], dim=0, num_chunks=2)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        result = compiled_fn(x.to("neuron"), y.to("neuron"))
        expected = fn(x, y)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormBackwardDecompositionE2E:
    """E2E tests for batch_norm_backward decomposition"""

    def test_batch_norm_backward_e2e(self):
        torch.manual_seed(42)

        class BNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(8)

            def forward(self, x):
                return self.bn(x).sum()

        model_cpu = BNModel().train()
        model_neuron = BNModel().train()
        model_neuron.load_state_dict(model_cpu.state_dict())
        model_neuron.to("neuron")

        compiled_model = torch.compile(model_neuron, backend="neuron")

        x_cpu = torch.randn(2, 8, 4, 4, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        model_cpu(x_cpu).backward()
        compiled_model(x_neuron).backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAllDimsDecompositionE2E:
    """E2E tests for all.dims decomposition"""

    def test_all_dims_single(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.ops.aten.all.dims(x, [1])

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.tensor([[True, True, False], [True, True, True]])
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    def test_all_dims_multiple(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.ops.aten.all.dims(x, [0, 2])

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.ones(4, 3, 5, dtype=torch.bool)
        x_cpu[2, 1, 3] = False
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    def test_all_dims_keepdim(self):
        torch.manual_seed(42)

        def fn(x):
            return torch.ops.aten.all.dims(x, [0, 1], True)

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.ones(4, 3, 5, dtype=torch.bool)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))

    @pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16, torch.float32])
    def test_all_dims_numeric_dtype(self, dtype):
        """Test all.dims with non-bool inputs (non-zero is truthy)"""

        def fn(x):
            return torch.ops.aten.all.dims(x, [1])

        compiled_fn = torch.compile(fn, backend="neuron")
        x_cpu = torch.ones(3, 4, dtype=dtype)
        x_cpu[1, 2] = 0
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestShrinkBackwardDecompositionsE2E:
    """E2E tests for hardshrink/softshrink backward via torch.compile(backend='neuron').

    Each op is driven through the full interception -> trace -> torch-mlir
    lowering -> execute path on device and compared against a CPU reference.
    Backward ops are run under torch.no_grad(), matching how autograd invokes them.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("lambd", [0.5, 1.0, 0.0, 2.0])
    def test_hardshrink_backward_e2e(self, lambd, dtype):
        torch.manual_seed(0)

        def fn(grad, x):
            return torch.ops.aten.hardshrink_backward.default(grad, x, lambd)

        compiled_fn = torch.compile(fn, backend="neuron")
        grad = torch.randn(64, dtype=dtype)
        x = torch.randn(64, dtype=dtype)
        with torch.no_grad():
            result = compiled_fn(grad.to("neuron"), x.to("neuron"))
            expected = fn(grad, x)
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("lambd", [0.5, 1.0, 0.0, 2.0])
    def test_softshrink_backward_e2e(self, lambd, dtype):
        torch.manual_seed(0)

        def fn(grad, x):
            return torch.ops.aten.softshrink_backward.default(grad, x, lambd)

        compiled_fn = torch.compile(fn, backend="neuron")
        grad = torch.randn(64, dtype=dtype)
        x = torch.randn(64, dtype=dtype)
        with torch.no_grad():
            result = compiled_fn(grad.to("neuron"), x.to("neuron"))
            expected = fn(grad, x)
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_shrink_backward_boundary_values(self, dtype):
        """Values exactly at +/-lambd are zeroed (strict inequality)."""

        def fn(grad, x):
            return torch.ops.aten.hardshrink_backward.default(grad, x, 0.5)

        compiled_fn = torch.compile(fn, backend="neuron")
        x = torch.tensor([-0.5, 0.5, 0.0, 1.0], dtype=dtype)
        grad = torch.ones_like(x)
        with torch.no_grad():
            result = compiled_fn(grad.to("neuron"), x.to("neuron"))
            expected = fn(grad, x)
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("op_name", ["hardshrink_backward", "softshrink_backward"])
    def test_shrink_backward_grad_input_overload_e2e(self, op_name, dtype):
        """The .grad_input out-variant reaches the registered .default decomp via
        functionalization (it is intentionally not in neuron_decompositions, so a
        registration assert can't cover it — only an e2e run can)."""
        torch.manual_seed(0)
        op = getattr(torch.ops.aten, op_name).grad_input

        def fn(grad, x):
            out = torch.empty_like(x)
            return op(grad, x, 0.5, grad_input=out)

        compiled_fn = torch.compile(fn, backend="neuron")
        grad = torch.randn(64, dtype=dtype)
        x = torch.randn(64, dtype=dtype)
        with torch.no_grad():
            result = compiled_fn(grad.to("neuron"), x.to("neuron"))
            expected = fn(grad, x)
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.hardshrink_backward.default,
            torch.ops.aten.softshrink_backward.default,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNllLoss2dBackwardDecompositionE2E:
    """E2E test for nll_loss2d_backward via torch.compile(backend='neuron')."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_nll_loss2d_backward_e2e(self, dtype):
        torch.manual_seed(0)
        n, c, h, w = 2, 4, 3, 3
        x = torch.randn(n, c, h, w, dtype=dtype)
        target = torch.randint(0, c, (n, h, w))
        out, total_weight = torch.ops.aten.nll_loss2d_forward.default(x, target, None, 1, -100)
        grad_output = torch.randn_like(out)

        def fn(grad_output, x, target, total_weight):
            return torch.ops.aten.nll_loss2d_backward.default(
                grad_output, x, target, None, 1, -100, total_weight
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        with torch.no_grad():
            result = compiled_fn(
                grad_output.to("neuron"),
                x.to("neuron"),
                target.to("neuron"),
                total_weight.to("neuron"),
            )
            expected = fn(grad_output, x, target, total_weight)
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.nll_loss2d_backward.default,
            torch.ops.aten.nll_loss2d_backward.grad_input,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestScatterReduceDecompositionE2E:
    """E2E tests for scatter.reduce / scatter.value_reduce decompositions."""

    def test_registered_in_decomposition_table(self):
        assert torch.ops.aten.scatter.reduce in decompositions.neuron_decompositions
        assert torch.ops.aten.scatter.value_reduce in decompositions.neuron_decompositions

    # torch.scatter(reduce=...) accepts "add" / "multiply". The decomposition
    # routes "add" to scatter_reduce.two(sum); "multiply" falls back to PyTorch
    # (prod semantics differ), so only "add" is exercised here -- and "add" is
    # the only reduce the shape registry / vertical suite uses.
    @pytest.mark.parametrize("reduce", ["add"])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_scatter_src_reduce_e2e(self, reduce, dim):
        """scatter(reduce=...) with a tensor src via torch.compile."""
        torch.manual_seed(0)

        def fn(x, index, src):
            return torch.scatter(x, dim, index, src, reduce=reduce)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(8, 8)
        index_cpu = torch.randint(0, 8, (8, 8))
        src_cpu = torch.randn(8, 8)

        expected = fn(x_cpu, index_cpu, src_cpu)
        result = compiled_fn(x_cpu.to("neuron"), index_cpu.to("neuron"), src_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("reduce", ["add"])
    def test_scatter_value_reduce_e2e(self, reduce):
        """scatter(reduce=...) with a scalar value via torch.compile."""
        torch.manual_seed(0)

        def fn(x, index):
            return torch.scatter(x, 0, index, 2.5, reduce=reduce)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(8, 8)
        index_cpu = torch.randint(0, 8, (8, 8))

        expected = fn(x_cpu, index_cpu)
        result = compiled_fn(x_cpu.to("neuron"), index_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestIndexReduceDecompositionE2E:
    """E2E tests for index_reduce decomposition."""

    def test_registered_in_decomposition_table(self):
        assert torch.ops.aten.index_reduce.default in decompositions.neuron_decompositions

    @pytest.mark.parametrize("reduce", ["prod", "amax", "amin", "mean"])
    @pytest.mark.parametrize("include_self", [True, False])
    def test_index_reduce_e2e(self, reduce, include_self):
        """index_reduce via torch.compile, including the sparse (untouched-row) case."""
        torch.manual_seed(0)

        def fn(x, index, source):
            return torch.index_reduce(x, 0, index, source, reduce, include_self=include_self)

        compiled_fn = torch.compile(fn, backend="neuron")

        # 64 indices into 256 rows -> many rows untouched, exercising the
        # untouched-row handling.
        x_cpu = torch.randn(256, 256)
        index_cpu = torch.randint(0, 256, (64,))
        source_cpu = torch.randn(64, 256)

        expected = fn(x_cpu, index_cpu, source_cpu)
        result = compiled_fn(x_cpu.to("neuron"), index_cpu.to("neuron"), source_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestDiagonalScatterDecompositionE2E:
    """E2E tests for diagonal_scatter decomposition."""

    def test_registered_in_decomposition_table(self):
        assert torch.ops.aten.diagonal_scatter.default in decompositions.neuron_decompositions

    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_diagonal_scatter_2d_e2e(self, offset):
        """diagonal_scatter on a 2-D matrix with varying offsets."""
        torch.manual_seed(0)

        def fn(x, src):
            return torch.diagonal_scatter(x, src, offset=offset)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(8, 8)
        diag_len = x_cpu.diagonal(offset=offset).shape[0]
        src_cpu = torch.randn(diag_len)

        expected = fn(x_cpu, src_cpu)
        result = compiled_fn(x_cpu.to("neuron"), src_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_diagonal_scatter_batched_e2e(self):
        """diagonal_scatter on a batched (3-D) tensor over dims (1, 2)."""
        torch.manual_seed(0)

        def fn(x, src):
            return torch.diagonal_scatter(x, src, offset=0, dim1=1, dim2=2)

        compiled_fn = torch.compile(fn, backend="neuron")

        x_cpu = torch.randn(4, 6, 6)
        src_cpu = torch.randn(4, 6)

        expected = fn(x_cpu, src_cpu)
        result = compiled_fn(x_cpu.to("neuron"), src_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSegmentReduceDecompositionE2E:
    """E2E tests for segment_reduce decomposition (lengths-based, axis 0)."""

    def test_registered_in_decomposition_table(self):
        assert torch.ops.aten.segment_reduce.default in decompositions.neuron_decompositions

    # The decomposition handles sum/mean (the reductions the shape registry
    # exercises); max/min/prod return NotImplemented and fall back to PyTorch.
    @pytest.mark.parametrize("reduce", ["sum", "mean"])
    def test_segment_reduce_lengths_e2e(self, reduce):
        """segment_reduce with lengths= along axis 0 via torch.compile."""
        torch.manual_seed(0)

        lengths_cpu = torch.tensor([2, 3, 1, 2])  # sums to 8 = data.shape[0]

        def fn(data, lengths):
            return torch.segment_reduce(data, reduce, lengths=lengths, axis=0)

        compiled_fn = torch.compile(fn, backend="neuron")

        data_cpu = torch.randn(8, 16)

        expected = fn(data_cpu, lengths_cpu)
        result = compiled_fn(data_cpu.to("neuron"), lengths_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSegmentReduceBackwardDecompositionE2E:
    """E2E tests for _segment_reduce_backward decomposition (sum/mean)."""

    def test_registered_in_decomposition_table(self):
        assert (
            torch.ops.aten._segment_reduce_backward.default in decompositions.neuron_decompositions
        )

    @pytest.mark.parametrize("reduce", ["sum", "mean"])
    def test_segment_reduce_backward_e2e(self, reduce):
        """Gradient of segment_reduce flows back correctly through torch.compile."""
        torch.manual_seed(0)

        lengths_cpu = torch.tensor([2, 3, 1, 2])  # sums to 8

        def fn(data, lengths):
            out = torch.segment_reduce(data, reduce, lengths=lengths, axis=0)
            return out.sum()

        compiled_fn = torch.compile(fn, backend="neuron")

        data_cpu = torch.randn(8, 16, requires_grad=True)
        data_neuron = data_cpu.detach().clone().to("neuron").requires_grad_(True)

        fn(data_cpu, lengths_cpu).backward()
        compiled_fn(data_neuron, lengths_cpu.to("neuron")).backward()

        torch.testing.assert_close(data_neuron.grad.cpu(), data_cpu.grad, rtol=1e-3, atol=1e-3)


class TestUpsampleNearest2dBackwardDecomposition:
    """Tests for the custom upsample_nearest2d_backward decomposition."""

    def _reference(self, grad, output_size, input_size, scales_h=None, scales_w=None):
        return torch.ops.aten.upsample_nearest2d_backward.default(
            grad, output_size, input_size, scales_h, scales_w
        )

    @pytest.mark.parametrize(
        "input_size,output_size",
        [
            ([2, 3, 4, 4], [8, 8]),  # integer 2x upsample
            ([1, 2, 5, 3], [10, 9]),  # non-integer, asymmetric
            ([2, 1, 4, 4], [2, 2]),  # downsample (output < input)
            ([1, 1, 3, 3], [7, 5]),  # odd ratios
            ([2, 3, 4, 4], [4, 4]),  # identity size
        ],
    )
    def test_matches_aten(self, input_size, output_size):
        """Decomposition matches aten's native backward for various shapes."""
        grad = torch.randn(input_size[0], input_size[1], *output_size, dtype=torch.float64)
        result = decompositions.upsample_nearest2d_backward_decomposition(
            grad, output_size, input_size
        )
        expected = self._reference(grad, output_size, input_size)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "input_size,output_size,scales_h,scales_w",
        [
            ([2, 3, 4, 4], [8, 8], 2.0, 2.0),
            ([1, 2, 4, 6], [8, 18], 2.0, 3.0),
        ],
    )
    def test_matches_aten_with_scales(self, input_size, output_size, scales_h, scales_w):
        """Explicit scales_h/scales_w match aten's source-index computation."""
        grad = torch.randn(input_size[0], input_size[1], *output_size, dtype=torch.float64)
        result = decompositions.upsample_nearest2d_backward_decomposition(
            grad, output_size, input_size, scales_h, scales_w
        )
        expected = self._reference(grad, output_size, input_size, scales_h, scales_w)
        torch.testing.assert_close(result, expected)

    def test_registered(self):
        """The op is registered in the Neuron decomposition table."""
        assert (
            torch.ops.aten.upsample_nearest2d_backward.default
            in decompositions.neuron_decompositions
        )


class TestUpsampleLinearFamilyDecomposition:
    """Tests for the custom linear-family upsample decompositions.

    PyTorch's builtins are wrong on Neuron, so we replace them with separable
    matmul decompositions; these must match aten on CPU.
    """

    @pytest.mark.parametrize(
        "in_l,out_l",
        [(8, 16), (16, 8), (7, 10), (8, 8)],
    )
    def test_linear1d_matches_aten(self, in_l, out_l):
        x = torch.randn(2, 3, in_l, dtype=torch.float64)
        result = decompositions.upsample_linear1d_decomposition(x, [out_l], False)
        expected = torch.ops.aten.upsample_linear1d.default(x, [out_l], False, None)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "in_hw,out_hw",
        [((8, 8), (16, 16)), ((16, 16), (8, 8)), ((7, 9), (10, 6)), ((8, 8), (8, 8))],
    )
    def test_bilinear2d_matches_aten(self, in_hw, out_hw):
        x = torch.randn(2, 3, *in_hw, dtype=torch.float64)
        result = decompositions.upsample_bilinear2d_decomposition(x, list(out_hw), False)
        expected = torch.ops.aten.upsample_bilinear2d.default(x, list(out_hw), False)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "in_dhw,out_dhw",
        [((4, 4, 4), (8, 8, 8)), ((6, 5, 7), (2, 3, 4))],
    )
    def test_trilinear3d_matches_aten(self, in_dhw, out_dhw):
        x = torch.randn(1, 2, *in_dhw, dtype=torch.float64)
        result = decompositions.upsample_trilinear3d_decomposition(x, list(out_dhw), False)
        expected = torch.ops.aten.upsample_trilinear3d.default(x, list(out_dhw), False)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("in_hw,out_hw", [((8, 8), (16, 16)), ((7, 9), (4, 4))])
    def test_bilinear2d_align_corners_matches_aten(self, in_hw, out_hw):
        """align_corners=True is supported (distinct center formula, no AA)."""
        x = torch.randn(2, 3, *in_hw, dtype=torch.float64)
        result = decompositions.upsample_bilinear2d_decomposition(x, list(out_hw), True)
        expected = torch.ops.aten.upsample_bilinear2d.default(x, list(out_hw), True)
        torch.testing.assert_close(result, expected)

    def test_vec_matches_aten(self):
        """The .vec overloads match aten (both output_size and scale_factors)."""
        x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
        torch.testing.assert_close(
            decompositions.upsample_bilinear2d_vec_decomposition(x, [16, 16], False, None),
            torch.ops.aten.upsample_bilinear2d.vec(x, [16, 16], False, None),
        )
        torch.testing.assert_close(
            decompositions.upsample_bilinear2d_vec_decomposition(x, None, False, [2.0, 2.0]),
            torch.ops.aten.upsample_bilinear2d.vec(x, None, False, [2.0, 2.0]),
        )

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.upsample_linear1d.default,
            torch.ops.aten.upsample_linear1d.vec,
            torch.ops.aten.upsample_bilinear2d.default,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.upsample_trilinear3d.default,
            torch.ops.aten.upsample_trilinear3d.vec,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


class TestUpsampleBilinear2dAADecomposition:
    """Tests for the anti-aliased bilinear upsample decompositions."""

    @pytest.mark.parametrize(
        "in_hw,out_hw",
        [
            ((8, 8), (4, 4)),  # downsample (AA active)
            ((8, 8), (3, 5)),  # asymmetric downsample
            ((7, 9), (8, 8)),  # mixed up/down
            ((16, 16), (10, 10)),  # downsample non-integer ratio
            ((4, 4), (8, 8)),  # pure upsample (AA collapses to bilinear)
        ],
    )
    def test_forward_matches_aten(self, in_hw, out_hw):
        x = torch.randn(2, 3, *in_hw, dtype=torch.float64)
        result = decompositions.upsample_bilinear2d_aa_decomposition(x, list(out_hw), False)
        expected = torch.ops.aten._upsample_bilinear2d_aa.default(x, list(out_hw), False)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "in_hw,out_hw",
        [((8, 8), (4, 4)), ((7, 9), (3, 5)), ((4, 4), (8, 8))],
    )
    def test_backward_matches_aten(self, in_hw, out_hw):
        grad = torch.randn(2, 3, *out_hw, dtype=torch.float64)
        input_size = [2, 3, *in_hw]
        result = decompositions.upsample_bilinear2d_aa_backward_decomposition(
            grad, list(out_hw), input_size, False
        )
        expected = torch.ops.aten._upsample_bilinear2d_aa_backward.default(
            grad, list(out_hw), input_size, False
        )
        torch.testing.assert_close(result, expected)

    def test_vec_matches_default(self):
        """The .vec overload resolves output_size and matches the default path."""
        x = torch.randn(1, 3, 8, 8, dtype=torch.float64)
        result = decompositions.upsample_bilinear2d_aa_vec_decomposition(x, [4, 4], False, None)
        expected = torch.ops.aten._upsample_bilinear2d_aa.default(x, [4, 4], False)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("out_hw", [(3, 3), (5, 5)])
    def test_align_corners_matches_aten(self, out_hw):
        """align_corners=True reproduces aten's (degenerate) AA output exactly.

        Uses the OpInfo sample shape (6,3,10,20) where this combination occurs.
        """
        x = torch.randn(6, 3, 10, 20, dtype=torch.float64)
        result = decompositions.upsample_bilinear2d_aa_decomposition(x, list(out_hw), True)
        expected = torch.ops.aten._upsample_bilinear2d_aa.default(x, list(out_hw), True)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten._upsample_bilinear2d_aa.default,
            torch.ops.aten._upsample_bilinear2d_aa.vec,
            torch.ops.aten._upsample_bilinear2d_aa_backward.default,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions
