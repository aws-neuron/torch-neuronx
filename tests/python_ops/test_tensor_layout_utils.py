"""Tests for tensor layout utilities"""

import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.python_ops.tensor_layout_utils import (
    KernelType,
    collapse_for_bulk_copy,
    get_collapsed_rank,
    get_contiguous_block_size,
    has_unit_stride,
    select_kernel,
)


class TestCollapseForBulkCopy:
    """Test the collapse_for_bulk_copy function"""

    @assert_raises(RuntimeError, match="Unexpected rank-1 contiguous tensor")
    def test_contiguous_tensor(self):
        """Test collapsing a standard contiguous tensor"""
        t = torch.randn(2, 3, 4)  # Shape: (2, 3, 4), Strides: (12, 4, 1)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Should collapse to a single dimension
        assert collapsed_shape == (24,)
        assert collapsed_strides == (1,)
        assert get_collapsed_rank(collapsed_shape) == 1
        assert has_unit_stride(collapsed_strides)
        # Should raise error - rank-1 contiguous tensors should not reach kernel selection
        select_kernel(collapsed_shape, collapsed_strides)

    def test_transposed_tensor(self):
        """Test collapsing a transposed tensor"""
        t = torch.randn(2, 3, 4).permute(1, 0, 2)  # Shape: (3, 2, 4)
        # Original strides would be (4, 12, 1)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Cannot collapse these dimensions due to non-standard strides
        # Middle dimension has stride 12, which is not 3*4
        assert collapsed_shape == (3, 2, 4)
        assert collapsed_strides == (4, 12, 1)
        assert get_collapsed_rank(collapsed_shape) == 3
        assert has_unit_stride(collapsed_strides)
        assert select_kernel(collapsed_shape, collapsed_strides) == KernelType.RANK3

    def test_broadcast_dimension(self):
        """Test collapsing with broadcast (size-1) dimensions"""
        t = torch.randn(2, 3, 4)[:, :1, :]  # Shape: (2, 1, 4), middle dim is size 1
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # The broadcast dimension (size 1) gets merged with the last dimension
        # First dimension stays separate
        assert collapsed_shape == (2, 4)
        assert collapsed_strides == (12, 1)
        assert get_collapsed_rank(collapsed_shape) == 2
        assert select_kernel(collapsed_shape, collapsed_strides) == KernelType.RANK2

    def test_diagonal_view(self):
        """Test collapsing a diagonal view (non-unit strides)"""
        t = torch.eye(4).diagonal()  # Shape: (4,), Stride: (5,)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Cannot collapse anything, stays as-is
        assert collapsed_shape == (4,)
        assert collapsed_strides == (5,)
        assert get_collapsed_rank(collapsed_shape) == 1
        assert not has_unit_stride(collapsed_strides)
        assert select_kernel(collapsed_shape, collapsed_strides) == KernelType.GENERIC

    @assert_raises(RuntimeError, match="Unexpected rank-1 contiguous tensor")
    def test_multiple_broadcast_dims(self):
        """Test multiple consecutive broadcast dimensions"""
        t = torch.randn(3, 1, 1, 4)  # Two consecutive size-1 dims
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Should collapse to (3, 4) since middle dims are broadcast
        assert collapsed_shape == (12,)
        assert collapsed_strides == (1,)
        assert get_collapsed_rank(collapsed_shape) == 1
        # Should raise error - rank-1 contiguous tensors should not reach kernel selection
        select_kernel(collapsed_shape, collapsed_strides)

    def test_non_contiguous_slice(self):
        """Test a slice that breaks contiguity"""
        t = torch.randn(4, 6)[:, ::2]  # Shape: (4, 3), Strides: (6, 2)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Cannot collapse these dimensions
        assert collapsed_shape == (4, 3)
        assert collapsed_strides == (6, 2)
        assert get_collapsed_rank(collapsed_shape) == 2
        assert not has_unit_stride(collapsed_strides)
        assert select_kernel(collapsed_shape, collapsed_strides) == KernelType.GENERIC

    @assert_raises(RuntimeError, match="Unexpected rank-1 contiguous tensor")
    def test_empty_tensor(self):
        """Test empty tensor"""
        t = torch.empty(0, 3, 4)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Should handle empty tensor gracefully
        assert collapsed_shape == (0,)
        assert get_collapsed_rank(collapsed_shape) == 1
        # Should raise error - rank-1 contiguous tensors should not reach kernel selection
        select_kernel(collapsed_shape, collapsed_strides)

    @assert_raises(RuntimeError, match="Unexpected scalar tensor")
    def test_scalar_tensor(self):
        """Test scalar tensor"""
        t = torch.tensor(42.0)
        collapsed_shape, collapsed_strides = collapse_for_bulk_copy(t)

        # Scalar should return empty tuples
        assert collapsed_shape == ()
        assert collapsed_strides == ()
        assert get_collapsed_rank(collapsed_shape) == 0
        # Should raise error - scalar tensors should not reach kernel selection
        select_kernel(collapsed_shape, collapsed_strides)


class TestGetContiguousBlockSize:
    """Test the get_contiguous_block_size function"""

    def test_fully_contiguous(self):
        """Test fully contiguous tensor"""
        shape = [2, 3, 4]
        strides = [12, 4, 1]
        block_size = get_contiguous_block_size(shape, strides)
        assert block_size == 24  # Can copy all elements at once

    def test_partial_contiguous(self):
        """Test partially contiguous tensor"""
        # Like a transposed tensor (3, 2, 4) from (2, 3, 4)
        shape = [3, 2, 4]
        strides = [4, 12, 1]
        block_size = get_contiguous_block_size(shape, strides)
        assert block_size == 4  # Can only copy last dim (stride 12 != 4*1)

    def test_non_contiguous(self):
        """Test non-contiguous tensor"""
        shape = [4, 3]
        strides = [6, 2]  # Strided view
        block_size = get_contiguous_block_size(shape, strides)
        assert block_size == 0  # Cannot do block copy

    def test_single_element(self):
        """Test single element tensor"""
        shape = [1, 1, 1]
        strides = [1, 1, 1]
        block_size = get_contiguous_block_size(shape, strides)
        assert block_size == 1

    def test_last_dim_non_unit_stride(self):
        """Test when last dimension doesn't have unit stride"""
        shape = [2, 3, 4]
        strides = [24, 8, 2]  # Last stride is 2, not 1
        block_size = get_contiguous_block_size(shape, strides)
        assert block_size == 0  # Cannot do block copy
