import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

from .view_test_utils import (
    assert_storage_shared,
    assert_view_properties,
    check_view_semantics,
    make_test_tensor,
)


class TestBasicView:
    """Test basic view operation"""

    def test_basic_view_works(self):
        """Test that basic view operation works and maintains storage sharing"""
        # Create a tensor on neuron device
        x = make_test_tensor((4, 4), device="neuron")
        original_numel = x.numel()

        # Create a simple view - reshape to 1D
        y = x.view(16)

        # Verify view properties
        assert_view_properties(x, y, expected_shape=(16,))
        assert (
            y.numel() == original_numel
        ), f"Number of elements changed: {original_numel} -> {y.numel()}"
        assert_storage_shared(x, y)

        # Test another view shape
        z = x.view(2, 8)
        assert_view_properties(x, z, expected_shape=(2, 8))
        assert_storage_shared(x, z)

        # Test view with -1 inference
        w = x.view(-1, 4)
        assert_view_properties(x, w, expected_shape=(4, 4))
        assert_storage_shared(x, w)

        # Test view semantics - modifications should propagate
        check_view_semantics(x, y)

    def test_view_maintains_contiguity(self):
        """Test that simple views maintain contiguous property"""
        x = make_test_tensor((4, 4), device="neuron", contiguous=True)
        assert x.is_contiguous(), "Original tensor should be contiguous"

        # Simple reshape maintains contiguity
        y = x.view(16)
        assert_view_properties(x, y, expected_shape=(16,))
        assert y.is_contiguous(), "Reshaped view should remain contiguous"

        # Another contiguous view
        z = x.view(2, 8)
        assert_view_properties(x, z, expected_shape=(2, 8))
        assert z.is_contiguous(), "Reshaped view should remain contiguous"

    def test_view_stride_properties(self):
        """Test that view operations have correct strides"""
        x = make_test_tensor((4, 4), device="neuron")
        original_stride = x.stride()

        # View to 1D - stride should be (1,)
        y = x.view(16)
        assert_view_properties(x, y, expected_shape=(16,), expected_stride=(1,))

        # View to different 2D shape
        z = x.view(2, 8)
        assert_view_properties(x, z, expected_shape=(2, 8), expected_stride=(8, 1))

        # View with same shape should maintain stride
        w = x.view(4, 4)
        assert_view_properties(x, w, expected_shape=(4, 4), expected_stride=original_stride)

    def test_view_storage_offset(self):
        """Test that basic views maintain zero storage offset"""
        x = make_test_tensor((4, 4), device="neuron")

        # Basic views should have same storage offset as original
        y = x.view(16)
        assert_view_properties(x, y, expected_shape=(16,))
        assert (
            y.storage_offset() == x.storage_offset()
        ), "Storage offset should not change for basic view"
        assert y.storage_offset() == 0, "Storage offset should be 0 for basic view"

    @assert_raises(RuntimeError, match="is invalid for input of size")
    def test_view_with_incompatible_size_15_fails(self):
        """Test that view with incompatible size raises error - can't reshape 16 elements to 15"""
        x = make_test_tensor((4, 4), device="neuron")

        # This should fail - can't reshape 16 elements to 15
        x.view(15)

    @assert_raises(RuntimeError, match="is invalid for input of size")
    def test_view_with_incompatible_size_3x6_fails(self):
        """Test that view with incompatible size raises error - can't reshape 16 elements to 3x6"""
        x = make_test_tensor((4, 4), device="neuron")

        # This should fail - can't reshape 16 elements to 3x6=18
        x.view(3, 6)
