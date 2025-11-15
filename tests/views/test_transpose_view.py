import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

from .view_test_utils import (
    assert_storage_shared,
    assert_view_properties,
    generate_test_shapes,
    make_test_tensor,
)


class TestTransposeView:
    """Test transpose view operation"""

    def test_transpose_2d_basic(self):
        """Test basic 2D transpose operation"""
        # Create a 2D tensor on neuron device
        x = make_test_tensor((3, 5), device="neuron")
        original_stride = x.stride()

        # Transpose dimensions 0 and 1
        y = x.transpose(0, 1)

        # Verify basic properties using utilities
        assert_view_properties(x, y, expected_shape=(5, 3), expected_stride=(1, 5))
        assert_storage_shared(x, y)

        # Verify stride is swapped
        assert original_stride == (5, 1), f"Original stride should be (5, 1), got {original_stride}"

    def test_transpose_breaks_contiguity(self):
        """Test that transpose creates non-contiguous views"""
        x = make_test_tensor((4, 6), device="neuron", contiguous=True)
        assert x.is_contiguous(), "Original tensor should be contiguous"

        # Transpose should break contiguity
        y = x.transpose(0, 1)
        assert_view_properties(x, y, expected_shape=(6, 4))
        assert not y.is_contiguous(), "Transposed tensor should NOT be contiguous"

        # Multiple transposes
        z = y.transpose(0, 1)  # Transpose back
        assert_view_properties(x, z, expected_shape=x.shape)
        assert z.is_contiguous(), "Double transpose should restore contiguity"

    def test_transpose_3d_tensors(self):
        """Test transpose on 3D tensors with different dimension pairs"""
        x = make_test_tensor((2, 3, 4), device="neuron")

        # Test different transpose combinations
        test_cases = [
            # (dim0, dim1, expected_shape, expected_stride)
            (0, 1, (3, 2, 4), (4, 12, 1)),
            (0, 2, (4, 3, 2), (1, 4, 12)),
            (1, 2, (2, 4, 3), (12, 1, 4)),
        ]

        for dim0, dim1, expected_shape, expected_stride in test_cases:
            y = x.transpose(dim0, dim1)
            assert_view_properties(
                x, y, expected_shape=expected_shape, expected_stride=expected_stride
            )
            assert_storage_shared(x, y)
            assert not y.is_contiguous(), f"transpose({dim0}, {dim1}) should break contiguity"

    def test_transpose_with_negative_dims(self):
        """Test transpose with negative dimension indices"""
        x = make_test_tensor((3, 4, 5), device="neuron")

        # Negative indices should work
        y1 = x.transpose(-1, -2)  # Swap last two dimensions
        y2 = x.transpose(1, 2)  # Should be equivalent

        assert_view_properties(x, y1, expected_shape=(3, 5, 4))
        assert_view_properties(x, y2, expected_shape=(3, 5, 4))
        assert y1.stride() == y2.stride(), "Negative and positive indices should give same stride"

    def test_transpose_preserves_storage_offset(self):
        """Test that transpose preserves storage offset for full tensors"""
        x = make_test_tensor((4, 6), device="neuron")

        y = x.transpose(0, 1)
        assert_view_properties(x, y, expected_shape=(6, 4))
        assert y.storage_offset() == x.storage_offset(), "Transpose should preserve storage offset"
        assert y.storage_offset() == 0, "Full tensor transpose should have zero offset"

    def test_transpose_same_dimension_is_noop(self):
        """Test that transposing same dimension is a no-op"""
        x = make_test_tensor((4, 4), device="neuron")

        # Transpose with same dimensions should return same tensor properties
        y = x.transpose(0, 0)

        assert_view_properties(x, y, expected_shape=x.shape, expected_stride=x.stride())
        assert_storage_shared(x, y)
        assert (
            y.is_contiguous() == x.is_contiguous()
        ), "Same-dim transpose should preserve contiguity"

    def test_transpose_method_vs_function(self):
        """Test that both transpose method and function work identically"""
        x = make_test_tensor((3, 5), device="neuron")

        # Method version
        y1 = x.transpose(0, 1)

        # Function version
        y2 = torch.transpose(x, 0, 1)

        assert_view_properties(x, y1, expected_shape=(5, 3))
        assert_view_properties(x, y2, expected_shape=(5, 3))
        assert y1.stride() == y2.stride(), "Method and function should give same stride"
        assert_storage_shared(y1, y2)

    def test_t_shorthand_for_2d(self):
        """Test that .t() works as shorthand for 2D transpose"""
        x = make_test_tensor((3, 5), device="neuron")

        y1 = x.t()
        y2 = x.transpose(0, 1)

        assert_view_properties(x, y1, expected_shape=(5, 3))
        assert_view_properties(x, y2, expected_shape=(5, 3))
        assert y1.stride() == y2.stride(), ".t() should match transpose(0, 1) stride"
        assert_storage_shared(y1, y2)

    @assert_raises(IndexError)
    def test_transpose_error_cases_dim_3(self):
        """Test error case for transpose operation - dim 3 doesn't exist"""
        x = make_test_tensor((3, 4, 5), device="neuron")

        # Out of range dimensions
        x.transpose(0, 3)  # dim 3 doesn't exist

    @assert_raises(IndexError)
    def test_transpose_error_cases_negative_dim(self):
        """Test error case for transpose operation - -4 is out of range"""
        x = make_test_tensor((3, 4, 5), device="neuron")

        x.transpose(-4, 0)  # -4 is out of range

    @pytest.mark.parametrize("shape", [s for s in generate_test_shapes() if len(s) >= 2])
    def test_transpose_various_shapes(self, shape):
        """Test transpose with various shapes"""
        x = make_test_tensor(shape, device="neuron")

        # For 2D+ tensors, test transpose of first two dimensions
        y = x.transpose(0, 1)

        # Build expected shape by swapping first two dimensions
        expected_shape = list(shape)
        expected_shape[0], expected_shape[1] = expected_shape[1], expected_shape[0]
        expected_shape = tuple(expected_shape)

        assert_view_properties(x, y, expected_shape=expected_shape)
        assert_storage_shared(x, y)

        # Transpose normally breaks contiguity, except for special cases like all 1s
        if not all(dim == 1 for dim in shape):
            assert not y.is_contiguous(), f"Transpose should break contiguity for shape {shape}"
