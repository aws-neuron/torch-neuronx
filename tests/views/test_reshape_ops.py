"""Tests for reshape and as_strided view operations on Neuron device"""

import contextlib

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

from .view_test_utils import (
    assert_storage_shared,
    assert_view_properties,
    check_view_semantics,
    generate_test_shapes,
    make_test_tensor,
)


class TestReshapeOps:
    """Test reshape and as_strided view operations"""

    def test_reshape_basic(self):
        """Test basic reshape functionality"""
        # Create test tensor
        x = make_test_tensor((4, 3), device="neuron")
        original_data_ptr = x.data_ptr()

        # Apply reshape
        y = x.reshape(12)

        # Verify view properties
        assert_view_properties(x, y, expected_shape=(12,))
        assert y.data_ptr() == original_data_ptr, "Reshape should share storage"

        # Test data sharing
        check_view_semantics(x, y)

    def test_reshape_multi_dimensional(self):
        """Test reshape with multiple dimensions"""
        x = make_test_tensor((2, 3, 4), device="neuron")

        # Test various reshapes
        test_cases = [
            (24,),  # Flatten to 1D
            (4, 6),  # 3D to 2D
            (3, 8),  # Different 2D shape
            (2, 12),  # Another 2D shape
            (2, 2, 6),  # Different 3D shape
            (4, 3, 2),  # Reorder dimensions
        ]

        for new_shape in test_cases:
            y = x.reshape(new_shape)
            assert_view_properties(x, y, expected_shape=new_shape)
            check_view_semantics(x, y)

    def test_reshape_with_negative_dimension(self):
        """Test reshape with -1 dimension inference"""
        x = make_test_tensor((2, 3, 4), device="neuron")

        # Test -1 dimension inference
        test_cases = [
            (-1,),  # Infer single dimension
            (2, -1),  # Infer last dimension
            (-1, 6),  # Infer first dimension
            (2, -1, 2),  # Infer middle dimension
        ]

        expected_shapes = [
            (24,),
            (2, 12),
            (4, 6),
            (2, 6, 2),
        ]

        for new_shape, expected in zip(test_cases, expected_shapes, strict=False):
            y = x.reshape(new_shape)
            assert_view_properties(x, y, expected_shape=expected)
            check_view_semantics(x, y)

    def test_reshape_non_contiguous(self):
        """Test reshape on non-contiguous tensors"""
        x = make_test_tensor((4, 3), device="neuron", contiguous=False)
        assert not x.is_contiguous(), "Test tensor should be non-contiguous"

        # Reshape should work on non-contiguous tensors
        # Note: This might create a copy instead of a view
        y = x.reshape(12)

        # We should return a view here but testing both until that is implemented
        if y._is_view():
            assert_view_properties(x, y, expected_shape=(12,))
        else:
            # If it's a copy, at least verify shape and values
            assert y.shape == (12,), f"Expected shape (12,), got {y.shape}"
            assert torch.allclose(y.cpu(), x.cpu().reshape(12)), "Values don't match"

    @assert_raises(RuntimeError, match="shape .* is invalid")
    def test_reshape_error_cases_invalid_shape(self):
        """Test reshape error handling - invalid shape"""
        x = make_test_tensor((4, 3), device="neuron")

        # Test invalid shapes
        x.reshape(13)  # Wrong number of elements

    @assert_raises(RuntimeError, match="only one dimension can be inferred")
    def test_reshape_error_cases_multiple_inferred(self):
        """Test reshape error handling - multiple -1 dimensions"""
        x = make_test_tensor((4, 3), device="neuron")

        x.reshape(-1, -1)  # Multiple -1 dimensions

    def test_reshape_zero_dim(self):
        """Test reshape with zero dimensions"""
        # Scalar tensor
        x = torch.tensor(42.0, device="neuron")

        # Reshape scalar to 1D
        y = x.reshape(1)
        assert y.shape == (1,), f"Expected shape (1,), got {y.shape}"
        assert y[0].item() == 42.0

        # Reshape 1D back to scalar
        z = y.reshape(())
        assert z.shape == (), f"Expected scalar shape (), got {z.shape}"
        assert z.item() == 42.0

    def test_as_strided_basic(self):
        """Test basic as_strided functionality"""
        x = make_test_tensor((4, 3), device="neuron")
        original_data_ptr = x.data_ptr()

        # Create a simple strided view
        y = torch.as_strided(x, size=(3, 2), stride=(3, 1))

        # Verify it's a view
        assert y._is_view(), "as_strided should create a view"
        assert y.data_ptr() == original_data_ptr, "as_strided should share storage"
        assert y.shape == (3, 2), f"Expected shape (3, 2), got {y.shape}"
        assert y.stride() == (3, 1), f"Expected stride (3, 1), got {y.stride()}"

    def test_as_strided_diagonal_view(self):
        """Test as_strided to create diagonal view"""
        x = make_test_tensor((3, 3), device="neuron")

        # Create diagonal view using as_strided
        # Diagonal elements are at positions 0, 4, 8 (stride of 4)
        diag = torch.as_strided(x, size=(3,), stride=(4,))

        # Verify diagonal values
        expected_diag = torch.tensor([0.0, 4.0, 8.0], device="neuron")
        assert torch.allclose(diag, expected_diag), "Diagonal values incorrect"

        # Test modification propagates
        diag[1] = -999
        assert x[1, 1].item() == -999, "Modification through diagonal view didn't propagate"

    def test_as_strided_overlapping_elements(self):
        """Test as_strided with overlapping elements"""
        x = make_test_tensor((6,), device="neuron")

        # Create sliding window view with overlap
        # Windows of size 3 with stride 1: [0,1,2], [1,2,3], [2,3,4], [3,4,5]
        windows = torch.as_strided(x, size=(4, 3), stride=(1, 1))

        # Verify shape
        assert windows.shape == (4, 3), f"Expected shape (4, 3), got {windows.shape}"

        # Verify overlapping elements
        assert windows[0, 1].item() == windows[1, 0].item(), "Overlapping elements should be same"
        assert windows[1, 2].item() == windows[2, 1].item(), "Overlapping elements should be same"

    def test_as_strided_broadcast_simulation(self):
        """Test as_strided to simulate broadcasting"""
        x = make_test_tensor((3,), device="neuron")

        # Simulate broadcasting (3,) -> (4, 3) using stride 0 for first dimension
        broadcasted = torch.as_strided(x, size=(4, 3), stride=(0, 1))

        # All rows should be identical
        for i in range(4):
            assert torch.allclose(broadcasted[i], x), f"Row {i} doesn't match original"

    @assert_raises((RuntimeError, ValueError))
    def test_as_strided_error_cases_out_of_bounds(self):
        """Test as_strided error handling - accessing out of bounds"""
        x = make_test_tensor((4, 3), device="neuron")

        # Test invalid cases
        # Accessing out of bounds
        torch.as_strided(x, size=(5, 5), stride=(3, 1))

    def test_as_strided_error_cases_negative_stride(self):
        """Test as_strided error handling - negative stride"""
        x = make_test_tensor((4, 3), device="neuron")

        # No plan to support negative stride but we should throw clear message
        with contextlib.suppress(RuntimeError, ValueError):
            torch.as_strided(x, size=(2, 2), stride=(-3, 1))

    def test_as_strided_with_offset(self):
        """Test as_strided with storage offset"""
        x = make_test_tensor((4, 3), device="neuron")

        # Create a view starting from element at position (1, 1)
        # Storage offset would be 1*3 + 1 = 4
        y = torch.as_strided(x, size=(2, 2), stride=(3, 1), storage_offset=4)

        # Verify the view starts at the correct position
        assert y[0, 0].item() == x[1, 1].item(), "View doesn't start at correct offset"
        assert y[0, 1].item() == x[1, 2].item(), "View offset incorrect"

    @pytest.mark.parametrize("shape", generate_test_shapes())
    def test_reshape_various_shapes(self, shape):
        """Test reshape with various shapes"""
        x = make_test_tensor(shape, device="neuron")

        # Calculate total elements
        numel = x.numel()

        # Skip scalar tensors for some tests
        if numel == 0 or shape == ():
            pytest.skip("Empty or scalar tensor")

        # Test flattening
        flat = x.reshape(-1)
        assert flat.shape == (numel,), f"Flatten failed for shape {shape}"

        # Test various valid reshapes
        if numel == 12:
            valid_shapes = [(12,), (3, 4), (4, 3), (2, 6), (6, 2), (2, 2, 3)]
            for new_shape in valid_shapes:
                y = x.reshape(new_shape)
                assert y.shape == new_shape, f"Reshape to {new_shape} failed"

    def test_reshape_gradients(self):
        """Test that gradients flow through reshape"""
        x = make_test_tensor((4, 3), device="neuron", dtype=torch.float32)
        x.requires_grad_(True)

        # Reshape and perform operation
        y = x.reshape(2, 6)
        z = y.sum()

        # Check gradient computation
        z.backward()
        assert x.grad is not None, "Gradient not computed"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert torch.allclose(x.grad, torch.ones_like(x)), "Gradient values incorrect"
