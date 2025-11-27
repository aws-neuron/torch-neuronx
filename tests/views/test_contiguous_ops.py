"""Tests for contiguous operation on Neuron device"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

from .view_test_utils import (
    assert_storage_shared,
    generate_test_shapes,
    make_test_tensor,
)


class TestContiguousOps:
    """Test contiguous operation and is_contiguous checks"""

    def test_contiguous_on_contiguous_tensor(self):
        """Test contiguous() on already contiguous tensor returns self"""
        x = make_test_tensor((4, 3), device="neuron")
        assert x.is_contiguous(), "Test tensor should be contiguous"

        # Get data pointer before contiguous call
        original_data_ptr = x.data_ptr()

        # Call contiguous on already contiguous tensor
        y = x.contiguous()

        # Should return self (same tensor object)
        assert x is y, "contiguous() on contiguous tensor should return self"
        assert y.data_ptr() == original_data_ptr, "Data pointer should not change"

    def test_contiguous_on_non_contiguous_tensor(self):
        """Test contiguous() on non-contiguous tensor creates a copy"""
        x = make_test_tensor((4, 3), device="neuron", contiguous=False)
        assert not x.is_contiguous(), "Test tensor should be non-contiguous"

        # Get original shape and stride
        original_shape = x.shape
        original_values = x.clone()

        # Make contiguous
        y = x.contiguous()

        # Verify properties
        assert y.is_contiguous(), "Result should be contiguous"
        assert y.shape == original_shape, "Shape should be preserved"
        assert torch.allclose(y, original_values), "Values should be preserved"

        # Should be a copy, not a view
        assert not y._is_view(), "contiguous() on non-contiguous should create a copy"

        # Modifying y should not affect x
        y[0, 0] = -999
        assert x[0, 0].item() != -999, "Modifying copy should not affect original"

    def test_is_contiguous_basic(self):
        """Test is_contiguous() for various tensor layouts"""
        # Contiguous tensor
        x = make_test_tensor((3, 4), device="neuron")
        assert x.is_contiguous(), "Regular tensor should be contiguous"

        # Non-contiguous via transpose
        y = x.transpose(0, 1)
        assert not y.is_contiguous(), "Transposed tensor should not be contiguous"

        # Non-contiguous via slice with step
        z = x[::2, :]
        assert not z.is_contiguous(), "Strided slice should not be contiguous"

        # More complex non-contiguous
        w = x[:, ::2]
        # Until we decide if this is going to be contiguous or not,
        # just verify is_contiguous returns a boolean
        assert isinstance(w.is_contiguous(), bool), "is_contiguous should return bool"

    def test_contiguous_default_memory_format(self):
        """Test contiguous with default memory format (tests RANK3 kernel path)"""
        x = make_test_tensor((2, 3, 4, 5), device="neuron", contiguous=False)

        # Default contiguous (should be same as memory_format=torch.contiguous_format)
        y1 = x.contiguous()
        y2 = x.contiguous(memory_format=torch.contiguous_format)

        assert y1.is_contiguous(), "Default contiguous should work"
        assert y2.is_contiguous(), "Explicit contiguous format should work"
        assert torch.allclose(y1, y2), "Both methods should produce same values"

    def test_contiguous_preserves_dtype(self):
        """Test that contiguous preserves dtype"""
        for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.int32]:
            x = make_test_tensor((3, 4), device="neuron", dtype=dtype, contiguous=False)
            y = x.contiguous()

            assert y.dtype == x.dtype, f"contiguous should preserve dtype {dtype}"
            assert y.is_contiguous(), "Result should be contiguous"

    def test_contiguous_empty_tensor(self):
        """Test contiguous on empty tensor"""
        x = torch.empty(0, 3, device="neuron")
        assert x.is_contiguous(), "Empty tensor should be contiguous"

        y = x.contiguous()
        assert x is y, "contiguous() on empty contiguous tensor should return self"

    def test_contiguous_scalar(self):
        """Test contiguous on scalar tensor"""
        x = torch.tensor(42.0, device="neuron")
        assert x.is_contiguous(), "Scalar should be contiguous"

        y = x.contiguous()
        assert x is y, "contiguous() on scalar should return self"

    def test_contiguous_stride_patterns(self):
        """Test is_contiguous with various stride patterns"""
        # Create base tensor
        x = make_test_tensor((4, 6), device="neuron")

        # Test cases with expected contiguity based on CPU behavior
        test_cases = [
            (x, True, "Original tensor"),
            (x.t(), False, "Transposed tensor"),
            (x[2:, :], True, "Row slice"),
            (x[:, 2:], False, "Column slice"),
            (x[::2, :], False, "Strided rows"),
            (x[:, ::2], False, "Strided columns"),
            (x.t().contiguous().t(), False, "Double transpose"),
        ]

        for tensor, expected_contiguous, description in test_cases:
            if expected_contiguous:
                assert tensor.is_contiguous(), f"{description} should be contiguous"
            else:
                assert not tensor.is_contiguous(), f"{description} should not be contiguous"

    def test_contiguous_with_views(self):
        """Test interaction between contiguous and view operations"""
        x = make_test_tensor((4, 3), device="neuron")

        # Create non-contiguous view
        y = x.t()
        assert not y.is_contiguous(), "Transpose should be non-contiguous"

        # Make it contiguous
        z = y.contiguous()
        assert z.is_contiguous(), "Result should be contiguous"
        assert z.shape == (3, 4), "Shape should match transposed shape"

        # Now we can do view operations that require contiguity
        w = z.view(-1)
        assert w.shape == (12,), "View after contiguous should work"

    @pytest.mark.xfail(reason="Copying into non-contiguous tensors is not supported")
    def test_contiguous_gradients(self):
        """Test that gradients flow through contiguous"""
        x = make_test_tensor((3, 4), device="neuron", dtype=torch.float32, contiguous=False)
        x.requires_grad_(True)

        # Make contiguous and perform operation
        y = x.contiguous()
        z = y.sum()

        # Check gradient computation
        z.backward()
        assert x.grad is not None, "Gradient not computed"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert torch.allclose(x.grad, torch.ones_like(x)), "Gradient values incorrect"

    @pytest.mark.parametrize("shape", generate_test_shapes())
    def test_contiguous_various_shapes(self, shape):
        """Test contiguous with various tensor shapes"""
        if shape == ():
            # Scalar tensor
            x = torch.tensor(42.0, device="neuron")
        else:
            x = make_test_tensor(shape, device="neuron")

        # Test on contiguous tensor
        assert x.is_contiguous(), f"Fresh tensor with shape {shape} should be contiguous"
        y = x.contiguous()
        assert x is y, "contiguous() on contiguous should return self"

        # Test on non-contiguous tensor (if possible)
        if len(shape) >= 2:
            x_noncontig = x.transpose(0, 1)
            # Note: Transposing tensors where all dimensions are 1 keeps them contiguous
            if not all(s == 1 for s in shape):
                assert (
                    not x_noncontig.is_contiguous()
                ), f"Transposed {shape} should be non-contiguous"
            y_noncontig = x_noncontig.contiguous()
            assert y_noncontig.is_contiguous(), "Result should be contiguous"
            assert torch.allclose(y_noncontig.cpu(), x_noncontig.cpu()), "Values should match"

    def test_contiguous_memory_format_channels_last(self):
        """Test contiguous with channels_last memory format"""
        # This is primarily for 4D tensors (NCHW format)
        x = make_test_tensor((2, 3, 4, 5), device="neuron")

        try:
            # Try to make it channels_last contiguous
            y = x.contiguous(memory_format=torch.channels_last)
            # If supported, verify it's contiguous in some format
            assert y.is_contiguous() or y.is_contiguous(
                memory_format=torch.channels_last
            ), "Should be contiguous in some format"
        except (RuntimeError, NotImplementedError):
            pytest.skip("channels_last memory format not supported")

    def test_contiguous_force_copy(self):
        """Test that contiguous can be forced to make a copy"""
        x = make_test_tensor((3, 4), device="neuron")
        assert x.is_contiguous(), "Test tensor should be contiguous"

        # Even though x is contiguous, we can clone it
        y = x.clone()

        # Verify it's a copy
        assert y.is_contiguous(), "Clone should be contiguous"
        assert y.data_ptr() != x.data_ptr(), "Clone should have different data pointer"

        # Modifications don't affect original
        y[0, 0] = -999
        assert x[0, 0].item() != -999, "Modifying clone should not affect original"

    @assert_raises(
        RuntimeError, match="Neuron tensors only support contiguous or preserve memory format"
    )
    def test_contiguous_rejects_unsupported_memory_format(self):
        """Reject unsupported memory formats with canonical message"""
        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        x.contiguous(memory_format=torch.channels_last)
