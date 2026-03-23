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

        y = x.contiguous(memory_format=torch.channels_last)
        assert y.is_contiguous(
            memory_format=torch.channels_last
        ), "Should be contiguous in channels_last format"
        torch.testing.assert_close(
            y.to("cpu"), x.to("cpu").contiguous(memory_format=torch.channels_last)
        )

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

    def test_contiguous_channels_last_is_supported(self):
        """channels_last memory format is supported on Neuron"""
        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        y = x.contiguous(memory_format=torch.channels_last)
        assert y.is_contiguous(memory_format=torch.channels_last)

    def test_contiguous_channels_last_from_broadcast(self):
        """channels_last on broadcast source materializes before strided copy"""
        # Create a broadcast view (stride=0 on expanded dims)
        base = make_test_tensor((1, 3, 1, 1), device="neuron")
        x = base.expand(2, 3, 4, 5)  # strides: (0, 1, 0, 0)
        assert any(s == 0 for s in x.stride()), "Should have broadcast dims"

        y = x.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y.cpu(), x.cpu().contiguous(memory_format=torch.channels_last))

    def test_contiguous_channels_last_3d_from_broadcast(self):
        """channels_last_3d on broadcast source materializes before strided copy"""
        base = make_test_tensor((1, 3, 1, 1, 1), device="neuron")
        x = base.expand(2, 3, 4, 5, 6)
        assert any(s == 0 for s in x.stride()), "Should have broadcast dims"

        y = x.contiguous(memory_format=torch.channels_last_3d)

        assert y.is_contiguous(memory_format=torch.channels_last_3d)
        torch.testing.assert_close(
            y.cpu(), x.cpu().contiguous(memory_format=torch.channels_last_3d)
        )

    def test_contiguous_channels_last_non_contiguous_input(self):
        """channels_last from transposed (non-contiguous) input"""
        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        x_t = x.transpose(1, 2)  # Now non-contiguous
        assert not x_t.is_contiguous()

        y = x_t.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y.cpu(), x_t.cpu().contiguous(memory_format=torch.channels_last))

    def test_fill_scalar_on_strided_view(self):
        """fill_ on non-contiguous view should work correctly"""
        base = torch.zeros((4, 5), device="neuron")
        view = base[::2, :]  # shape (2, 5), non-contiguous
        assert not view.is_contiguous()

        view.fill_(42.0)

        expected = torch.zeros((4, 5))
        expected[::2, :] = 42.0
        torch.testing.assert_close(base.cpu(), expected)

    def test_fill_scalar_on_transposed_view(self):
        """fill_ on transposed tensor should work correctly"""
        base = torch.zeros((3, 4), device="neuron")
        view = base.t()  # shape (4, 3), non-contiguous
        assert not view.is_contiguous()

        view.fill_(7.0)

        torch.testing.assert_close(base.cpu(), torch.full((3, 4), 7.0))

    def test_contiguous_channels_last_partial_broadcast(self):
        """channels_last with some (not all) broadcast dims"""
        base = make_test_tensor((2, 3, 1, 5), device="neuron")
        x = base.expand(2, 3, 4, 5)  # strides: (15, 5, 0, 1)

        y = x.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y.cpu(), x.cpu().contiguous(memory_format=torch.channels_last))


class TestNrtStridedCopy:
    """Tests for _nrt_copy_strided_neuron_to_neuron"""

    def test_nrt_strided_copy_contiguous_to_contiguous(self):
        """Fast path: both contiguous uses single bulk copy"""
        import torch_neuronx._C as _C

        src = make_test_tensor((4, 5), device="neuron")
        dst = torch.empty_like(src)

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_nrt_strided_copy_transposed_to_contiguous(self):
        """Copy from transposed source to contiguous dest"""
        import torch_neuronx._C as _C

        base = make_test_tensor((3, 4), device="neuron")
        src = base.t()  # shape (4, 3), strides (1, 4)
        dst = torch.empty((4, 3), device="neuron")

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_nrt_strided_copy_contiguous_to_strided(self):
        """Copy from contiguous source to strided dest (channels_last pattern)"""
        import torch_neuronx._C as _C

        src = make_test_tensor((2, 3, 4, 5), device="neuron")
        dst = torch.empty_like(src, memory_format=torch.channels_last)

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)

        assert dst.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_nrt_strided_copy_strided_to_strided(self):
        """Both source and dest are strided"""
        import torch_neuronx._C as _C

        base_src = make_test_tensor((4, 6), device="neuron")
        src = base_src[:, ::2]  # shape (4, 3), strided

        base_dst = torch.zeros((4, 6), device="neuron")
        dst = base_dst[:, ::2]  # shape (4, 3), strided

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_nrt_strided_copy_broadcast_source(self):
        """Copy from broadcast source (stride=0)"""
        import torch_neuronx._C as _C

        base = make_test_tensor((1, 3), device="neuron")
        src = base.expand(4, 3)  # strides (0, 1)
        dst = torch.empty((4, 3), device="neuron")

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())

    def test_nrt_strided_copy_empty_tensor(self):
        """Empty tensor should be no-op"""
        import torch_neuronx._C as _C

        src = torch.empty((0, 3), device="neuron")
        dst = torch.empty((0, 3), device="neuron")

        _C._nrt_copy_strided_neuron_to_neuron(src, dst)  # Should not crash

    def test_nrt_strided_copy_shape_mismatch_raises(self):
        """Mismatched shapes should raise"""
        import torch_neuronx._C as _C

        src = make_test_tensor((3, 4), device="neuron")
        dst = torch.empty((4, 3), device="neuron")

        with pytest.raises(RuntimeError):
            _C._nrt_copy_strided_neuron_to_neuron(src, dst)

    def test_nrt_strided_copy_dtype_mismatch_raises(self):
        """Mismatched dtypes should raise"""
        import torch_neuronx._C as _C

        src = make_test_tensor((3, 4), device="neuron", dtype=torch.float32)
        dst = torch.empty((3, 4), device="neuron", dtype=torch.float16)

        with pytest.raises(RuntimeError):
            _C._nrt_copy_strided_neuron_to_neuron(src, dst)


class TestChannelsLastCopyCount:
    """Tests to verify copy optimization reduces number of operations"""

    def test_channels_last_single_nrt_call_for_contiguous_input(self, monkeypatch):
        """channels_last on contiguous input should use single strided copy, no intermediate"""
        import torch_neuronx._C as _C

        copy_calls = {"count": 0}
        orig_copy = _C._nrt_copy_strided_neuron_to_neuron

        def counting_copy(src, dst):
            copy_calls["count"] += 1
            return orig_copy(src, dst)

        monkeypatch.setattr(_C, "_nrt_copy_strided_neuron_to_neuron", counting_copy)

        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        assert x.is_contiguous()

        y = x.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        assert copy_calls["count"] == 1, f"Expected 1 NRT copy call, got {copy_calls['count']}"

    def test_channels_last_broadcast_materializes_then_copies(self, monkeypatch):
        """Broadcast input should clone first, then single strided copy"""
        import torch_neuronx._C as _C

        copy_calls = {"count": 0}
        clone_calls = {"count": 0}
        orig_copy = _C._nrt_copy_strided_neuron_to_neuron

        def counting_copy(src, dst):
            copy_calls["count"] += 1
            # After materialization, src should NOT have stride=0
            assert all(s != 0 for s in src.stride()), "Source should be materialized (no stride=0)"
            return orig_copy(src, dst)

        monkeypatch.setattr(_C, "_nrt_copy_strided_neuron_to_neuron", counting_copy)

        # Patch clone to count materializations
        orig_clone = torch.Tensor.clone

        def counting_clone(self, *args, **kwargs):
            clone_calls["count"] += 1
            return orig_clone(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "clone", counting_clone)

        base = make_test_tensor((1, 3, 1, 1), device="neuron")
        x = base.expand(2, 3, 4, 5)
        assert any(s == 0 for s in x.stride())

        y = x.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        assert clone_calls["count"] >= 1, "Broadcast source should be cloned"
        assert copy_calls["count"] == 1, f"Expected 1 NRT copy call, got {copy_calls['count']}"
