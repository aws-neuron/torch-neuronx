"""Tests for contiguous operation on Neuron device"""

import typing

import pytest
import torch

import torch_neuronx
import torch_neuronx._C as _C
from tests.utils.neuron_test_utils import track_neuron_ops

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


class TestStridedCopyStrideS:
    """Uniform stride-S neuron->neuron copies run on-device (no host bounce).

    A contiguous <-> stride-S lane view (any supported dtype — e.g. an
    object-collective uint8 buffer, or a ``[:, k]`` column of an ``(N, S)``
    tensor) is routed by ``nki_copy_strided_neuron_to_neuron`` to the NKI
    gather/scatter kernel instead of a per-element span copy (PR #1458 /
    MFU-415). These pin correctness (copied values + untouched lanes preserved)
    and that no CPU-bounce fallback fires. ``_C._nki_copy_strided_neuron_to_neuron``
    is the direct binding for that entry point; it's called directly because a
    plain ``dst.copy_(src)`` of a supported dtype is served by the MLIR epilogue
    and never reaches this path.

    ``rows`` covers both tiling paths: < P_MAX (tail) and >= P_MAX (main tile).
    """

    _CPU_BOUNCE_FALLBACKS: typing.ClassVar = {
        "aten::copy_[noncontiguous_scatter_cpu_bounce]",
    }

    # dtypes the kernel moves bit-exactly. Excludes bool (NKI rejects it) and
    # int64 (packed-int64 special case, covered by the packed suites).
    _DTYPES: typing.ClassVar = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]

    @staticmethod
    def _values(n, dtype):
        """Distinct test values in an explicit range that fits every tested dtype.

        ``[0, 100)`` sits inside int8's [-128, 127] and uint8's [0, 255]. A local
        seeded generator keeps the values deterministic (reproducible failures)
        without touching the global RNG that other tests in the process share.
        """
        gen = torch.Generator().manual_seed(0)
        return torch.randint(0, 100, (n,), generator=gen, dtype=torch.int64).to(dtype)

    def _assert_no_bounce(self, fallbacks):
        hit = set(fallbacks) & self._CPU_BOUNCE_FALLBACKS
        assert not hit, f"strided copy took a host round-trip: {sorted(hit)}"

    def _check_scatter(self, dtype, stride, rows, lane):
        """Contiguous src -> lane ``lane`` of an (rows, stride) dst; other lanes kept."""
        base = torch.zeros((rows, stride), dtype=dtype, device="neuron")
        other_lanes = {}  # lanes the copy must leave untouched
        for other in range(stride):
            if other == lane:
                continue
            fill = int(((other + 3) * 7) % 100)
            base[:, other] = fill
            other_lanes[other] = fill

        src = self._values(rows, dtype).to("neuron")
        dst = base[:, lane]  # (rows,), stride (stride,) — rows single-element spans
        assert not dst.is_contiguous()

        with track_neuron_ops():
            _C._nki_copy_strided_neuron_to_neuron(src, dst)
            fallbacks = torch_neuronx.get_fallback_ops()

        torch.testing.assert_close(dst.cpu(), src.cpu())
        for other, fill in other_lanes.items():
            torch.testing.assert_close(base[:, other].cpu(), torch.full((rows,), fill, dtype=dtype))
        self._assert_no_bounce(fallbacks)

    def _check_gather(self, dtype, stride, rows, lane):
        """Lane ``lane`` of an (rows, stride) src -> contiguous dst."""
        base = torch.zeros((rows, stride), dtype=dtype, device="neuron")
        base[:, lane] = self._values(rows, dtype).to("neuron")
        src = base[:, lane]  # stride (stride,)
        dst = torch.empty(rows, dtype=dtype, device="neuron")
        assert not src.is_contiguous()

        with track_neuron_ops():
            _C._nki_copy_strided_neuron_to_neuron(src, dst)
            fallbacks = torch_neuronx.get_fallback_ops()

        torch.testing.assert_close(dst.cpu(), base[:, lane].cpu())
        self._assert_no_bounce(fallbacks)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dtype_roundtrip(self, dtype):
        """Every supported dtype scatters/gathers bit-exactly (main-tile + tail)."""
        self._check_scatter(dtype, stride=3, rows=200, lane=1)
        self._check_gather(dtype, stride=3, rows=200, lane=1)

    @pytest.mark.parametrize("stride,lane", [(2, 0), (2, 1), (3, 0), (3, 2), (4, 3)])
    @pytest.mark.parametrize("rows", [8, 256])
    def test_shapes_and_lanes(self, stride, lane, rows):
        """Strides, lane offsets, and both tiling paths (tail: 8, main: 256)."""
        self._check_scatter(torch.float32, stride, rows, lane)
        self._check_gather(torch.float32, stride, rows, lane)


class TestStridedCopySpanDecomposition:
    """Strided copies the NKI kernel does not handle, decomposed into spans.

    ``nki_copy_strided_neuron_to_neuron`` — the default strided-copy entry point —
    routes uniform stride-S gather/scatter to the NKI kernel
    (``TestStridedCopyStrideS``) and delegates everything it declines — both sides
    strided, non-uniform strides, unsupported dtypes — to
    ``copy_strided_neuron_to_neuron``, which submits one stream-ordered device copy
    op per contiguous span.

    Every shape below is deliberately *not* kernel-eligible, so it exercises that
    span decomposition. Each asserts the bytes outside the destination view are
    left alone, which comparing only the view cannot detect.
    """

    def test_element_spans_preserve_dst_gaps(self):
        """No unit stride on either side: 12 single-element spans."""
        base_src = torch.arange(24, dtype=torch.float32).reshape(4, 6).to("neuron")
        src = base_src[:, ::2]  # (4, 3), strides (6, 2)

        base_dst = torch.zeros((4, 6), device="neuron")
        sentinel = torch.full((4, 3), -7.0, device="neuron")
        base_dst[:, 1::2] = sentinel  # lanes the copy must leave alone
        dst = base_dst[:, ::2]  # (4, 3), strides (6, 2)

        # Strided on both sides, so the kernel declines it (it needs one
        # contiguous side).
        assert not src.is_contiguous() and not dst.is_contiguous()

        _C._nki_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())
        torch.testing.assert_close(base_dst[:, 1::2].cpu(), sentinel.cpu())

    def test_row_spans_preserve_dst_gaps(self):
        """Unit innermost stride on both sides: 4 spans of 3 elements."""
        rows = 4
        base_dst = torch.zeros((2 * rows, 3), device="neuron")
        sentinel = torch.full((rows, 3), -7.0, device="neuron")
        base_dst[1::2] = sentinel  # rows the copy must leave alone

        src = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3).to("neuron")
        # A row slice: strides (6, 1) are not a uniform stride-S layout, so the
        # kernel declines it and the copy decomposes per row.
        dst = base_dst[::2]
        assert not dst.is_contiguous()

        _C._nki_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())
        torch.testing.assert_close(base_dst[1::2].cpu(), sentinel.cpu())

    def test_contiguous_is_single_span(self):
        """A fully contiguous copy is one bulk span."""
        src = torch.arange(20, dtype=torch.float32).reshape(4, 5).to("neuron")
        dst = torch.empty_like(src)

        _C._nki_copy_strided_neuron_to_neuron(src, dst)

        torch.testing.assert_close(dst.cpu(), src.cpu())


class TestChannelsLastCopyCount:
    """Tests to verify copy optimization reduces number of operations"""

    @pytest.mark.xfail(
        reason="C++ dispatch uses MLIR permute+clone instead of NRT strided copy "
        "for contiguous inputs"
    )
    def test_channels_last_single_nrt_call_for_contiguous_input(self, monkeypatch):
        """channels_last on contiguous input should use single strided copy, no intermediate"""
        import torch_neuronx._C as _C

        copy_calls = {"count": 0}
        orig_copy = _C._nki_copy_strided_neuron_to_neuron

        def counting_copy(src, dst):
            copy_calls["count"] += 1
            return orig_copy(src, dst)

        monkeypatch.setattr(_C, "_nki_copy_strided_neuron_to_neuron", counting_copy)

        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        assert x.is_contiguous()

        y = x.contiguous(memory_format=torch.channels_last)

        assert y.is_contiguous(memory_format=torch.channels_last)
        assert copy_calls["count"] == 1, f"Expected 1 NRT copy call, got {copy_calls['count']}"

    def test_channels_last_contiguous_input_runs_on_neuron(self):
        """channels_last on contiguous input produces correct result on neuron"""
        from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

        x = make_test_tensor((2, 3, 4, 5), device="neuron")
        assert x.is_contiguous()

        with track_neuron_ops():
            y = x.contiguous(memory_format=torch.channels_last)
            assert_op_runs_on_neuron("aten::contiguous")

        assert y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y.cpu(), x.cpu().contiguous(memory_format=torch.channels_last))

    @pytest.mark.xfail(reason="C++ dispatch clone/copy calls bypass Python monkeypatch")
    def test_channels_last_broadcast_materializes_then_copies(self, monkeypatch):
        """Broadcast input should clone first, then single strided copy"""
        import torch_neuronx._C as _C

        copy_calls = {"count": 0}
        clone_calls = {"count": 0}
        orig_copy = _C._nki_copy_strided_neuron_to_neuron

        def counting_copy(src, dst):
            copy_calls["count"] += 1
            # After materialization, src should NOT have stride=0
            assert all(s != 0 for s in src.stride()), "Source should be materialized (no stride=0)"
            return orig_copy(src, dst)

        monkeypatch.setattr(_C, "_nki_copy_strided_neuron_to_neuron", counting_copy)

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

    def test_channels_last_broadcast_runs_on_neuron(self):
        """Broadcast input channels_last produces correct result on neuron"""
        from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

        base = make_test_tensor((1, 3, 1, 1), device="neuron")
        x = base.expand(2, 3, 4, 5)
        assert any(s == 0 for s in x.stride())

        with track_neuron_ops():
            y = x.contiguous(memory_format=torch.channels_last)
            assert_op_runs_on_neuron("aten::contiguous")

        assert y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y.cpu(), x.cpu().contiguous(memory_format=torch.channels_last))


class TestSlicedViewContiguousPerformance:
    """Regression test: slice + permute should not trigger slow O(numel) NRT copy."""

    def test_sliced_permute_add_not_slow(self):
        """slice → permute → add must use fast on-device kernel, not NRT strided copy."""
        import time

        t, h, w, c, b = 15, 15, 20, 32, 2
        padded_seq = t * h * w + 50
        x = torch.randn(b, padded_seq, c, dtype=torch.bfloat16, device="neuron")
        x = x[:, : t * h * w, :]  # slice: storage > numel
        out = x.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

        # warmup
        _ = out + 1.0
        torch.neuron.synchronize()

        # timed run
        t0 = time.time()
        result = out + 1.0
        torch.neuron.synchronize()
        elapsed_ms = (time.time() - t0) * 1000

        # correctness
        ref = x.cpu().reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).float() + 1.0
        torch.testing.assert_close(result.cpu().float(), ref, atol=1e-2, rtol=1e-2)

        # performance: fast path ~1ms, slow path ~2000ms
        assert elapsed_ms < 50, (
            f"Took {elapsed_ms:.0f}ms, expected <50ms. "
            "Likely hit slow copy_strided_neuron_to_neuron fallback."
        )


class TestContiguousUnsupportedDtypes:
    """Tests for contiguous() on float64/int64 non-contiguous tensors.

    Regression tests for the bug where DowncastArgs/NeuronSafeCast makes the
    tensor contiguous during lowering, causing composite_lower to return None.
    The bug manifests when contiguous_internal is called from CopyToNonContiguous
    during CPU fallback copy-back of unsupported-dtype tensors.
    """

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_copy_into_permuted_3d(self, dtype):
        """Reproduce: CPU fallback result copied into non-contiguous Neuron tensor.

        This is the exact pattern that triggers the bug:
        1. A non-contiguous Neuron tensor (dst) with unsupported dtype exists
        2. A contiguous tensor (src) is copied into it via copy_
        3. CopyToNonContiguous calls contiguous_internal on src.permute(perm)
        4. DispatchContiguousImpl fails because DowncastArgs makes it contiguous
        """
        # Create a non-contiguous destination (simulates channels_last output)
        base = torch.arange(24, device="neuron", dtype=dtype).reshape(2, 3, 4)
        dst = base.permute(0, 2, 1)  # non-contiguous, shape (2, 4, 3)
        assert not dst.is_contiguous()

        # Create contiguous source with same logical shape
        src = torch.ones(2, 4, 3, device="neuron", dtype=dtype)

        # This triggers CopyToNonContiguous → contiguous_internal
        dst.copy_(src)

        assert dst.dtype == dtype
        torch.testing.assert_close(dst.cpu(), src.cpu())

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_upsample_like_cpu_fallback_pattern(self, dtype):
        """Simulate the upsample_bilinear2d CPU fallback copy-back pattern.

        The original failure: upsample_bilinear2d falls back to CPU, produces
        a channels_last float64 result, _copy_from_and_resize copies it back
        to a channels_last Neuron tensor.
        """
        # Channels-last Neuron tensor (the "output" that CPU fallback writes into)
        dst = torch.zeros(2, 3, 4, 4, device="neuron", dtype=dtype)
        dst = dst.contiguous(memory_format=torch.channels_last)
        assert dst.is_contiguous(memory_format=torch.channels_last)
        assert not dst.is_contiguous()  # not contiguous in standard format

        # Simulate CPU fallback result being copied back
        if dtype == torch.float64:
            cpu_result = torch.randn(2, 3, 4, 4, dtype=dtype)
        else:
            cpu_result = torch.randint(0, 100, (2, 3, 4, 4), dtype=dtype)
        cpu_result = cpu_result.contiguous(memory_format=torch.channels_last)
        src = cpu_result.to("neuron")

        dst.copy_(src)

        assert dst.dtype == dtype
        torch.testing.assert_close(dst.cpu(), cpu_result)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_copy_into_transposed_2d(self, dtype):
        """Copy into a transposed 2D destination."""
        base = torch.arange(12, device="neuron", dtype=dtype).reshape(3, 4)
        dst = base.t()  # shape (4, 3), non-contiguous
        assert not dst.is_contiguous()

        src = torch.ones(4, 3, device="neuron", dtype=dtype) * 7
        dst.copy_(src)

        assert dst.dtype == dtype
        torch.testing.assert_close(dst.cpu(), src.cpu())

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_copy_into_sliced(self, dtype):
        """Copy into a strided-slice destination."""
        base = torch.zeros(6, 4, device="neuron", dtype=dtype)
        dst = base[::2, :]  # shape (3, 4), non-contiguous
        assert not dst.is_contiguous()

        src = torch.ones(3, 4, device="neuron", dtype=dtype) * 5
        dst.copy_(src)

        assert dst.dtype == dtype
        torch.testing.assert_close(dst.cpu(), src.cpu())
        # Verify only sliced rows were written
        expected = torch.zeros(6, 4, dtype=dtype)
        expected[::2, :] = 5
        torch.testing.assert_close(base.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_copy_into_permuted_4d(self, dtype):
        """Copy into a 4D permuted destination."""
        base = torch.zeros(2, 3, 4, 5, device="neuron", dtype=dtype)
        dst = base.permute(0, 2, 3, 1)  # shape (2, 4, 5, 3), non-contiguous
        assert not dst.is_contiguous()

        src = torch.ones(2, 4, 5, 3, device="neuron", dtype=dtype) * 3
        dst.copy_(src)

        assert dst.dtype == dtype
        torch.testing.assert_close(dst.cpu(), src.cpu())

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
    def test_contiguous_direct(self, dtype):
        """Calling .contiguous() directly on a non-contiguous unsupported-dtype tensor."""
        x = torch.arange(12, device="neuron", dtype=dtype).reshape(3, 4)
        x_t = x.t()
        assert not x_t.is_contiguous()

        result = x_t.contiguous()

        assert result.is_contiguous()
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), x_t.cpu().contiguous())
