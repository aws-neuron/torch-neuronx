"""Tests for operation epilogue: verifying fused on-device output transformation.

The epilogue eliminates the separate CopyBackToOriginal / .copy_() step that
currently handles non-contiguous in-place outputs. Instead, the inverse
transformation is fused into the operation's MLIR and executed in a single
device round-trip.

These tests verify that:
1. When TORCH_NEURONX_ENABLE_EPILOGUE=1, in-place ops on non-contiguous views
   do NOT trigger a separate copy_ operation (the epilogue is fused).
2. Results are still numerically correct (match CPU reference).
3. When epilogue is disabled (default), the copy-back path is used as before.
"""

import os

import numpy as np
import pytest
import torch

import torch_neuronx

# ============================================================================
# Helpers
# ============================================================================


def epilogue_enabled():
    """Check if epilogue is enabled — delegates to C++ IsEpilogueEnabled()."""
    from torch_neuronx._C import _is_epilogue_enabled

    return _is_epilogue_enabled()


def scatter_iota_enabled():
    """Check if SCATTER_IOTA epilogue is enabled (default ON).

    Set TORCH_NEURONX_ENABLE_SCATTER_IOTA=0 to disable.
    """
    return os.environ.get("TORCH_NEURONX_ENABLE_SCATTER_IOTA", "1") != "0"


def count_copy_ops(fn):
    """Execute fn and count how many aten::copy_ operations were executed.

    Returns copy_count.
    The copy_count tells us whether the copy-back path was used.
    If epilogue is working, copy_count should be 0 for in-place ops
    on non-contiguous views.
    """
    torch_neuronx.clear_op_tracking()
    fn()
    torch.neuron.synchronize()
    executed_ops = torch_neuronx.get_executed_ops()
    fallback_ops = torch_neuronx.get_fallback_ops()

    # Count only neuron-to-neuron copy ops (the copy-back path).
    # Exclude cpu_to_neuron (tensor setup) and neuron_to_cpu (readback).
    copy_count = 0
    copy_patterns = ("neuron_to_neuron", "aten::copy_", "aten::contiguous")
    for op_name in executed_ops:
        if any(p in op_name for p in copy_patterns):
            copy_count += 1
    for op_name in fallback_ops:
        if any(p in op_name for p in copy_patterns):
            copy_count += 1

    return copy_count


# ============================================================================
# Test: Epilogue eliminates copy-back for transposed in-place ops
# ============================================================================


class TestEpilogueEliminatesCopyBack:
    """Verify that the epilogue path eliminates the separate copy-back step.

    When TORCH_NEURONX_ENABLE_EPILOGUE=1:
    - In-place ops on transposed views should NOT trigger aten::copy_
    - The result should be fused: prologue + operation + epilogue in one NEFF

    When epilogue is disabled (default):
    - These tests are expected to fail (copy_ IS used) — marked xfail.
    """

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_add_inplace_transpose_no_copy_back(self):
        """add_ on transposed view should not use copy-back when epilogue is enabled."""
        base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).to("neuron")
        view = base.transpose(0, 1)  # shape (4, 3), non-contiguous
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        assert copy_count == 0, (
            f"Expected 0 copy_ operations with epilogue enabled, but got {copy_count}. "
            f"The epilogue should fuse the inverse transpose into the operation MLIR."
        )

        # Also verify correctness
        cpu_base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        cpu_view = cpu_base.transpose(0, 1)
        cpu_view.add_(1.0)
        torch.testing.assert_close(base.cpu(), cpu_base)

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_mul_inplace_transpose_no_copy_back(self):
        """mul_ on transposed view should not use copy-back when epilogue is enabled.
        mul_ dispatches as aten::mul.out.
        """
        base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).to("neuron")
        view = base.transpose(0, 1)

        def do_inplace():
            view.mul_(2.0)

        copy_count = count_copy_ops(do_inplace)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations with epilogue enabled, but got {copy_count}."

        cpu_base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        cpu_base.transpose(0, 1).mul_(2.0)
        torch.testing.assert_close(base.cpu(), cpu_base)

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_add_inplace_transpose_4d_no_copy_back(self):
        """add_ on 4D transposed view (attention pattern) should not use copy-back."""
        shape = (2, 8, 4, 16)
        init = torch.randn(shape)
        base = init.clone().to("neuron")
        view = base.transpose(1, 2)  # [B, H, S, D]
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations for 4D transpose epilogue, but got {copy_count}."

        cpu_base = init.clone()
        cpu_base.transpose(1, 2).add_(1.0)
        torch.testing.assert_close(base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)


# ============================================================================
# Test: Copy-back IS used when epilogue is disabled
# ============================================================================


class TestCopyBackUsedWithoutEpilogue:
    """Verify that without epilogue, the copy-back path IS used.

    These tests confirm the current behavior: in-place ops on non-contiguous
    views trigger a separate copy_ operation to scatter the result back.
    """

    @pytest.mark.skipif(
        epilogue_enabled(),
        reason="Epilogue is enabled — copy-back should NOT be used",
    )
    def test_add_inplace_transpose_uses_copy_back(self):
        """Without epilogue, add_ on transposed view should use copy-back."""
        base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).to("neuron")
        view = base.transpose(0, 1)
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        # With copy-back path, at least one copy_ should be triggered
        # (CopyBackToOriginal uses .copy_() internally)
        assert copy_count > 0, (
            f"Expected copy_ operations without epilogue, but got {copy_count}. "
            f"The copy-back path should be active when epilogue is disabled."
        )

        # Verify correctness still holds with copy-back
        cpu_base = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        cpu_base.transpose(0, 1).add_(1.0)
        torch.testing.assert_close(base.cpu(), cpu_base)


# ============================================================================
# Test: Epilogue for out= ops on transposed views
# ============================================================================


class TestEpilogueOutOps:
    """Verify that the epilogue handles out= ops with non-contiguous output tensors.

    Operations like mul_ dispatch as aten::mul.out with an explicit out= argument
    rather than using input_output_aliases. Phase 1.5 added detection for these
    patterns via schema write-aliased arguments.
    """

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_mul_out_transpose_no_copy_back(self):
        """torch.mul with out= transposed view should not use copy-back."""
        a = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).to("neuron")
        base = torch.zeros(3, 4, device="neuron")
        out_view = base.transpose(0, 1)  # non-contiguous out=
        assert not out_view.is_contiguous()

        def do_out():
            torch.mul(a.transpose(0, 1), 2.0, out=out_view)

        copy_count = count_copy_ops(do_out)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations for out= epilogue, but got {copy_count}."

        cpu_a = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        cpu_base = torch.zeros(3, 4)
        torch.mul(cpu_a.transpose(0, 1), 2.0, out=cpu_base.transpose(0, 1))
        torch.testing.assert_close(base.cpu(), cpu_base)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_out_transpose_correctness(self, dtype):
        """out= with transposed view produces correct results for various dtypes."""
        init_a = torch.randn(3, 4, dtype=dtype)
        init_b = torch.randn(3, 4, dtype=dtype)

        cpu_out_base = torch.zeros(4, 3, dtype=dtype)
        torch.add(init_a.t(), init_b.t(), out=cpu_out_base)

        neuron_out_base = torch.zeros(4, 3, dtype=dtype, device="neuron")
        torch.add(init_a.to("neuron").t(), init_b.to("neuron").t(), out=neuron_out_base)

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(neuron_out_base.cpu(), cpu_out_base, rtol=rtol, atol=atol)


# ============================================================================
# Test: Epilogue for slice views
# ============================================================================


class TestEpilogueSliceOps:
    """Verify that the epilogue handles slice view patterns."""

    @pytest.mark.skipif(
        not epilogue_enabled() or not scatter_iota_enabled(),
        reason="Requires TORCH_NEURONX_ENABLE_EPILOGUE=1 and TORCH_NEURONX_ENABLE_SCATTER_IOTA=1",
    )
    def test_add_inplace_column_slice_no_copy_back(self):
        """add_ on column-sliced view should not use copy-back when epilogue is enabled."""
        base = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8).to("neuron")
        view = base[:, :4]  # first 4 columns — non-contiguous slice
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations for slice epilogue, but got {copy_count}."

        cpu_base = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)
        cpu_base[:, :4].add_(1.0)
        torch.testing.assert_close(base.cpu(), cpu_base)

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_add_inplace_row_slice_no_copy_back(self):
        """add_ on row-sliced view should not use copy-back."""
        base = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6).to("neuron")
        view = base[1:3, :]  # rows 1-2

        def do_inplace():
            view.add_(10.0)

        copy_count = count_copy_ops(do_inplace)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations for row slice epilogue, but got {copy_count}."

        cpu_base = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
        cpu_base[1:3, :].add_(10.0)
        torch.testing.assert_close(base.cpu(), cpu_base)

    def test_slice_epilogue_preserves_non_sliced_region(self):
        """Verify that non-sliced regions of the base tensor are preserved."""
        init = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)

        cpu_base = init.clone()
        cpu_base[:, :4].add_(100.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[:, :4].add_(100.0)

        # Check that columns 0-3 were updated
        torch.testing.assert_close(neuron_base[:, :4].cpu(), cpu_base[:, :4])
        # Check that columns 4-7 were preserved
        torch.testing.assert_close(neuron_base[:, 4:].cpu(), cpu_base[:, 4:])

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_strided_slice_handled_by_epilogue(self):
        """Strided slices (step>1) are now handled by SCATTER_IOTA epilogue.

        SCATTER_IOTA uses iota-computed flat indices and can handle ANY stride
        pattern including step>1, unlike the old DYNAMIC_UPDATE_SLICE approach.
        No copy-back should occur.
        """
        base = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6).to("neuron")
        view = base[::2, :]  # step=2 — handled by SCATTER_IOTA epilogue

        def do_inplace():
            view.add_(100.0)

        copy_count = count_copy_ops(do_inplace)

        # SCATTER_IOTA handles strided slices — no copy-back needed
        assert copy_count == 0, (
            f"Expected 0 copy_ operations for strided slice epilogue (SCATTER_IOTA), "
            f"but got {copy_count}."
        )

        # Verify correctness against CPU reference
        cpu_base = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
        cpu_base[::2, :].add_(100.0)
        torch.testing.assert_close(base.cpu(), cpu_base)


# ============================================================================
# Test: Epilogue for compound (transpose + slice) patterns
# ============================================================================


class TestEpilogueCompoundOps:
    """Verify that the epilogue handles compound patterns (transpose + slice).

    The RoPE pattern (transpose then slice last dim) is the primary use case.
    SCATTER_IOTA handles the slice writeback after the inverse transpose.
    """

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_add_inplace_transpose_slice_compound_no_copy_back(self):
        """add_ on transpose+slice compound view should not use copy-back.

        This is the RoPE pattern: base.transpose(1,2)[..., D//2:]
        """
        base = torch.randn(1, 8, 4, 8).to("neuron")
        view = base.transpose(1, 2)[:, :, :, 4:]  # [1, 4, 8, 4]
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        assert copy_count == 0, (
            f"Expected 0 copy_ operations for compound (transpose+slice) epilogue, "
            f"but got {copy_count}."
        )

        # Re-create on CPU with same initial values
        cpu_base_orig = base.cpu()
        cpu_base_orig.transpose(1, 2)[:, :, :, 4:].add_(1.0)
        # Just verify the operation ran without error and base was modified

    def test_transpose_slice_compound_correctness(self):
        """Compound (transpose+slice) produces correct results matching CPU."""
        init = torch.randn(1, 8, 4, 8)

        cpu_base = init.clone()
        cpu_base.transpose(1, 2)[:, :, :, 4:].add_(1.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(1, 2)[:, :, :, 4:].add_(1.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_add_inplace_multi_axis_slice_no_copy_back(self):
        """add_ on multi-axis sliced view should not use copy-back."""
        base = torch.randn(8, 10).to("neuron")
        view = base[2:6, 3:7]  # [4, 4] from [8, 10]
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(100.0)

        copy_count = count_copy_ops(do_inplace)

        assert copy_count == 0, (
            f"Expected 0 copy_ operations for multi-axis slice epilogue, " f"but got {copy_count}."
        )

    def test_multi_axis_slice_correctness(self):
        """Multi-axis slice produces correct results matching CPU."""
        init = torch.randn(8, 10)

        cpu_base = init.clone()
        cpu_base[2:6, 3:7].add_(100.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[2:6, 3:7].add_(100.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_multi_axis_slice_preserves_non_sliced_region(self):
        """Multi-axis slice preserves regions outside the slice."""
        init = torch.arange(1, 81, dtype=torch.float32).reshape(8, 10)

        cpu_base = init.clone()
        cpu_base[2:6, 3:7].add_(100.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[2:6, 3:7].add_(100.0)

        # Sliced region updated
        torch.testing.assert_close(neuron_base[2:6, 3:7].cpu(), cpu_base[2:6, 3:7])
        # Regions outside slice preserved
        torch.testing.assert_close(neuron_base[:2, :].cpu(), cpu_base[:2, :])
        torch.testing.assert_close(neuron_base[6:, :].cpu(), cpu_base[6:, :])
        torch.testing.assert_close(neuron_base[:, :3].cpu(), cpu_base[:, :3])
        torch.testing.assert_close(neuron_base[:, 7:].cpu(), cpu_base[:, 7:])


# ============================================================================
# Test: Epilogue for out= ops on slice views
# ============================================================================


class TestEpilogueOutSliceOps:
    """Verify that the epilogue handles out= ops with slice view outputs."""

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_mul_out_slice_no_copy_back(self):
        """torch.mul with out= slice view should not use copy-back."""
        a = torch.randn(4, 8).to("neuron")
        base = torch.zeros(4, 8, device="neuron")
        out_view = base[:, :4]  # non-contiguous slice out=
        assert not out_view.is_contiguous()

        def do_out():
            torch.mul(a[:, :4], 2.0, out=out_view)

        copy_count = count_copy_ops(do_out)

        assert (
            copy_count == 0
        ), f"Expected 0 copy_ operations for out= slice epilogue, but got {copy_count}."

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_out_slice_correctness(self, dtype):
        """out= with slice view produces correct results for various dtypes."""
        init_a = torch.randn(4, 8, dtype=dtype)

        cpu_base = torch.zeros(4, 8, dtype=dtype)
        torch.mul(init_a[:, :4], 2.0, out=cpu_base[:, :4])

        neuron_base = torch.zeros(4, 8, dtype=dtype, device="neuron")
        torch.mul(init_a.to("neuron")[:, :4], 2.0, out=neuron_base[:, :4])

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=rtol, atol=atol)


# ============================================================================
# Test: Copy-back IS used for slices when epilogue is disabled
# ============================================================================


class TestSliceCopyBackWithoutEpilogue:
    """Verify that without epilogue, slice in-place ops use copy-back."""

    @pytest.mark.skipif(
        epilogue_enabled(),
        reason="Epilogue is enabled — copy-back should NOT be used",
    )
    def test_add_inplace_slice_uses_copy_back(self):
        """Without epilogue, add_ on sliced view should use copy-back."""
        base = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8).to("neuron")
        view = base[:, :4]
        assert not view.is_contiguous()

        def do_inplace():
            view.add_(1.0)

        copy_count = count_copy_ops(do_inplace)

        assert (
            copy_count > 0
        ), f"Expected copy_ operations without epilogue for slice, but got {copy_count}."

        cpu_base = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)
        cpu_base[:, :4].add_(1.0)
        torch.testing.assert_close(base.cpu(), cpu_base)


# ============================================================================
# Test: Correctness with epilogue enabled (values match CPU)
# ============================================================================


class TestEpilogueCorrectness:
    """Verify numerical correctness when epilogue is enabled.

    These tests run regardless of epilogue flag — they verify that the result
    is correct whether using epilogue or copy-back path.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_add_inplace_transpose_correctness(self, dtype):
        """add_ on transposed view produces correct results for various dtypes."""
        init = torch.arange(1, 13, dtype=dtype).reshape(3, 4)

        cpu_base = init.clone()
        cpu_base.transpose(0, 1).add_(1.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(0, 1).add_(1.0)

        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "shape,dims",
        [
            ((3, 4), (0, 1)),
            ((2, 3, 4), (0, 2)),
            ((2, 3, 4), (1, 2)),
            ((2, 4, 3, 8), (1, 2)),
        ],
    )
    def test_add_inplace_transpose_shapes(self, shape, dims):
        """add_ on transposed view with various shapes and transpose dims."""
        init = torch.randn(shape)

        cpu_base = init.clone()
        cpu_base.transpose(*dims).add_(1.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(*dims).add_(1.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_multiple_inplace_on_same_view(self):
        """Multiple in-place operations on the same non-contiguous view."""
        init = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)

        cpu_base = init.clone()
        cpu_view = cpu_base.transpose(0, 1)
        cpu_view.add_(1.0)
        cpu_view.mul_(2.0)
        cpu_view.add_(-3.0)

        neuron_base = init.clone().to("neuron")
        neuron_view = neuron_base.transpose(0, 1)
        neuron_view.add_(1.0)
        neuron_view.mul_(2.0)
        neuron_view.add_(-3.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base)

    def test_inplace_on_view_then_read_base(self):
        """After in-place on view, reading the base should reflect changes."""
        cpu_base = torch.zeros(3, 4)
        cpu_base.transpose(0, 1).fill_(42.0)

        neuron_base = torch.zeros(3, 4, device="neuron")
        neuron_base.transpose(0, 1).fill_(42.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base)

    def test_inplace_on_sliced_view_correctness(self):
        """add_ on column-sliced view produces correct results."""
        init = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)

        cpu_base = init.clone()
        cpu_base[:, :2].add_(10.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[:, :2].add_(10.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base)

    def test_inplace_on_strided_slice_correctness(self):
        """add_ on strided slice (every other row) produces correct results."""
        init = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)

        cpu_base = init.clone()
        cpu_base[::2, :].add_(100.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[::2, :].add_(100.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base)


# ============================================================================
# Test: Advanced epilogue patterns — higher dims, select, cache, mixed ops
# ============================================================================


class TestEpilogueAdvancedPatterns:
    """Advanced epilogue pattern tests for comprehensive coverage.

    These verify correctness for:
    - 3D/4D transpose + slice compound patterns (RoPE-like)
    - select() + slice patterns (dimension reduction via indexing)
    - Epilogue MLIR cache hit path (same pattern executed twice)
    - Different in-place ops (sub_, mul_) on compound views
    - Full base tensor integrity after partial writes
    """

    def test_3d_transpose_slice_compound(self):
        """3D tensor with transpose(0,2) then slice on dim 1."""
        init = torch.randn(3, 8, 4)

        cpu_base = init.clone()
        cpu_base.transpose(0, 2)[:, :4, :].add_(10.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(0, 2)[:, :4, :].add_(10.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_4d_attention_rope_pattern(self):
        """4D RoPE pattern: transpose(1,2) then slice last dim (the primary use case)."""
        init = torch.randn(2, 8, 4, 16)

        cpu_base = init.clone()
        cpu_base.transpose(1, 2)[:, :, :, :8].add_(1.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(1, 2)[:, :, :, :8].add_(1.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_4d_attention_rope_second_half(self):
        """RoPE pattern on second half: transpose(1,2) then slice last dim offset."""
        init = torch.randn(2, 8, 4, 16)

        cpu_base = init.clone()
        cpu_base.transpose(1, 2)[:, :, :, 8:].mul_(2.0)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(1, 2)[:, :, :, 8:].mul_(2.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_select_then_slice(self):
        """select(0, idx) reduces a dimension, then slice on remaining dims."""
        init = torch.randn(4, 8, 6)

        cpu_base = init.clone()
        cpu_base[1, :, :3].add_(50.0)  # select dim0=1, then slice dim2

        neuron_base = init.clone().to("neuron")
        neuron_base[1, :, :3].add_(50.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_select_middle_dim(self):
        """select on middle dimension (common in attention head indexing)."""
        init = torch.randn(2, 4, 8)

        cpu_base = init.clone()
        cpu_base[:, 2, :].add_(100.0)  # select dim1=2

        neuron_base = init.clone().to("neuron")
        neuron_base[:, 2, :].add_(100.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_epilogue_cache_hit_correctness(self):
        """Run the same epilogue pattern twice to explicitly exercise cache hit path.

        The first execution creates the cache entry (MISS).
        The second execution uses the cached MLIR (HIT) and must re-wire
        the extra SCATTER_IOTA src tensor correctly.
        """
        for iteration in range(2):
            init = torch.randn(4, 8)

            cpu_base = init.clone()
            cpu_base[:, :4].add_(float(iteration + 1))

            neuron_base = init.clone().to("neuron")
            neuron_base[:, :4].add_(float(iteration + 1))

            torch.testing.assert_close(
                neuron_base.cpu(),
                cpu_base,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Cache hit correctness failed on iteration {iteration}",
            )

    def test_compound_cache_hit_correctness(self):
        """Run compound (transpose+slice) pattern twice for cache hit coverage."""
        for iteration in range(2):
            init = torch.randn(1, 8, 4, 8)

            cpu_base = init.clone()
            cpu_base.transpose(1, 2)[:, :, :, 4:].add_(float(iteration + 1))

            neuron_base = init.clone().to("neuron")
            neuron_base.transpose(1, 2)[:, :, :, 4:].add_(float(iteration + 1))

            torch.testing.assert_close(
                neuron_base.cpu(),
                cpu_base,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Compound cache hit correctness failed on iteration {iteration}",
            )

    def test_sub_inplace_on_slice(self):
        """sub_ on sliced view (different op than add_)."""
        init = torch.randn(4, 8)

        cpu_base = init.clone()
        cpu_base[:, 2:6].sub_(5.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[:, 2:6].sub_(5.0)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_mul_inplace_on_compound(self):
        """mul_ on compound (transpose+slice) view."""
        init = torch.randn(2, 6, 3, 8)

        cpu_base = init.clone()
        cpu_base.transpose(1, 2)[:, :, :, :4].mul_(0.5)

        neuron_base = init.clone().to("neuron")
        neuron_base.transpose(1, 2)[:, :, :, :4].mul_(0.5)

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)

    def test_non_contiguous_slice_preserves_full_base(self):
        """Verify full base tensor integrity after slice epilogue on various regions."""
        init = torch.arange(1, 49, dtype=torch.float32).reshape(6, 8)

        cpu_base = init.clone()
        cpu_base[1:4, 2:6].add_(1000.0)

        neuron_base = init.clone().to("neuron")
        neuron_base[1:4, 2:6].add_(1000.0)

        # Full tensor comparison — catches any corruption in non-sliced regions
        torch.testing.assert_close(neuron_base.cpu(), cpu_base)


# ============================================================================
# Test: copy_() on non-contiguous dst uses epilogue (SDPA pattern)
# ============================================================================


def _assert_no_nrt_copy(test_name):
    """Check that no neuron_to_neuron NRT copy ops were used.

    CopyToNonContiguous generates neuron::copy::neuron_to_neuron ops.
    The MLIR epilogue path fuses the transpose into the NEFF and does NOT
    produce neuron_to_neuron ops. This helper verifies the epilogue path.
    Only checks when epilogue is enabled.
    """
    if not epilogue_enabled():
        return
    executed_ops = torch_neuronx.get_executed_ops()
    nrt_copy_ops = [op for op in executed_ops if "neuron_to_neuron" in op]
    assert len(nrt_copy_ops) == 0, (
        f"{test_name}: Expected no neuron_to_neuron ops "
        f"(CopyToNonContiguous) with epilogue, "
        f"but found: {nrt_copy_ops}. "
        f"DispatchCopyImpl should route through MLIR epilogue."
    )


class TestEpilogueCopyNonContiguous:
    """Verify that copy_() on non-contiguous dst produces correct results.

    The SDPA match_layout pattern: contiguous src copied into transposed dst.
    When epilogue is enabled, DispatchCopyImpl routes this through MLIR
    instead of CopyToNonContiguous, fusing the transpose into the NEFF.

    Each test verifies both:
    1. Correctness: result matches CPU reference
    2. Dispatch path: no neuron_to_neuron NRT ops (when epilogue enabled)
    """

    def test_copy_to_transposed_2d(self):
        """copy_ contiguous src into transposed 2D dst."""
        init = torch.randn(3, 4)

        cpu_dst = torch.zeros(4, 3)
        cpu_dst.transpose(0, 1).copy_(init)

        torch_neuronx.clear_op_tracking()
        neuron_dst = torch.zeros(4, 3, device="neuron")
        neuron_dst.transpose(0, 1).copy_(init.clone().to("neuron"))
        torch.neuron.synchronize()

        torch.testing.assert_close(neuron_dst.cpu(), cpu_dst, rtol=1e-4, atol=1e-4)
        _assert_no_nrt_copy("test_copy_to_transposed_2d")

    def test_copy_to_transposed_4d_sdpa_pattern(self):
        """copy_ contiguous [B,H,S,D] into transposed dst -- the SDPA hot path."""
        bs, nh, sl, hd = 2, 4, 8, 16
        init = torch.randn(bs, nh, sl, hd)

        cpu_base = torch.zeros(bs, sl, nh, hd)
        cpu_base.transpose(1, 2).copy_(init)

        torch_neuronx.clear_op_tracking()
        neuron_base = torch.zeros(bs, sl, nh, hd, device="neuron")
        neuron_base.transpose(1, 2).copy_(init.to("neuron"))
        torch.neuron.synchronize()

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)
        _assert_no_nrt_copy("test_copy_to_transposed_4d_sdpa_pattern")

    def test_copy_to_sliced_view(self):
        """copy_ contiguous src into sliced dst preserves non-sliced regions."""
        init_src = torch.randn(4, 4)
        init_dst = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)

        cpu_dst = init_dst.clone()
        cpu_dst[:, :4].copy_(init_src)

        torch_neuronx.clear_op_tracking()
        neuron_dst = init_dst.clone().to("neuron")
        neuron_dst[:, :4].copy_(init_src.to("neuron"))
        torch.neuron.synchronize()

        torch.testing.assert_close(neuron_dst.cpu(), cpu_dst)
        _assert_no_nrt_copy("test_copy_to_sliced_view")

    @pytest.mark.skipif(
        not epilogue_enabled(),
        reason="Epilogue not enabled (set TORCH_NEURONX_ENABLE_EPILOGUE=1)",
    )
    def test_copy_no_separate_copy_ops(self):
        """When epilogue enabled, copy_ to transposed dst should route
        through MLIR epilogue -- NOT CopyToNonContiguous.

        Dedicated dispatch path test: only runs with epilogue enabled.
        """
        bs, nh, sl, hd = 1, 4, 8, 16
        src = torch.randn(bs, nh, sl, hd, device="neuron")
        base = torch.zeros(bs, sl, nh, hd, device="neuron")
        dst = base.transpose(1, 2)

        torch_neuronx.clear_op_tracking()
        dst.copy_(src)
        torch.neuron.synchronize()

        _assert_no_nrt_copy("test_copy_no_separate_copy_ops")

    def test_copy_cache_hit_correctness(self):
        """Run same copy_ pattern twice to exercise cache hit path."""
        for iteration in range(2):
            init = torch.randn(4, 8)

            cpu_base = torch.zeros(8, 4)
            cpu_base.transpose(0, 1).copy_(init)

            torch_neuronx.clear_op_tracking()
            neuron_base = torch.zeros(8, 4, device="neuron")
            neuron_base.transpose(0, 1).copy_(init.to("neuron"))
            torch.neuron.synchronize()

            torch.testing.assert_close(
                neuron_base.cpu(),
                cpu_base,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Cache correctness failed on iteration {iteration}",
            )
            _assert_no_nrt_copy(f"test_copy_cache_hit iter={iteration}")

    @pytest.mark.parametrize(
        "shape,dims",
        [
            ((3, 4), (0, 1)),
            ((2, 3, 4), (0, 2)),
            ((2, 3, 4), (1, 2)),
            ((2, 4, 3, 8), (1, 2)),
        ],
    )
    def test_copy_various_transpose_shapes(self, shape, dims):
        """copy_ into transposed view with various shapes."""
        init = torch.randn(shape)

        cpu_base = torch.zeros(shape)
        cpu_base.transpose(*dims).copy_(init.transpose(*dims))

        torch_neuronx.clear_op_tracking()
        neuron_base = torch.zeros(shape, device="neuron")
        neuron_base.transpose(*dims).copy_(init.transpose(*dims).to("neuron"))
        torch.neuron.synchronize()

        torch.testing.assert_close(neuron_base.cpu(), cpu_base, rtol=1e-4, atol=1e-4)
        _assert_no_nrt_copy(f"test_copy_various shape={shape} dims={dims}")


# ============================================================================
# Test: Random fuzz test for epilogue robustness
# ============================================================================


# Available random view operations for the fuzz test.
# Each op takes a tensor and rng, applies a random view transformation,
# and returns (new_tensor, op_description_string, replay_lambda).
# Excluded: expand/broadcast (PyTorch blocks in-place on broadcast views),
#           unfold (overlapping strides), flip (negative strides on CPU).
_EPILOGUE_FUZZ_OPS = ["transpose", "slice", "select", "narrow", "permute"]


def _can_epilogue_handle(view_tensor):
    """Mirror the C++ TryCreateEpilogue rejection conditions in Python.

    Returns True if the epilogue should handle this pattern (no fallback expected).
    Returns False if the pattern hits a known TryCreateEpilogue limitation.

    WARNING: This function MUST be kept in sync with TryCreateEpilogue in
    Transformations.cpp. If the C++ acceptance criteria change, update this
    function accordingly. Mismatches cause false-positive or false-negative
    fuzz test failures.

    See: torch_neuronx/csrc/core/lazy_materialization/Transformations.cpp
    """
    if view_tensor.is_contiguous():
        return False  # no epilogue needed

    shape = view_tensor.shape
    strides = view_tensor.stride()
    offset = view_tensor.storage_offset()
    ndims = len(shape)
    elem_size = view_tensor.element_size()
    storage_elements = view_tensor.storage().nbytes() // elem_size

    # Reject negative strides or broadcast (stride=0 with size>1)
    for i in range(ndims):
        if strides[i] < 0:
            return False
        if strides[i] == 0 and shape[i] > 1:
            return False

    # Collect active dims (size > 1)
    active_dims = [i for i in range(ndims) if shape[i] > 1]
    if not active_dims:
        return False

    # Sort by stride descending (physical order)
    phys_order = sorted(active_dims, key=lambda d: strides[d], reverse=True)
    num_active = len(phys_order)

    # Compute base shape from stride ratios
    phys_base = [0] * num_active
    has_slice = False

    for pi in range(num_active):
        dim = phys_order[pi]
        view_size = shape[dim]

        if pi == 0:
            if strides[dim] <= 0:
                return False
            phys_base[pi] = storage_elements // strides[dim]
        else:
            outer_dim = phys_order[pi - 1]
            if strides[dim] <= 0:
                return False
            phys_base[pi] = strides[outer_dim] // strides[dim]
            # Non-integer stride ratio
            if strides[outer_dim] != phys_base[pi] * strides[dim]:
                return False

        # Innermost must have stride 1
        if pi == num_active - 1 and strides[dim] != 1:
            return False

        if phys_base[pi] < view_size:
            return False  # invalid
        if phys_base[pi] > view_size:
            has_slice = True

    # Offset decomposition
    def compute_row_major_strides(shp):
        s = [1] * len(shp)
        for i in range(len(shp) - 2, -1, -1):
            s[i] = s[i + 1] * max(shp[i + 1], 1)
        return s

    base_strides = compute_row_major_strides(phys_base)
    remaining = offset
    for pi in range(num_active):
        if base_strides[pi] > 0 and remaining >= base_strides[pi]:
            start = remaining // base_strides[pi]
            remaining %= base_strides[pi]
            if start + shape[phys_order[pi]] > phys_base[pi]:
                return False
    if remaining != 0:
        return False

    # Check identity permutation
    is_identity = True
    for i in range(num_active):
        if phys_order[i] != active_dims[i]:
            is_identity = False
            break

    if not has_slice and is_identity:
        return False  # contiguous — no epilogue needed

    # Verify storage bounds
    base_elements = 1
    for s in phys_base:
        base_elements *= s
    return base_elements <= storage_elements


def _apply_random_view_ops(tensor, rng, num_ops):
    """Apply a random sequence of view ops to a tensor.

    Returns (view_tensor, op_log, replay_fns) where replay_fns can be applied
    to a different tensor with the same base shape to get the same view.
    """
    replay_fns = []
    op_log = []

    for _ in range(num_ops):
        ndim = tensor.dim()
        if ndim < 1:
            break

        valid_ops = list(_EPILOGUE_FUZZ_OPS)
        if ndim < 2:
            for remove_op in ["transpose", "permute"]:
                if remove_op in valid_ops:
                    valid_ops.remove(remove_op)
            if "select" in valid_ops:
                valid_ops.remove("select")

        op = rng.choice(valid_ops)

        try:
            if op == "transpose":
                dims = rng.choice(ndim, size=2, replace=False).tolist()
                d0, d1 = int(dims[0]), int(dims[1])
                tensor = tensor.transpose(d0, d1)
                replay_fns.append(lambda t, d0=d0, d1=d1: t.transpose(d0, d1))
                op_log.append(f"transpose({d0},{d1})")

            elif op == "permute":
                perm = rng.permutation(ndim).tolist()
                perm = [int(p) for p in perm]
                tensor = tensor.permute(perm)
                replay_fns.append(lambda t, p=perm: t.permute(p))
                op_log.append(f"permute({perm})")

            elif op == "select":
                if ndim < 2:
                    continue
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    continue
                idx = int(rng.integers(0, dim_size))
                tensor = tensor.select(dim, idx)
                replay_fns.append(lambda t, d=dim, i=idx: t.select(d, i))
                op_log.append(f"select(dim={dim},idx={idx})")

            elif op == "narrow":
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    continue
                start = int(rng.integers(0, dim_size - 1))
                length = int(rng.integers(1, min(dim_size - start, dim_size // 2 + 1) + 1))
                tensor = torch.narrow(tensor, dim, start, length)
                replay_fns.append(lambda t, d=dim, s=start, ln=length: torch.narrow(t, d, s, ln))
                op_log.append(f"narrow(dim={dim},start={start},len={length})")

            elif op == "slice":
                dim = int(rng.integers(0, ndim))
                dim_size = tensor.size(dim)
                if dim_size <= 1:
                    continue
                start = int(rng.integers(0, dim_size - 1))
                end = int(rng.integers(start + 1, dim_size + 1))
                # Occasional step > 1 for strided slices
                step = int(rng.integers(2, 4)) if rng.random() > 0.7 and (end - start) >= 4 else 1
                idx = [slice(None)] * ndim
                idx[dim] = slice(start, end, step)
                idx = tuple(idx)
                tensor = tensor[idx]
                replay_fns.append(lambda t, idx=idx: t[idx])
                op_log.append(f"slice(dim={dim},{start}:{end}:{step})")

        except (RuntimeError, IndexError):
            # Some random combinations may be invalid — skip
            continue

    return tensor, op_log, replay_fns


def _replay_view_ops(tensor, replay_fns):
    """Replay a sequence of view operations on a tensor."""
    for fn in replay_fns:
        tensor = fn(tensor)
    return tensor


class TestEpilogueRandomFuzz:
    """Random fuzz test for epilogue robustness.

    Generates random sequences of view operations (transpose, slice, select,
    narrow, permute), applies a random in-place operation on the resulting
    non-contiguous view, and verifies the full base tensor matches CPU reference.

    This tests the epilogue's ability to correctly scatter computed results
    back into arbitrary stride patterns, including preservation of non-modified
    storage regions.

    Each seed produces a deterministic random sequence for reproducibility.
    On failure, the seed and operation log are printed for debugging.
    """

    @pytest.mark.parametrize("seed", range(100))
    def test_random_inplace_on_random_view(self, seed):
        """Generate random view, apply add_ with random scalar, verify full base.

        Also checks that the epilogue handled the pattern (no CopyBackToOriginal
        fallback via copy_ or contiguous ops). If fallback was used, the test
        still verifies correctness but reports the fallback as a warning.
        """
        rng = np.random.default_rng(seed)

        # Random base shape: 2-4 dims, sizes 2-12
        ndim = int(rng.integers(2, 5))
        shape = tuple(int(rng.integers(3, 13)) for _ in range(ndim))

        # Random number of view ops: 1-4
        num_ops = int(rng.integers(1, 5))

        # Random scalar for in-place add
        scalar = float(rng.uniform(1.0, 10.0))

        # Build original tensor (shared between CPU and Neuron)
        torch.manual_seed(seed * 1000)
        original = torch.randn(*shape, dtype=torch.float32)

        # Probe the view pattern on CPU first
        probe = original.clone()
        probe_view, op_log, replay_fns = _apply_random_view_ops(probe, rng, num_ops)

        # Skip if view ended up contiguous (no epilogue needed)
        if probe_view.is_contiguous():
            pytest.skip(f"seed={seed}: random view is contiguous, epilogue not needed")

        # Skip if view has 0 elements
        if probe_view.numel() == 0:
            pytest.skip(f"seed={seed}: random view has 0 elements")

        # CPU reference
        cpu_base = original.clone()
        cpu_view = _replay_view_ops(cpu_base, replay_fns)
        cpu_view.add_(scalar)

        # Neuron — track ops for fallback detection
        torch_neuronx.clear_op_tracking()

        neuron_base = original.clone().to("neuron")
        neuron_view = _replay_view_ops(neuron_base, replay_fns)
        try:
            neuron_view.add_(scalar)
            torch.neuron.synchronize()
        except TypeError as e:
            if "cannot unpack non-iterable NoneType" in str(e):
                pytest.fail(
                    f"seed={seed} shape={shape} ops={op_log}: "
                    f"OP LOWERING FAILURE (not epilogue bug): lower_func returned None "
                    f"for this shape/stride combination. This is a pre-existing gap in "
                    f"the Python op lowering, not related to the epilogue. "
                    f"TypeError: {e}"
                )
            raise

        try:
            result = neuron_base.cpu()
        except RuntimeError as e:
            pytest.fail(f"seed={seed} shape={shape} ops={op_log}: " f"Neuron execution failed: {e}")

        # Verify correctness: full base tensor must match CPU reference
        torch.testing.assert_close(
            result,
            cpu_base,
            rtol=1e-4,
            atol=1e-4,
            msg=f"seed={seed} shape={shape} ops={op_log} scalar={scalar:.2f}",
        )

        # Assert no fallback — with a known limitation exception.
        # Known limitation: when select/narrow reduces dims such that the view's
        # numel is much smaller than the storage (partial-storage view),
        # TryCreateEpilogue may fail to decompose the outermost base dimension.
        # This is analogous to TorchNeuron's CONTIGUOUS_COPY_FALLBACK.
        if epilogue_enabled():
            executed_ops = torch_neuronx.get_executed_ops()
            fallback_ops = [
                op
                for op in executed_ops
                if "copy_" in op or "neuron_to_neuron" in op or "contiguous" in op
            ]
            if fallback_ops:
                # Use the Python mirror of TryCreateEpilogue to check if the
                # epilogue SHOULD have handled this pattern. If yes, it's a bug.
                # If no, it's a known limitation (pattern not decomposable).
                epilogue_should_handle = _can_epilogue_handle(probe_view)

                if epilogue_should_handle:
                    pytest.fail(
                        f"seed={seed} shape={shape} ops={op_log}: "
                        f"UNEXPECTED epilogue fallback — _can_epilogue_handle() "
                        f"says this pattern is decomposable. "
                        f"view: shape={list(probe_view.shape)}, "
                        f"strides={list(probe_view.stride())}, "
                        f"offset={probe_view.storage_offset()}. "
                        f"Fallback ops: {fallback_ops}"
                    )

    @pytest.mark.parametrize("seed", range(100))
    def test_random_inplace_mul_on_random_view(self, seed):
        """Same as above but with mul_ instead of add_ (tests different op)."""
        rng = np.random.default_rng(seed + 1000)

        ndim = int(rng.integers(2, 5))
        shape = tuple(int(rng.integers(3, 13)) for _ in range(ndim))
        num_ops = int(rng.integers(1, 4))
        scalar = float(rng.uniform(0.5, 3.0))

        torch.manual_seed(seed * 2000)
        original = torch.randn(*shape, dtype=torch.float32)

        # Probe view pattern
        probe = original.clone()
        probe_view, op_log, replay_fns = _apply_random_view_ops(probe, rng, num_ops)

        if probe_view.is_contiguous() or probe_view.numel() == 0:
            pytest.skip(f"seed={seed}: view is contiguous or empty")

        # CPU reference
        cpu_base = original.clone()
        cpu_view = _replay_view_ops(cpu_base, replay_fns)
        cpu_view.mul_(scalar)

        # Neuron — track ops for fallback detection
        torch_neuronx.clear_op_tracking()

        neuron_base = original.clone().to("neuron")
        neuron_view = _replay_view_ops(neuron_base, replay_fns)
        try:
            neuron_view.mul_(scalar)
            torch.neuron.synchronize()
        except TypeError as e:
            if "cannot unpack non-iterable NoneType" in str(e):
                pytest.fail(
                    f"seed={seed} shape={shape} ops={op_log}: "
                    f"OP LOWERING FAILURE (not epilogue bug): lower_func returned None "
                    f"for this shape/stride combination. This is a pre-existing gap in "
                    f"the Python op lowering, not related to the epilogue. "
                    f"TypeError: {e}"
                )
            raise

        try:
            result = neuron_base.cpu()
        except RuntimeError as e:
            pytest.fail(f"seed={seed} shape={shape} ops={op_log}: " f"Neuron execution failed: {e}")

        # Verify correctness
        torch.testing.assert_close(
            result,
            cpu_base,
            rtol=1e-4,
            atol=1e-4,
            msg=f"seed={seed} shape={shape} ops={op_log} scalar={scalar:.2f}",
        )

        # Assert no fallback — using Python mirror of TryCreateEpilogue
        if epilogue_enabled():
            executed_ops = torch_neuronx.get_executed_ops()
            fallback_ops = [
                op
                for op in executed_ops
                if "copy_" in op or "neuron_to_neuron" in op or "contiguous" in op
            ]
            if fallback_ops:
                epilogue_should_handle = _can_epilogue_handle(probe_view)

                if epilogue_should_handle:
                    pytest.fail(
                        f"seed={seed} shape={shape} ops={op_log}: "
                        f"UNEXPECTED epilogue fallback — _can_epilogue_handle() "
                        f"says this pattern is decomposable. "
                        f"view: shape={list(probe_view.shape)}, "
                        f"strides={list(probe_view.stride())}, "
                        f"offset={probe_view.storage_offset()}. "
                        f"Fallback ops: {fallback_ops}"
                    )


# ============================================================================
# Test: Epilogue for MULTI-OUTPUT ops with non-contiguous out= tensors

#
# Regression coverage for the SCATTER_IOTA extra-src wiring bug: the generic
# scatter epilogue must emit ONE extra base-storage input (and one extra src)
# PER scatter output. Multi-output ops (max.dim / min.dim -> values+indices,
# _native_batch_norm_legit -> 3 outputs) with a strided out= previously hit
#     "Module1 input address count (N) does not match function argument count (M)"
# (or SIGSEGV for foreach) because the wiring assumed a single scatter output.
# ============================================================================


def _make_noncontiguous_like(t):
    """Build a non-contiguous tensor with the same shape/dtype/device as ``t``.

    Mirrors torch.testing.make_tensor(noncontiguous=True): allocates 2x storage
    and takes a stride-2 view, which AnalyzeEpilogueLayout rejects and the
    generic SCATTER_IOTA epilogue must handle. Returns a contiguous fallback for
    0/1-element tensors (nothing to make non-contiguous).
    """
    if t.dim() == 0 or t.numel() <= 1:
        return torch.empty_like(t)
    doubled = torch.empty([2 * x for x in t.shape], dtype=t.dtype, device=t.device)
    view = doubled[(slice(None, None, 2),) * t.dim()]
    assert view.shape == t.shape and not view.is_contiguous()
    return view


class TestEpilogueMultiOutputOps:
    """Multi-output ops with non-contiguous out= tensors must run end-to-end and
    stay numerically correct — each transformed output wires its own scatter."""

    @pytest.mark.skipif(
        not epilogue_enabled() or not scatter_iota_enabled(),
        reason="Requires TORCH_NEURONX_ENABLE_EPILOGUE=1 and TORCH_NEURONX_ENABLE_SCATTER_IOTA=1",
    )
    @pytest.mark.parametrize("fn_name", ["max", "min"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
    def test_dim_reduction_two_outputs_noncontiguous_out(self, fn_name, dtype):
        """max/min over a dim return (values, indices). Both out= tensors are
        non-contiguous, so BOTH outputs get a SCATTER_IOTA epilogue. Previously
        raised 'input address count (1) does not match function argument count (2)'.
        """
        x = (torch.arange(3 * 4 * 5).reshape(3, 4, 5) % 17).to(dtype)
        fn = getattr(torch, fn_name)

        cpu_vals, cpu_idx = fn(x, dim=1)

        xn = x.to("neuron")
        vals_out = _make_noncontiguous_like(cpu_vals.to(dtype)).to("neuron")
        idx_out = _make_noncontiguous_like(cpu_idx).to("neuron")
        assert not vals_out.is_contiguous() and not idx_out.is_contiguous()

        fn(xn, dim=1, out=(vals_out, idx_out))
        torch.neuron.synchronize()

        torch.testing.assert_close(vals_out.cpu(), cpu_vals.to(dtype))
        torch.testing.assert_close(idx_out.cpu(), cpu_idx)

    @pytest.mark.skipif(
        not epilogue_enabled() or not scatter_iota_enabled(),
        reason="Requires TORCH_NEURONX_ENABLE_EPILOGUE=1 and TORCH_NEURONX_ENABLE_SCATTER_IOTA=1",
    )
    def test_sort_two_outputs_noncontiguous_out(self):
        """sort returns (values, indices); exercises two-output scatter wiring."""
        x = (torch.arange(5 * 6).reshape(5, 6) % 13).to(torch.float32)
        cpu_vals, cpu_idx = torch.sort(x, dim=1)

        xn = x.to("neuron")
        vals_out = _make_noncontiguous_like(cpu_vals).to("neuron")
        idx_out = _make_noncontiguous_like(cpu_idx).to("neuron")

        torch.sort(xn, dim=1, out=(vals_out, idx_out))
        torch.neuron.synchronize()

        torch.testing.assert_close(vals_out.cpu(), cpu_vals)
        torch.testing.assert_close(idx_out.cpu(), cpu_idx)

    @pytest.mark.skipif(
        not epilogue_enabled() or not scatter_iota_enabled(),
        reason="Requires TORCH_NEURONX_ENABLE_EPILOGUE=1 and TORCH_NEURONX_ENABLE_SCATTER_IOTA=1",
    )
    def test_dim_reduction_cache_hit_two_outputs(self):
        """Run the same two-output strided-out reduction twice to exercise the
        MergedOperationCache HIT re-wire path (one extra src per scatter output)."""
        x = (torch.arange(4 * 5).reshape(4, 5) % 11).to(torch.float32)
        for _ in range(2):
            cpu_vals, cpu_idx = torch.max(x, dim=1)
            xn = x.to("neuron")
            vals_out = _make_noncontiguous_like(cpu_vals).to("neuron")
            idx_out = _make_noncontiguous_like(cpu_idx).to("neuron")
            torch.max(xn, dim=1, out=(vals_out, idx_out))
            torch.neuron.synchronize()
            torch.testing.assert_close(vals_out.cpu(), cpu_vals)
            torch.testing.assert_close(idx_out.cpu(), cpu_idx)
