"""
Tests for prologue transformation handling of non-contiguous inputs.

Verifies that various non-contiguous tensor patterns are handled by the prologue
system (fused into the NEFF) rather than falling back to CPU roundtrip.
"""

import pytest
import torch

import torch_neuronx

device = torch.device("neuron")


@pytest.fixture(autouse=True)
def reset_op_tracking():
    """Clear op tracking before each test to get clean fallback counts."""
    torch_neuronx.clear_op_tracking()
    yield


# Ops that legitimately fall back to CPU (random number generation, etc.)
_EXPECTED_FALLBACK_OPS = {"aten::normal_", "aten::uniform_", "aten::random_"}


def assert_no_unexpected_fallback():
    """Assert that no unexpected CPU fallback occurred during the test."""
    fallback_ops = torch_neuronx.get_fallback_ops()
    unexpected = [op for op in fallback_ops if op not in _EXPECTED_FALLBACK_OPS]
    assert len(unexpected) == 0, (
        f"Unexpected CPU fallback detected for ops: {unexpected}. "
        "This indicates the prologue failed to handle the non-contiguous pattern."
    )


class TestSharedStorageDifferentShapes:
    """Tests for multiple views of shared storage with different compound base shapes.

    This is the pattern that previously triggered the conflicting-base-shapes
    fallback in ProcessInputTransformations — now fixed by shape-aware dedup.
    """

    def test_stack_split_transpose_reshape(self):
        """torch.stack on split→transpose→reshape (GR00T pixel_shuffle_back pattern)."""
        channels, height, width = 1152, 18, 18
        num_slices = 4

        flat = torch.randn(
            num_slices * height * width, channels, dtype=torch.float32, device=device
        )
        slices = torch.split(flat, [height * width] * num_slices, dim=0)
        non_contig = [s.transpose(0, 1).reshape(channels, height, width) for s in slices]

        assert all(not t.is_contiguous() for t in non_contig)

        result = torch.stack(non_contig, dim=0)

        contig = [t.contiguous() for t in non_contig]
        expected = torch.stack(contig, dim=0)
        assert torch.allclose(result.cpu(), expected.cpu()), "Numerical mismatch!"

        assert_no_unexpected_fallback()

    def test_cat_split_transpose_reshape(self):
        """torch.cat on split→transpose→reshape."""
        channels, height, width = 1152, 18, 18
        num_slices = 4

        flat = torch.randn(
            num_slices * height * width, channels, dtype=torch.float32, device=device
        )
        slices = torch.split(flat, [height * width] * num_slices, dim=0)
        non_contig = [s.transpose(0, 1).reshape(channels, height, width) for s in slices]

        assert all(not t.is_contiguous() for t in non_contig)

        result = torch.cat(non_contig, dim=0)

        contig = [t.contiguous() for t in non_contig]
        expected = torch.cat(contig, dim=0)
        assert torch.allclose(result.cpu(), expected.cpu()), "Numerical mismatch!"

        assert_no_unexpected_fallback()

    def test_stack_split_transpose_small(self):
        """Small tensors: split→transpose→stack with shared storage."""
        flat = torch.randn(8, 6, dtype=torch.float32, device=device)
        slices = torch.split(flat, [2, 2, 2, 2], dim=0)
        non_contig = [s.transpose(0, 1) for s in slices]

        assert all(not t.is_contiguous() for t in non_contig)

        result = torch.stack(non_contig, dim=0)

        contig = [t.contiguous() for t in non_contig]
        expected = torch.stack(contig, dim=0)
        assert torch.allclose(result.cpu(), expected.cpu()), "Numerical mismatch!"

        assert_no_unexpected_fallback()


class TestSplitIndexCat:
    """Tests for split→index→cat patterns with shared storage.

    These patterns create contiguous views at different offsets within the same
    storage. When input dedup collapses them, size_bytes must cover the full
    storage range the prologue accesses.
    """

    def test_split_index_cat_multimodal_rope(self):
        """Qwen2.5-VL apply_multimodal_rotary_pos_emb pattern (HunyuanVideo).

        split(unequal sections, dim=-1) → index(dim=0) → cat(dim=-1)
        All views share storage with different offsets and shapes.
        """
        cos_cpu = torch.randn(3, 1, 1108, 128, dtype=torch.float32)
        cos_neuron = cos_cpu.to(device)

        mrope_section = [16, 24, 24, 16, 24, 24]
        splits_cpu = cos_cpu.split(mrope_section, dim=-1)
        splits_neuron = cos_neuron.split(mrope_section, dim=-1)

        cat_inputs_cpu = [s[i % 3] for i, s in enumerate(splits_cpu)]
        cat_inputs_neuron = [s[i % 3] for i, s in enumerate(splits_neuron)]

        expected = torch.cat(cat_inputs_cpu, dim=-1)
        result = torch.cat(cat_inputs_neuron, dim=-1)

        assert torch.allclose(
            result.cpu(), expected, atol=1e-5
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_index_cat_equal_sections(self):
        """Equal-sized split → index → cat."""
        x_cpu = torch.randn(4, 8, 64, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        splits_cpu = x_cpu.split(16, dim=-1)  # 4 equal chunks of 16
        splits_neuron = x_neuron.split(16, dim=-1)

        # Index different slices along dim=0
        cat_inputs_cpu = [s[i % 4] for i, s in enumerate(splits_cpu)]
        cat_inputs_neuron = [s[i % 4] for i, s in enumerate(splits_neuron)]

        expected = torch.cat(cat_inputs_cpu, dim=-1)
        result = torch.cat(cat_inputs_neuron, dim=-1)

        assert torch.allclose(
            result.cpu(), expected, atol=1e-5
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_select_cat_small(self):
        """Small tensor: split → select → cat (minimal repro)."""
        x_cpu = torch.randn(2, 4, 8, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # Split into unequal chunks along last dim
        splits_cpu = x_cpu.split([3, 5], dim=-1)
        splits_neuron = x_neuron.split([3, 5], dim=-1)

        # Select along dim=0
        cat_inputs_cpu = [splits_cpu[0][0], splits_cpu[1][1]]
        cat_inputs_neuron = [splits_neuron[0][0], splits_neuron[1][1]]

        expected = torch.cat(cat_inputs_cpu, dim=-1)
        result = torch.cat(cat_inputs_neuron, dim=-1)

        assert torch.allclose(
            result.cpu(), expected, atol=1e-5
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"


class TestDedupSliceDimension:
    """Tests for input dedup with shared-storage views sliced on different dimensions.

    Validates that the dedup size_bytes computation (max byte range) is correct
    for both major-dimension slices (contiguous views at different offsets) and
    non-major-dimension slices (non-contiguous views requiring prologue transforms).
    """

    # ─── Slice on major (outermost) dimension ───────────────────────────────
    # split(dim=0) produces contiguous views at different offsets within the
    # same storage. These share (base_ptr, shape) so they WILL dedup. The
    # deduped slot's size_bytes must cover the full storage range.

    def test_split_dim0_cat_equal_chunks(self):
        """split(dim=0, equal) → cat: contiguous views, same shape, different offsets."""
        x_cpu = torch.randn(16, 32, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # 4 equal chunks along dim=0: each is [4, 32], contiguous, different offsets
        chunks_cpu = x_cpu.split(4, dim=0)
        chunks_neuron = x_neuron.split(4, dim=0)

        # All chunks share storage and have same shape → dedup collapses to 1 slot
        expected = torch.cat(list(chunks_cpu), dim=0)
        result = torch.cat(list(chunks_neuron), dim=0)

        assert torch.allclose(
            result.cpu(), expected, atol=0
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim0_add_shared_views(self):
        """split(dim=0) → add pairs: same-shaped views from same storage added together."""
        x_cpu = torch.randn(8, 16, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        chunks_cpu = x_cpu.split(2, dim=0)  # 4 chunks of [2, 16]
        chunks_neuron = x_neuron.split(2, dim=0)

        # Add pairs: chunks[0]+chunks[2], chunks[1]+chunks[3]
        # Both inputs to each add share storage + shape → dedup
        expected = torch.cat([chunks_cpu[0] + chunks_cpu[2], chunks_cpu[1] + chunks_cpu[3]], dim=0)
        result = torch.cat(
            [chunks_neuron[0] + chunks_neuron[2], chunks_neuron[1] + chunks_neuron[3]],
            dim=0,
        )

        assert torch.allclose(
            result.cpu(), expected, atol=1e-5
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim0_unequal_cat(self):
        """split(dim=0, unequal) → index → cat: different shapes, no dedup expected."""
        x_cpu = torch.randn(10, 8, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # Unequal split: [3, 3, 4] along dim=0 — different shapes, same storage
        chunks_cpu = x_cpu.split([3, 3, 4], dim=0)
        chunks_neuron = x_neuron.split([3, 3, 4], dim=0)

        expected = torch.cat(list(chunks_cpu), dim=0)
        result = torch.cat(list(chunks_neuron), dim=0)

        assert torch.allclose(
            result.cpu(), expected, atol=0
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    # ─── Slice NOT on major dimension ───────────────────────────────────────
    # split(dim=1) or split(dim=-1) produces non-contiguous views (the stride
    # in the sliced dimension is larger than shape * inner_stride). The
    # prologue must handle these via SLICE transformations. These views have
    # different storage offsets but same shape when split equally.

    def test_split_dim1_cat_equal_chunks(self):
        """split(dim=1, equal) → cat: non-contiguous views, same shape, different offsets."""
        x_cpu = torch.randn(8, 32, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # 4 equal chunks along dim=1: each is [8, 8], NON-contiguous
        chunks_cpu = x_cpu.split(8, dim=1)
        chunks_neuron = x_neuron.split(8, dim=1)

        assert all(
            not c.is_contiguous() for c in chunks_neuron[1:]
        ), "Non-first dim=1 chunks should be non-contiguous"

        expected = torch.cat(list(chunks_cpu), dim=1)
        result = torch.cat(list(chunks_neuron), dim=1)

        assert torch.allclose(
            result.cpu(), expected, atol=0
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim1_add_shared_views(self):
        """split(dim=1) → add pairs: non-contiguous views requiring prologue slice."""
        x_cpu = torch.randn(4, 16, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        chunks_cpu = x_cpu.split(4, dim=1)  # 4 chunks of [4, 4]
        chunks_neuron = x_neuron.split(4, dim=1)

        # Add pairs: non-contiguous views with same base_ptr and same shape → dedup
        expected = torch.cat([chunks_cpu[0] + chunks_cpu[2], chunks_cpu[1] + chunks_cpu[3]], dim=1)
        result = torch.cat(
            [chunks_neuron[0] + chunks_neuron[2], chunks_neuron[1] + chunks_neuron[3]],
            dim=1,
        )

        assert torch.allclose(
            result.cpu(), expected, atol=1e-5
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim_last_cat_3d(self):
        """split(dim=-1) on 3D tensor → cat: non-contiguous slices on last dim."""
        x_cpu = torch.randn(2, 4, 24, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # Split last dim into 4 equal parts: each [2, 4, 6], non-contiguous
        chunks_cpu = x_cpu.split(6, dim=-1)
        chunks_neuron = x_neuron.split(6, dim=-1)

        assert all(not c.is_contiguous() for c in chunks_neuron[1:])

        expected = torch.cat(list(chunks_cpu), dim=-1)
        result = torch.cat(list(chunks_neuron), dim=-1)

        assert torch.allclose(
            result.cpu(), expected, atol=0
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim1_unequal_cat(self):
        """split(dim=1, unequal) → cat: non-contiguous, different shapes, no dedup."""
        x_cpu = torch.randn(4, 12, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # Unequal split along dim=1: [4, 4, 4] would dedup, [3, 4, 5] won't
        chunks_cpu = x_cpu.split([3, 4, 5], dim=1)
        chunks_neuron = x_neuron.split([3, 4, 5], dim=1)

        expected = torch.cat(list(chunks_cpu), dim=1)
        result = torch.cat(list(chunks_neuron), dim=1)

        assert torch.allclose(
            result.cpu(), expected, atol=0
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"

    def test_split_dim1_matmul_shared_storage(self):
        """split(dim=1) → matmul with shared-storage non-contiguous views.

        This tests that prologue correctly handles the size_bytes for
        non-contiguous deduped inputs used in compute (not just cat).
        """
        x_cpu = torch.randn(4, 8, dtype=torch.float32)
        x_neuron = x_cpu.to(device)

        # Split into 2 chunks of [4, 4] along dim=1 (non-contiguous)
        a_cpu, b_cpu = x_cpu.split(4, dim=1)
        a_neuron, b_neuron = x_neuron.split(4, dim=1)

        # a @ b.T — both share storage, same shape, but different offsets in dim=1
        expected = a_cpu @ b_cpu.T
        result = a_neuron @ b_neuron.T

        assert torch.allclose(
            result.cpu(), expected, atol=1e-4
        ), f"Numerical mismatch! max_diff={(result.cpu() - expected).abs().max().item()}"


@pytest.fixture
def native_int64_mode(monkeypatch):
    """Pin TORCH_NEURONX_INT64_MODE=native across the Python + C++ layers.

    Mirrors the ``int64_mode`` marker fixture in tests/python_ops/conftest.py
    (which isn't in scope here) so this regression is deterministic regardless
    of the shell's default mode.
    """
    from torch_neuronx import _C
    from torch_neuronx.int64_policy import init_cpp_policy, reset_int64_mode_cache
    from torch_neuronx.utils import global_version_hash

    monkeypatch.setenv("TORCH_NEURONX_INT64_MODE", "native")
    reset_int64_mode_cache()
    global_version_hash.cache_clear()
    if hasattr(_C, "_set_int64_mode_for_testing"):
        _C._set_int64_mode_for_testing("native")
    init_cpp_policy()

    yield

    reset_int64_mode_cache()
    global_version_hash.cache_clear()
    if hasattr(_C, "_reset_int64_mode_from_env"):
        _C._reset_int64_mode_from_env()
    init_cpp_policy()


class TestNonContiguousInt64ScalarPrologue:
    """Regression for the native-cascade int64 scalar over-promotion.

    A ``_foreach_add`` ScalarList over a *non-contiguous* int32 input must keep
    the python-int scalar at int32 (``int32 + int_scalar -> int32``). Previously,
    native mode promoted the wrapped scalar to int64 in the scalar-producing op
    (emitting a packed ``int32[..., 2]`` == ``2xi32``) while the consuming ``add``
    correctly kept it int32. Because the non-contiguous input splits the work into
    a prologue module + a main module, ``validateDepCompatibility`` (OpConcatUtils)
    then rejected the boundary with:

        Type mismatch in dependency: module1 output N (tensor<2xi32>)
        cannot connect to module2 input N (expected tensor<i32>)

    Packed mode was unaffected (uniformly packed on both sides).

    Fixed by making ``FlattenScalarListToTensors`` materialize ``Scalar[]``
    integer elements with weak-scalar promotion (the accompanying tensor
    operand's integer dtype) instead of blindly int64 in native mode.
    """

    def test_foreach_add_scalarlist_noncontiguous_int32(self, native_int64_mode):
        # Non-contiguous int32 [20, 20] (stride-2 view) forces the prologue split.
        x = torch.arange(20 * 40, dtype=torch.int32, device=device).reshape(20, 40)[:, ::2]
        assert not x.is_contiguous()

        result = torch._foreach_add([x], [3])[0]

        expected = x.cpu() + 3
        assert result.dtype == torch.int32, f"expected int32, got {result.dtype}"
        torch.testing.assert_close(result.cpu(), expected)

    def test_foreach_add_scalarlist_noncontiguous_int32_multi(self, native_int64_mode):
        # Same pattern with several differently-shaped non-contiguous tensors,
        # closer to the upstream OpInfo sample that first surfaced this.
        shapes = [(20, 20), (19, 19), (18, 18)]
        xs = [
            torch.arange(h * w * 2, dtype=torch.int32, device=device).reshape(h, w * 2)[:, ::2]
            for h, w in shapes
        ]
        assert all(not t.is_contiguous() for t in xs)
        scalars = [3, 9, 4]

        results = torch._foreach_add(xs, scalars)

        for r, x, s in zip(results, xs, scalars, strict=False):
            assert r.dtype == torch.int32, f"expected int32, got {r.dtype}"
            torch.testing.assert_close(r.cpu(), x.cpu() + s)
