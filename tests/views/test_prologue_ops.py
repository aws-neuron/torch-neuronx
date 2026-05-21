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
