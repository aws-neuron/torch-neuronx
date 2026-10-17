"""Test cat with shared-storage views (FSDP unshard pattern).

Reproduces the HBM limit error where torch.chunk creates views sharing
the same storage, and the MLIR tracer emits each view's parent shape
as the HLO input parameter instead of the actual view shape.
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCatSharedStorage:
    """Tests for torch.cat with inputs that share underlying storage."""

    def test_chunk_dim1_cat_dedup_all_same_storage(self):
        """4 dim=1 chunks from same tensor — dedup should reduce 4 inputs to 1."""
        device = "neuron"
        t = torch.randn(16, 64, device=device)
        chunks = torch.chunk(t, 4, dim=1)
        result = torch.cat(list(chunks), dim=1)
        assert result.shape == (16, 64)
        torch.testing.assert_close(result.cpu(), t.cpu(), rtol=0, atol=0)

    def test_chunk_dim1_cat_dedup_other_first(self):
        """other, a0, a1, a2 — different tensor first, 3 chunks deduped."""
        device = "neuron"
        t = torch.randn(16, 64, device=device)
        other = torch.randn(16, 16, device=device)
        chunks = torch.chunk(t, 4, dim=1)
        result = torch.cat([other, chunks[0], chunks[1], chunks[2]], dim=1)
        expected = torch.cat(
            [other.cpu(), chunks[0].cpu(), chunks[1].cpu(), chunks[2].cpu()], dim=1
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_chunk_dim1_cat_dedup_other_last(self):
        """a0, a1, a2, other — different tensor last."""
        device = "neuron"
        t = torch.randn(16, 64, device=device)
        other = torch.randn(16, 16, device=device)
        chunks = torch.chunk(t, 4, dim=1)
        result = torch.cat([chunks[0], chunks[1], chunks[2], other], dim=1)
        expected = torch.cat(
            [chunks[0].cpu(), chunks[1].cpu(), chunks[2].cpu(), other.cpu()], dim=1
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_chunk_dim1_cat_dedup_interleaved_two_tensors(self):
        """a0, b0, a1, b1 — interleaved chunks from two tensors (double dedup)."""
        device = "neuron"
        tensor_a = torch.randn(16, 64, device=device)
        tensor_b = torch.randn(16, 64, device=device)
        ac = torch.chunk(tensor_a, 4, dim=1)
        bc = torch.chunk(tensor_b, 4, dim=1)
        result = torch.cat([ac[0], bc[0], ac[1], bc[1]], dim=1)
        expected = torch.cat([ac[0].cpu(), bc[0].cpu(), ac[1].cpu(), bc[1].cpu()], dim=1)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_chunk_dim1_cat_no_dedup(self):
        """All different tensors — no dedup should happen."""
        device = "neuron"
        parts = [torch.randn(16, 16, device=device) for _ in range(4)]
        result = torch.cat(parts, dim=1)
        expected = torch.cat([p.cpu() for p in parts], dim=1)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_chunk_dim0_cat_no_prologue(self):
        """dim=0 chunks are contiguous — no prologue/dedup needed."""
        device = "neuron"
        t = torch.randn(16, 64, device=device)
        chunks = torch.chunk(t, 4, dim=0)
        result = torch.cat(list(chunks), dim=0)
        assert result.shape == (16, 64)
        torch.testing.assert_close(result.cpu(), t.cpu(), rtol=0, atol=0)

    def test_chunk_dim1_cat_dedup_triple_interleaved(self):
        """a0, b0, c0, a1, b1, c1 — triple dedup from three tensors."""
        device = "neuron"
        tensor_a = torch.randn(16, 48, device=device)
        tensor_b = torch.randn(16, 48, device=device)
        tensor_c = torch.randn(16, 48, device=device)
        ac = torch.chunk(tensor_a, 3, dim=1)
        bc = torch.chunk(tensor_b, 3, dim=1)
        cc = torch.chunk(tensor_c, 3, dim=1)
        result = torch.cat([ac[0], bc[0], cc[0], ac[1], bc[1], cc[1]], dim=1)
        expected = torch.cat(
            [ac[0].cpu(), bc[0].cpu(), cc[0].cpu(), ac[1].cpu(), bc[1].cpu(), cc[1].cpu()], dim=1
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
