"""Test cases for RANK3 contiguous kernel with transformer use cases."""

import os

import pytest
import torch

import torch_neuronx


class TestContiguousRank3Transformer:
    """Test cases for RANK3 contiguous operations in transformer scenarios."""

    def test_attention_head_selection(self):
        """Test contiguous operation on strided attention head selection."""
        # Typical multi-head attention dimensions
        batch_size = 8
        num_heads = 12
        seq_len = 128

        # Attention weights: [batch, heads, seq, seq]
        attention_weights_cpu = torch.randn(
            batch_size, num_heads, seq_len, seq_len, dtype=torch.float32
        )
        attention_weights = attention_weights_cpu.to("neuron")

        # Select every other attention head (common in pruning/analysis)
        selected_heads = attention_weights[:, ::2, :, :]  # Shape: [8, 6, 128, 128]
        assert not selected_heads.is_contiguous()

        # Expected shape and strides
        assert selected_heads.shape == (batch_size, num_heads // 2, seq_len, seq_len)
        # Stride in head dimension should be 2x the original

        # Make contiguous
        contiguous_selected = selected_heads.contiguous()
        assert contiguous_selected.is_contiguous()

        # Verify correctness
        expected = attention_weights_cpu[:, ::2, :, :].contiguous()
        assert torch.allclose(contiguous_selected.cpu(), expected)

    def test_strided_sequence_positions(self):
        """Test contiguous operation on strided sequence position selection."""
        # Hidden states dimensions
        batch_size = 16
        seq_len = 512
        hidden_dim = 768

        # Hidden states: [batch, seq, hidden]
        hidden_states_cpu = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
        hidden_states = hidden_states_cpu.to("neuron")

        # Select every 4th position (for hierarchical processing)
        strided_positions = hidden_states[:, ::4, :]  # Shape: [16, 128, 768]
        assert not strided_positions.is_contiguous()

        # Expected shape
        assert strided_positions.shape == (batch_size, seq_len // 4, hidden_dim)

        # Make contiguous
        contiguous_strided = strided_positions.contiguous()
        assert contiguous_strided.is_contiguous()

        # Verify correctness
        expected = hidden_states_cpu[:, ::4, :].contiguous()
        assert torch.allclose(contiguous_strided.cpu(), expected)

    def test_sparse_chunk_processing(self):
        """Test contiguous operation on sparse chunk selection."""
        # Chunked sequence dimensions
        batch_size = 4
        seq_len = 1024
        chunk_size = 64
        hidden_dim = 512
        num_chunks = seq_len // chunk_size  # 16

        # Start with hidden states
        hidden_states_cpu = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
        hidden_states = hidden_states_cpu.to("neuron")

        # Reshape to chunks: [batch, num_chunks, chunk_size, hidden]
        chunked = hidden_states.view(batch_size, num_chunks, chunk_size, hidden_dim)

        # Process every other chunk (sparse processing)
        sparse_chunks = chunked[:, ::2, :, :]  # Shape: [4, 8, 64, 512]
        assert not sparse_chunks.is_contiguous()

        # Expected shape
        assert sparse_chunks.shape == (batch_size, num_chunks // 2, chunk_size, hidden_dim)

        # Make contiguous
        contiguous_sparse = sparse_chunks.contiguous()
        assert contiguous_sparse.is_contiguous()

        # Verify correctness
        chunked_cpu = hidden_states_cpu.view(batch_size, num_chunks, chunk_size, hidden_dim)
        expected = chunked_cpu[:, ::2, :, :].contiguous()
        assert torch.allclose(contiguous_sparse.cpu(), expected)

    def test_attention_head_pruning_pattern(self):
        """Test contiguous operation with specific head selection pattern."""
        # Multi-head attention output dimensions
        batch_size = 8
        seq_len = 256
        num_heads = 8
        head_dim = 64

        # Attention output: [batch, seq, heads, head_dim]
        attention_output_cpu = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float32
        )
        attention_output = attention_output_cpu.to("neuron")

        # Select heads with stride 3 (keeping heads 0, 3, 6)
        pruned_heads = attention_output[:, :, ::3, :]  # Shape: [8, 256, 3, 64]
        assert not pruned_heads.is_contiguous()

        # Make contiguous
        contiguous_pruned = pruned_heads.contiguous()
        assert contiguous_pruned.is_contiguous()

        # Verify correctness
        expected = attention_output_cpu[:, :, ::3, :].contiguous()
        assert torch.allclose(contiguous_pruned.cpu(), expected)

    def test_mixed_dtypes_rank3(self):
        """Test RANK3 operations with different data types."""
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            # Simple 3D case with middle stride
            x_cpu = torch.randn(4, 8, 16, dtype=dtype)
            x = x_cpu.to("neuron")

            # Create RANK3 pattern
            y = x[:, ::2, :]  # Shape: [4, 4, 16]
            assert not y.is_contiguous()

            # Make contiguous
            z = y.contiguous()
            assert z.is_contiguous()

            # Verify
            expected = x_cpu[:, ::2, :].contiguous()
            assert torch.allclose(z.cpu().float(), expected.float(), rtol=1e-3, atol=1e-3)

    def test_large_scale_rank3(self):
        """Test RANK3 with larger tensors typical in transformers."""
        # Large-scale dimensions
        batch_size = 32
        num_heads = 16
        seq_len = 2048

        # Large attention pattern
        attention_cpu = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.float16)
        attention = attention_cpu.to("neuron")

        # Select every 4th head (aggressive pruning)
        selected = attention[:, ::4, :, :]  # Shape: [32, 4, 2048, 2048]
        assert not selected.is_contiguous()

        # Make contiguous
        contiguous_selected = selected.contiguous()
        assert contiguous_selected.is_contiguous()

        # Verify shape and sample values
        assert contiguous_selected.shape == (batch_size, num_heads // 4, seq_len, seq_len)

        # Check a few values
        assert torch.allclose(
            contiguous_selected.cpu()[0, 0, 0, :10], attention_cpu[0, 0, 0, :10], rtol=1e-3
        )
