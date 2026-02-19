"""Test as_strided decomposition with offset conversion for non-contiguous tensors."""

import pytest
import torch


class TestAsStridedOffsetConversion:
    """Test as_strided handles offset conversion when stride changes."""

    def test_transpose_then_chunk(self):
        """Test chunk on transposed tensor which triggers as_strided with offset."""
        linear = torch.nn.Linear(800, 1200, device="neuron")

        @torch.compile(backend="neuron")
        def fn(x):
            out = linear(x)
            x_t = out.t()
            chunks = torch.chunk(x_t, 3, dim=0)
            return chunks[1]  # Second chunk has non-zero offset

        x = torch.randn(64, 800, device="neuron")
        result = fn(x)

        # Compare with CPU
        linear_cpu = linear.cpu()
        x_cpu = x.cpu()
        out_cpu = linear_cpu(x_cpu)
        x_t_cpu = out_cpu.t()
        chunks_cpu = torch.chunk(x_t_cpu, 3, dim=0)
        expected = chunks_cpu[1]

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_transpose_chunk_different_sizes(self):
        """Test chunk on transposed tensor with different dimensions."""
        linear = torch.nn.Linear(512, 768, device="neuron")

        @torch.compile(backend="neuron")
        def fn(x):
            out = linear(x)
            x_t = out.t()
            chunks = torch.chunk(x_t, 4, dim=0)
            return chunks[2]

        x = torch.randn(32, 512, device="neuron")
        result = fn(x)

        linear_cpu = linear.cpu()
        x_cpu = x.cpu()
        out_cpu = linear_cpu(x_cpu)
        x_t_cpu = out_cpu.t()
        chunks_cpu = torch.chunk(x_t_cpu, 4, dim=0)
        expected = chunks_cpu[2]

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_transpose_split(self):
        """Test split on transposed linear output."""
        linear = torch.nn.Linear(256, 1024, device="neuron")

        @torch.compile(backend="neuron")
        def fn(x):
            out = linear(x)
            x_t = out.t()
            splits = torch.split(x_t, 256, dim=0)
            return splits[1]

        x = torch.randn(128, 256, device="neuron")
        result = fn(x)

        linear_cpu = linear.cpu()
        x_cpu = x.cpu()
        out_cpu = linear_cpu(x_cpu)
        x_t_cpu = out_cpu.t()
        splits_cpu = torch.split(x_t_cpu, 256, dim=0)
        expected = splits_cpu[1]

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)
