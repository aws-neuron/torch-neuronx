"""Tests for scatter_add NKI kernel on TNE."""

import pytest
import torch

from torch_neuronx.python_ops.nki_kernels.scatter_add import scatter_add_kernel


def _scatter_add_ref(input, dim, index_1d, src):
    """CPU reference: scatter_add with 1D index expanded to 2D."""
    index_2d = index_1d.unsqueeze(1).expand_as(src)
    return input.scatter_add(dim, index_2d, src)


class TestScatterAdd:
    """Test scatter_add NKI kernel with various sizes and dtypes."""

    @pytest.mark.parametrize(
        "bs_slen, dim_size, top_k, dtype",
        [
            (128, 512, 2, torch.bfloat16),
            (128, 512, 2, torch.float32),
            (256, 2048, 8, torch.bfloat16),
            (256, 2048, 8, torch.float32),
            (512, 1024, 4, torch.bfloat16),
            (512, 1024, 4, torch.float32),
            (1024, 4096, 2, torch.bfloat16),
            (1024, 4096, 2, torch.float32),
            (4096, 2880, 4, torch.bfloat16),
            (4096, 2880, 4, torch.float32),
        ],
    )
    def test_scatter_add_kernel(self, bs_slen, dim_size, top_k, dtype):
        """Test scatter_add kernel correctness against PyTorch reference."""
        torch.manual_seed(42)

        src_rows = bs_slen * top_k
        input_cpu = torch.randn(bs_slen, dim_size, dtype=dtype)
        src_cpu = torch.randn(src_rows, dim_size, dtype=dtype)
        # Indices must be unique within each 128-row tile (kernel constraint)
        indices = []
        for start in range(0, src_rows, 128):
            tile_len = min(128, src_rows - start)
            indices.append(torch.randperm(bs_slen)[:tile_len])
        index_1d_cpu = torch.cat(indices).to(torch.int32)

        expected = _scatter_add_ref(input_cpu, 0, index_1d_cpu.long(), src_cpu)

        result = scatter_add_kernel(
            input_cpu.to("neuron"),
            0,
            index_1d_cpu.to("neuron"),
            src_cpu.to("neuron"),
        )

        atol, rtol = (1e-1, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
        assert torch.allclose(result.cpu(), expected, atol=atol, rtol=rtol)
