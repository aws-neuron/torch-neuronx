"""Tests for gather NKI kernel on TNE."""

import pytest
import torch

from torch_neuronx.python_ops.nki_kernels.gather import gather_kernel


class TestGather:
    """Test gather NKI kernel with various sizes and dtypes."""

    @pytest.mark.parametrize(
        "bs_slen, dim_size, top_k, dtype",
        [
            (128, 512, 2, torch.bfloat16),
            (4096, 2880, 4, torch.bfloat16),
            (512, 1024, 4, torch.bfloat16),
            (256, 2048, 8, torch.bfloat16),
            (1024, 4096, 2, torch.bfloat16),
            (128, 512, 2, torch.float32),
            (4096, 2880, 4, torch.float32),
        ],
    )
    def test_gather_kernel(self, bs_slen, dim_size, top_k, dtype):
        """Test gather kernel correctness against PyTorch reference."""
        torch.manual_seed(42)

        output_rows = bs_slen * top_k
        input_cpu = torch.randn(bs_slen, dim_size, dtype=dtype)
        index_1d_cpu = torch.randint(0, bs_slen, (output_rows,), dtype=torch.int32)

        expected = input_cpu[index_1d_cpu.long()]

        result = gather_kernel(
            input_cpu.to("neuron"),
            0,
            index_1d_cpu.to("neuron"),
        )

        atol, rtol = (0, 1e-8) if dtype == torch.bfloat16 else (1e-4, 1e-5)
        assert torch.allclose(result.cpu(), expected, atol=atol, rtol=rtol)
