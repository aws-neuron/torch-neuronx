"""Tests for dynamic tiling in matrix multiplication operations."""

import pytest
import torch

import torch_neuronx


class TestMmTiling:
    """Test cases for dynamic tiling in mm operation"""

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (128, 64, 128),  # Exact tile size
            (256, 64, 128),  # Multiple of tile size
            (384, 64, 128),  # 256 + 128
            (512, 64, 128),  # Exact tile size (512)
            (600, 64, 128),  # 512 + 88 (needs padding)
            (1280, 64, 128),  # 1024 + 256
            (1984, 128, 256),  # 1024 + 512 + 256 + 128 + 64 (complex decomposition)
        ],
    )
    def test_mm_tiling_various_sizes(self, m, k, n):
        """Test mm with various input sizes that trigger tiling"""
        mat1 = torch.randn(m, k, dtype=torch.bfloat16).to("neuron")
        mat2 = torch.randn(k, n, dtype=torch.bfloat16).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.mm(mat1_cpu, mat2_cpu)

        # Compute on device with tiling
        result = torch.mm(mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)

    def test_mm_tiling_with_padding(self):
        """Test mm with size that requires padding in last tile"""
        # 600 = 512 + 88, last tile needs padding from 88 to 128
        m, k, n = 600, 64, 128
        mat1 = torch.randn(m, k, dtype=torch.bfloat16).to("neuron")
        mat2 = torch.randn(k, n, dtype=torch.bfloat16).to("neuron")

        expected = torch.mm(mat1.cpu(), mat2.cpu())
        result = torch.mm(mat1, mat2)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)

    def test_mm_tiling_small_input(self):
        """Test mm with input smaller than smallest tile size"""
        # 64 < 128 (smallest tile), should use tile_size=128 with padding
        m, k, n = 64, 32, 64
        mat1 = torch.randn(m, k, dtype=torch.bfloat16).to("neuron")
        mat2 = torch.randn(k, n, dtype=torch.bfloat16).to("neuron")

        expected = torch.mm(mat1.cpu(), mat2.cpu())
        result = torch.mm(mat1, mat2)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)

    def test_mm_tiling_with_out_parameter(self):
        """Test mm tiling with output parameter"""
        m, k, n = 384, 64, 128
        mat1 = torch.randn(m, k, dtype=torch.bfloat16).to("neuron")
        mat2 = torch.randn(k, n, dtype=torch.bfloat16).to("neuron")
        out = torch.empty(m, n, dtype=torch.bfloat16).to("neuron")

        expected = torch.mm(mat1.cpu(), mat2.cpu())
        torch.mm(mat1, mat2, out=out)

        torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_mm_tiling_dtypes(self, dtype):
        """Test mm tiling with different data types"""
        m, k, n = 384, 64, 128
        mat1 = torch.randn(m, k, dtype=dtype).to("neuron")
        mat2 = torch.randn(k, n, dtype=dtype).to("neuron")

        expected = torch.mm(mat1.cpu(), mat2.cpu())
        result = torch.mm(mat1, mat2)

        assert result.dtype == dtype
        rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
