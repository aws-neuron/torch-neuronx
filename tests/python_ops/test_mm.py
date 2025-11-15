import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


class TestMm:
    """Test cases for matrix multiplication (mm) operation"""

    def test_mm_basic(self):
        """Test basic matrix multiplication"""
        # Create test matrices
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.mm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.mm(mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_mm_out(self):
        """Test matrix multiplication with output tensor"""
        # Create test matrices
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")
        out = torch.empty(3, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.mm(mat1_cpu, mat2_cpu)

        # Compute on device with out parameter
        torch.mm(mat1, mat2, out=out)

        # Compare results
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_mm_square_matrices(self):
        """Test multiplication of square matrices"""
        # Create square matrices
        mat1 = torch.randn(5, 5).to("neuron")
        mat2 = torch.randn(5, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.mm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.mm(mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_mm_different_sizes(self):
        """Test multiplication with various matrix sizes"""
        test_cases = [
            ((2, 3), (3, 4)),  # 2x3 @ 3x4 -> 2x4
            ((1, 5), (5, 1)),  # 1x5 @ 5x1 -> 1x1
            ((10, 2), (2, 8)),  # 10x2 @ 2x8 -> 10x8
            ((4, 4), (4, 4)),  # 4x4 @ 4x4 -> 4x4
        ]

        for (m, k1), (k2, n) in test_cases:
            assert k1 == k2, "Inner dimensions must match"

            mat1 = torch.randn(m, k1).to("neuron")
            mat2 = torch.randn(k2, n).to("neuron")

            # Compute on CPU for reference
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.mm(mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.mm(mat1, mat2)

            # Check shape and values
            assert result.shape == (m, n)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_mm_empty_result(self):
        """Test multiplication resulting in empty tensor"""
        # 0x5 @ 5x3 -> 0x3
        mat1 = torch.randn(0, 5).to("neuron")
        mat2 = torch.randn(5, 3).to("neuron")

        result = torch.mm(mat1, mat2)

        assert result.shape == (0, 3)

    def test_mm_identity(self):
        """Test multiplication with identity matrix"""
        size = 4
        mat = torch.randn(size, size).to("neuron")
        identity = torch.eye(size).to("neuron")

        # A @ I = A
        result1 = torch.mm(mat, identity)
        torch.testing.assert_close(result1.cpu(), mat.cpu(), rtol=1e-4, atol=1e-4)

        # I @ A = A
        result2 = torch.mm(identity, mat)
        torch.testing.assert_close(result2.cpu(), mat.cpu(), rtol=1e-4, atol=1e-4)

    @assert_raises(RuntimeError, match="must have same reduction dim")
    def test_mm_dimension_mismatch_error(self, monkeypatch):
        """Test that dimension mismatch raises appropriate error"""
        monkeypatch.setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1")

        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(5, 6).to("neuron")  # Incompatible dimensions

        torch.mm(mat1, mat2)

    @assert_raises(RuntimeError, match="must be 2D")
    def test_mm_wrong_dimensions_error(self, monkeypatch):
        """Test that non-2D tensors raise appropriate error"""
        monkeypatch.setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1")
        # 3D tensor
        tensor_3d = torch.randn(2, 3, 4).to("neuron")
        tensor_2d = torch.randn(4, 5).to("neuron")

        torch.mm(tensor_3d, tensor_2d)

        # 1D tensor
        tensor_1d = torch.randn(4).to("neuron")

        torch.mm(tensor_2d, tensor_1d)

    def test_mm_different_dtypes(self, monkeypatch):
        """Test matrix multiplication with different data types"""
        monkeypatch.setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1")
        dtypes = [torch.float32, torch.bfloat16, torch.float16]

        for dtype in dtypes:
            mat1 = torch.randn(3, 4, dtype=dtype).to("neuron")
            mat2 = torch.randn(4, 5, dtype=dtype).to("neuron")

            # Compute on CPU for reference
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.mm(mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.mm(mat1, mat2)

            # Check dtype is preserved
            assert result.dtype == dtype
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-4 if dtype == torch.float32 else 1e-8,
                atol=1e-4 if dtype == torch.float32 else 1e-8,
            )
