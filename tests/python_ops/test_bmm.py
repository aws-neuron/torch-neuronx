import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


class TestBmm:
    """Test cases for batch matrix multiplication (bmm) operation"""

    def test_bmm_basic(self):
        """Test basic batch matrix multiplication"""

        # Create test tensors with batch dimension
        batch_size = 4
        mat1 = torch.randn(batch_size, 3, 4).to("neuron")
        mat2 = torch.randn(batch_size, 4, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.bmm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.bmm(mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_out(self):
        """Test batch matrix multiplication with output tensor"""

        # Create test tensors
        batch_size = 4
        mat1 = torch.randn(batch_size, 3, 4).to("neuron")
        mat2 = torch.randn(batch_size, 4, 5).to("neuron")
        out = torch.empty(batch_size, 3, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.bmm(mat1_cpu, mat2_cpu)

        # Compute on device with out parameter
        torch.bmm(mat1, mat2, out=out)

        # Compare results
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_single_batch(self):
        """Test bmm with batch size 1"""

        mat1 = torch.randn(1, 3, 4).to("neuron")
        mat2 = torch.randn(1, 4, 5).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.bmm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.bmm(mat1, mat2)

        # Compare results
        assert result.shape == (1, 3, 5)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_square_matrices(self):
        """Test bmm with square matrices"""

        batch_size = 3
        size = 5
        mat1 = torch.randn(batch_size, size, size).to("neuron")
        mat2 = torch.randn(batch_size, size, size).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.bmm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.bmm(mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_different_sizes(self):
        """Test bmm with various tensor sizes"""

        test_cases = [
            ((2, 3, 4), (2, 4, 5)),  # batch=2, 3x4 @ 4x5 -> 3x5
            ((5, 1, 7), (5, 7, 1)),  # batch=5, 1x7 @ 7x1 -> 1x1
            ((3, 10, 2), (3, 2, 8)),  # batch=3, 10x2 @ 2x8 -> 10x8
            ((1, 4, 4), (1, 4, 4)),  # batch=1, 4x4 @ 4x4 -> 4x4
            ((10, 2, 3), (10, 3, 4)),  # batch=10, 2x3 @ 3x4 -> 2x4
        ]

        for shape1, shape2 in test_cases:
            mat1 = torch.randn(*shape1).to("neuron")
            mat2 = torch.randn(*shape2).to("neuron")

            # Compute on CPU for reference
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.bmm(mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.bmm(mat1, mat2)

            # Check shape and values
            batch, m, k1 = shape1
            _, k2, n = shape2
            assert result.shape == (batch, m, n)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_empty_batch(self):
        """Test bmm with empty batch dimension"""

        # 0x3x4 @ 0x4x5 -> 0x3x5
        mat1 = torch.randn(0, 3, 4).to("neuron")
        mat2 = torch.randn(0, 4, 5).to("neuron")

        result = torch.bmm(mat1, mat2)

        assert result.shape == (0, 3, 5)

    def test_bmm_identity(self):
        """Test bmm with identity matrices"""

        batch_size = 3
        size = 4
        mat = torch.randn(batch_size, size, size).to("neuron")

        # Create batch of identity matrices
        identity = torch.eye(size).to("neuron").unsqueeze(0).expand(batch_size, -1, -1)

        # A @ I = A
        result1 = torch.bmm(mat, identity)
        torch.testing.assert_close(result1.cpu(), mat.cpu(), rtol=1e-4, atol=1e-4)

        # I @ A = A
        result2 = torch.bmm(identity, mat)
        torch.testing.assert_close(result2.cpu(), mat.cpu(), rtol=1e-4, atol=1e-4)

    @assert_raises(RuntimeError)
    def test_bmm_batch_mismatch_error(self):
        """Test that batch dimension mismatch raises error"""

        mat1 = torch.randn(3, 4, 5).to("neuron")
        mat2 = torch.randn(4, 5, 6).to("neuron")  # Different batch size

        torch.bmm(mat1, mat2)

    @assert_raises(RuntimeError)
    def test_bmm_dimension_mismatch_error(self):
        """Test that matrix dimension mismatch raises error"""

        batch_size = 3
        mat1 = torch.randn(batch_size, 3, 4).to("neuron")
        mat2 = torch.randn(batch_size, 5, 6).to("neuron")  # Incompatible dimensions

        torch.bmm(mat1, mat2)

    @assert_raises(RuntimeError, match="batch1 must be a 3D tensor")
    def test_bmm_wrong_dimensions_error(self):
        """Test that non-3D tensors raise appropriate error"""

        # 2D tensor (should use mm instead)
        tensor_2d = torch.randn(3, 4).to("neuron")
        tensor_3d = torch.randn(2, 4, 5).to("neuron")

        torch.bmm(tensor_2d, tensor_3d)

        # 4D tensor
        tensor_4d = torch.randn(2, 3, 4, 5).to("neuron")

        torch.bmm(tensor_4d, tensor_3d)

    def test_bmm_different_dtypes(self):
        """Test bmm with different data types"""

        dtypes = [torch.float32, torch.bfloat16, torch.float16]

        for dtype in dtypes:
            batch_size = 2
            mat1 = torch.randn(batch_size, 3, 4, dtype=dtype).to("neuron")
            mat2 = torch.randn(batch_size, 4, 5, dtype=dtype).to("neuron")

            # Compute on CPU for reference
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.bmm(mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.bmm(mat1, mat2)

            # Check dtype is preserved
            assert result.dtype == dtype
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-4 if dtype == torch.float32 else 1e-8,
                atol=1e-4 if dtype == torch.float32 else 1e-8,
            )

    def test_bmm_large_batch(self):
        """Test bmm with larger batch sizes"""

        batch_size = 32
        mat1 = torch.randn(batch_size, 8, 16).to("neuron")
        mat2 = torch.randn(batch_size, 16, 12).to("neuron")

        # Compute on CPU for reference
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.bmm(mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.bmm(mat1, mat2)

        # Compare results
        assert result.shape == (batch_size, 8, 12)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
