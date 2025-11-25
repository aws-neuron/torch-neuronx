import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


class TestAddmm:
    """Test cases for matrix multiply-add (addmm) operation"""

    def test_addmm_basic(self):
        """Test basic matrix multiply-add: out = beta*input + alpha*(mat1 @ mat2)"""

        # Create test tensors
        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.addmm(input, mat1, mat2)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_out(self):
        """Test addmm with output tensor"""

        # Create test tensors
        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")
        out = torch.empty(3, 5).to("neuron")

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

        # Compute on device with out parameter
        torch.addmm(input, mat1, mat2, out=out)

        # Compare results
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_with_alpha_beta(self):
        """Test addmm with custom alpha and beta values"""

        # Create test tensors
        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")
        alpha = 2.5
        beta = -0.5

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu, alpha=alpha, beta=beta)

        # Compute on device
        result = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

        # Compare results
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_beta_zero(self):
        """Test addmm with beta=0 (equivalent to mm with scaling)"""

        # Create test tensors
        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")
        alpha = 2.0
        beta = 0.0

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu, alpha=alpha, beta=beta)

        # Compute on device
        result = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

        # Verify it's equivalent to alpha * mm
        mm_result = alpha * torch.mm(mat1, mat2)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result.cpu(), mm_result.cpu(), rtol=1e-4, atol=1e-4)

    def test_addmm_alpha_zero(self):
        """Test addmm with alpha=0 (equivalent to beta * input)"""

        # Create test tensors
        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")
        alpha = 0.0
        beta = 3.0

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu, alpha=alpha, beta=beta)

        # Compute on device
        result = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

        # Verify it's equivalent to beta * input
        scaled_input = beta * input
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result.cpu(), scaled_input.cpu(), rtol=1e-4, atol=1e-4)

    def test_addmm_broadcasting(self):
        """Test addmm with broadcasting of input tensor"""

        # Create test tensors with input that needs broadcasting
        input = torch.randn(1, 5).to("neuron")  # Will broadcast to 3x5
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.addmm(input, mat1, mat2)

        # Compare results
        assert result.shape == (3, 5)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_broadcasting_scalar(self):
        """Test addmm with scalar input (0D tensor)"""

        # Create test tensors with scalar input
        input = torch.tensor(2.0).to("neuron")  # Scalar will broadcast
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")

        # Compute on CPU for reference
        input_cpu = input.cpu()
        mat1_cpu = mat1.cpu()
        mat2_cpu = mat2.cpu()
        expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

        # Compute on device
        result = torch.addmm(input, mat1, mat2)

        # Compare results
        assert result.shape == (3, 5)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_addmm_different_sizes(self):
        """Test addmm with various tensor sizes"""

        test_cases = [
            ((2, 4), (2, 3), (3, 4)),  # 2x4 bias, 2x3 @ 3x4
            ((1, 1), (1, 5), (5, 1)),  # 1x1 bias, 1x5 @ 5x1
            ((10, 8), (10, 2), (2, 8)),  # 10x8 bias, 10x2 @ 2x8
            ((4, 4), (4, 4), (4, 4)),  # 4x4 bias, 4x4 @ 4x4
        ]

        for input_shape, mat1_shape, mat2_shape in test_cases:
            input = torch.randn(*input_shape).to("neuron")
            mat1 = torch.randn(*mat1_shape).to("neuron")
            mat2 = torch.randn(*mat2_shape).to("neuron")

            # Compute on CPU for reference
            input_cpu = input.cpu()
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.addmm(input, mat1, mat2)

            # Check shape and values
            assert result.shape == (mat1_shape[0], mat2_shape[1])
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    @assert_raises(RuntimeError, match="shapes cannot be multiplied")
    def test_addmm_dimension_mismatch_error(self):
        """Test that dimension mismatch raises appropriate error"""

        input = torch.randn(3, 5).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(5, 6).to("neuron")  # Incompatible dimensions

        torch.addmm(input, mat1, mat2)

    @assert_raises(RuntimeError)
    def test_addmm_wrong_dimensions_error(self):
        """Test that non-2D matrices raise appropriate error"""

        input = torch.randn(3, 5).to("neuron")

        # 3D tensor
        tensor_3d = torch.randn(2, 3, 4).to("neuron")
        tensor_2d = torch.randn(4, 5).to("neuron")

        torch.addmm(input, tensor_3d, tensor_2d)

        # 1D tensor
        tensor_1d = torch.randn(4).to("neuron")

        torch.addmm(input, tensor_2d, tensor_1d)

    @assert_raises(RuntimeError)
    def test_addmm_broadcasting_error(self):
        """Test that non-broadcastable input raises error"""

        # Input shape (3, 4) is not broadcastable to result shape (3, 5)
        input = torch.randn(3, 4).to("neuron")
        mat1 = torch.randn(3, 4).to("neuron")
        mat2 = torch.randn(4, 5).to("neuron")

        torch.addmm(input, mat1, mat2)

    def test_addmm_different_dtypes(self):
        """Test addmm with different data types"""

        dtypes = [torch.float32, torch.bfloat16, torch.float16]

        for dtype in dtypes:
            input = torch.randn(3, 5, dtype=dtype).to("neuron")
            mat1 = torch.randn(3, 4, dtype=dtype).to("neuron")
            mat2 = torch.randn(4, 5, dtype=dtype).to("neuron")

            # Compute on CPU for reference
            input_cpu = input.cpu()
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu)

            # Compute on device
            result = torch.addmm(input, mat1, mat2)

            # Check dtype is preserved
            assert result.dtype == dtype
            torch.testing.assert_close(
                result.cpu(),
                expected,
                rtol=1e-4 if dtype == torch.float32 else 1e-8,
                atol=1e-4 if dtype == torch.float32 else 1e-8,
            )

    def test_addmm_special_values(self):
        """Test addmm with special alpha/beta values"""

        input = torch.randn(2, 3).to("neuron")
        mat1 = torch.randn(2, 4).to("neuron")
        mat2 = torch.randn(4, 3).to("neuron")

        test_cases = [
            (1.0, 1.0),  # Default values
            (0.0, 1.0),  # Only input
            (1.0, 0.0),  # Only matmul
            (-1.0, 1.0),  # Subtract matmul
            (1.0, -1.0),  # Subtract input
            (2.0, 0.5),  # Scale both
        ]

        for alpha, beta in test_cases:
            # Compute on CPU for reference
            input_cpu = input.cpu()
            mat1_cpu = mat1.cpu()
            mat2_cpu = mat2.cpu()
            expected = torch.addmm(input_cpu, mat1_cpu, mat2_cpu, alpha=alpha, beta=beta)

            # Compute on device
            result = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

            # Compare results
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
