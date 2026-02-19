import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestMul:
    """Test cases for element-wise multiplication operation"""

    def test_mul_runs_on_neuron(self):
        """Test that mul runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([2.0, 3.0, 4.0], device="neuron")
            b = torch.tensor([5.0, 6.0, 7.0], device="neuron")
            result = torch.mul(a, b)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::mul")

    def test_mul_basic(self):
        """Test basic element-wise multiplication"""
        # Create CPU tensors
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b_cpu = torch.tensor([2.0, 3.0, 4.0, 5.0])

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform multiplication
        result_neuron = torch.mul(a_neuron, b_neuron)

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_out(self):
        """Test multiplication with output tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b_cpu = torch.tensor([2.0, 3.0, 4.0, 5.0])
        out_cpu = torch.empty_like(a_cpu)

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")
        out_neuron = torch.empty_like(a_neuron)

        # Perform multiplication with output tensor
        result_neuron = torch.mul(a_neuron, b_neuron, out=out_neuron)

        # Verify result is the same as output tensor
        assert result_neuron is out_neuron

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu, out=out_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_broadcast(self):
        """Test multiplication with broadcasting"""
        # Create CPU tensors with different shapes
        a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_cpu = torch.tensor([2.0, 3.0])

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform multiplication (should broadcast)
        result_neuron = torch.mul(a_neuron, b_neuron)

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_different_dtypes(self):
        """Test multiplication with different data types"""
        dtypes = [torch.float32, torch.bfloat16, torch.float16]

        for dtype in dtypes:
            # Create CPU tensors
            a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
            b_cpu = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype)

            # Move to Neuron device
            a_neuron = a_cpu.to("privateuseone")
            b_neuron = b_cpu.to("privateuseone")

            # Perform multiplication
            result_neuron = torch.mul(a_neuron, b_neuron)

            # Compare with CPU result
            expected = torch.mul(a_cpu, b_cpu)
            torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_empty(self):
        """Test multiplication with empty tensors"""
        # Create empty CPU tensors
        a_cpu = torch.empty(0)
        b_cpu = torch.empty(0)

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform multiplication
        result_neuron = torch.mul(a_neuron, b_neuron)

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_scalar_broadcast(self):
        """Test multiplication with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b_cpu = torch.tensor(2.0)  # scalar-like tensor

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform multiplication
        result_neuron = torch.mul(a_neuron, b_neuron)

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_true_scalar(self):
        """Test multiplication with Python scalar"""
        # Create CPU tensor
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scalar = 2.5

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform multiplication with scalar
        result_neuron = torch.mul(a_neuron, scalar)

        # Compare with CPU result
        expected = torch.mul(a_cpu, scalar)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_scalar_first(self):
        """Test multiplication with float scalar as first argument"""
        # Create CPU tensor
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scalar = 2.5

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform multiplication with scalar first
        result_neuron = torch.mul(scalar, a_neuron)

        # Compare with CPU result
        expected = torch.mul(scalar, a_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_with_different_devices_with_tensor_scalar(self):
        """Test mul with Neuron tensor and CPU scalar tensor

        Scalar tensors (ndim=0) on CPU should be automatically moved to Neuron device,
        matching CUDA semantics.
        """
        # Create tensors
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b_cpu = torch.tensor(2.0)  # scalar tensor on CPU

        # Move only first tensor to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform multiplication (CPU scalar tensor should be moved to Neuron)
        result_neuron = torch.mul(a_neuron, b_cpu)

        # Compare with CPU result
        expected = torch.mul(a_cpu, b_cpu)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_mul_with_different_devices(self):
        """Test mul with non-scalar tensors on different devices fails

        Non-scalar tensors on different devices should raise RuntimeError,
        matching CUDA behavior.
        """
        # Create CPU tensors
        a_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b_cpu = torch.tensor([2.0, 3.0, 4.0, 5.0])

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Should raise RuntimeError for non-scalar cross-device operation
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            torch.mul(a_neuron, b_cpu)

    def test_mul_transpose_contiguous(self):
        """Test multiplication with transposed tensor and broadcasting

        The transpose is used to create non-contiguous input.
        """
        # Create CPU tensors
        a_cpu = torch.randn(8, 4)
        b_cpu = torch.randn(4, 1)

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Transpose and multiply on CPU
        a_transposed_cpu = a_cpu.transpose(0, 1)
        expected = a_transposed_cpu * b_cpu

        with track_neuron_ops():
            # Transpose and multiply on Neuron
            a_transposed_neuron = a_neuron.transpose(0, 1)
            result_neuron = a_transposed_neuron * b_neuron
            assert_op_runs_on_neuron("aten::mul")

        # Compare results
        torch.testing.assert_close(result_neuron.cpu(), expected)
