"""Test that sub operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSubRegistration:
    """Test sub operation registration and functionality."""

    def test_sub_runs_on_neuron(self):
        """Test that sub runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([5.0, 6.0, 7.0], device="neuron")
            b = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            result = torch.sub(a, b)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::sub")

    def test_sub_inplace_runs_on_neuron(self):
        """Test that sub_ runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([5.0, 6.0, 7.0], device="neuron")
            b = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            original_data_ptr = a.data_ptr()
            result = a.sub_(b)
            assert result is a
            assert result.data_ptr() == original_data_ptr
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("sub")

    def test_sub_operator_syntax(self):
        """Test that a - b syntax works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use - operator (calls sub.Tensor)
        c = a - b

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_function_syntax(self):
        """Test that torch.sub works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.sub function
        c = torch.sub(a, b)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_with_alpha(self):
        """Test sub with alpha parameter."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 5
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.sub with alpha
        c = torch.sub(a, b, alpha=2)

        # Verify result: a - alpha * b = 5 - 2 * 1 = 3
        assert c.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 3
        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_out(self):
        """Test sub.out operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        out = torch.empty(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.sub with out parameter
        result = torch.sub(a, b, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(out.cpu(), expected)

    def test_sub_inplace(self):
        """Test in-place sub operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Store original tensor id
        original_id = id(a)

        # Use -= operator (calls sub_ which uses sub.out internally)
        a -= b

        # Verify it was in-place
        assert id(a) == original_id
        assert a.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(a.cpu(), expected)

    def test_sub_mixed_devices(self):
        """Test that mixing CPU and neuron tensors raises error."""
        # One tensor on neuron, one on CPU
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(16, 16, dtype=torch.float32)  # CPU

        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            _ = a - b

    def test_sub_small_tensor(self):
        """Test that small tensors work with XLA implementation."""
        # Create small neuron tensors (< 256 elements)
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron") * 3
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Since we removed can_handle, XLA will handle all cases
        c = a - b

        # Verify result
        assert c.device.type == "neuron"
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_first_operand_python_scalar(self):
        """Test subtraction with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = b_cpu - a_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform subtraction
        c = b_cpu - a_neuron

        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_second_operand_python_scalar(self):
        """Test subtraction with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = a_cpu - b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform subtraction
        c = a_neuron - b_cpu

        torch.testing.assert_close(c.cpu(), expected)

    def test_sub_type_promotions(self):
        """Test subtraction with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0], dtype=torch.float32)
        b_cpu = torch.tensor(2, dtype=torch.int32)
        expected = a_cpu - b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform subtraction
        c = a_neuron - b_neuron

        torch.testing.assert_close(c.cpu(), expected)
