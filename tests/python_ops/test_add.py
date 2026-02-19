"""Test that add operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestAddRegistration:
    """Test add operation registration and functionality."""

    def test_add_runs_on_neuron(self):
        """Test that add runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            b = torch.tensor([4.0, 5.0, 6.0], device="neuron")
            result = torch.add(a, b)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::add")

    def test_add_inplace_runs_on_neuron(self):
        """Test that add_ runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            b = torch.tensor([4.0, 5.0, 6.0], device="neuron")
            original_data_ptr = a.data_ptr()
            result = a.add_(b)
            assert result is a
            assert result.data_ptr() == original_data_ptr
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("add")

    @assert_raises(RuntimeError, match="Cannot (add|process) non-contiguous tensor.*")
    def test_add_inplace_non_contiguous_output(self, monkeypatch):
        """Test in-place add operations on tensor slices"""
        monkeypatch.setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1")
        m1 = torch.ones(10, 10, device="neuron")
        m1[:, 3].add_(2)

    def test_add_operator_syntax(self):
        """Test that a + b syntax works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use + operator (calls add.Tensor)
        c = a + b

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_function_syntax(self):
        """Test that torch.add works with neuron tensors."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add function
        c = torch.add(a, b)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (16, 16)
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_with_alpha(self):
        """Test add with alpha parameter."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add with alpha
        c = torch.add(a, b, alpha=2)

        # Verify result: a + alpha * b = 1 + 2 * 1 = 3
        assert c.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 3
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_out(self):
        """Test add.out operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        out = torch.empty(16, 16, dtype=torch.float32).to("neuron")

        # Use torch.add with out parameter
        result = torch.add(a, b, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(out.cpu(), expected)

    def test_add_inplace(self):
        """Test in-place add operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Store original tensor id
        original_id = id(a)

        # Use += operator (calls add_ which uses add.out internally)
        a += b

        # Verify it was in-place
        assert id(a) == original_id
        assert a.device.type == "neuron"
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(a.cpu(), expected)

    def test_add_mixed_devices(self):
        """Test that mixing CPU and neuron tensors promotes cpu tensor to device."""
        # One tensor on neuron, one on CPU
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32)  # CPU

        # Should raise RuntimeError for non-scalar cross-device operation
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            _ = a + b

    def test_add_small_tensor(self):
        """Test that small tensors work with XLA implementation."""
        # Create small neuron tensors (< 256 elements)
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Since we removed can_handle, XLA will handle all cases
        c = a + b

        # Verify result
        assert c.device.type == "neuron"
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_add_first_operand_python_scalar(self):
        """Test addition with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = b_cpu + a_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = b_cpu + a_neuron

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_second_operand_python_scalar(self):
        """Test addition with scalar"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
        b_cpu = 2.0
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_cpu

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_type_promotions(self):
        """Test addition with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0], dtype=torch.float32)
        b_cpu = torch.tensor(2, dtype=torch.int32)
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_neuron

        torch.testing.assert_close(c.cpu(), expected)

    def test_add_with_int64(self):
        """Test addition with int64 tensor"""
        # Create CPU tensors
        a_cpu = torch.tensor([4, 6, 10], dtype=torch.int64)
        b_cpu = 0
        expected = a_cpu + b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform addition
        c = a_neuron + b_cpu

        torch.testing.assert_close(c.cpu(), expected)
