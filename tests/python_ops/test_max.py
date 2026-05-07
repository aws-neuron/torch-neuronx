"""Test that max operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMaxRegistration:
    """Test max operation registration and functionality."""

    def test_max_function_no_dim(self):
        """Test that torch.max works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).to("neuron")

        # Use torch.max function
        c = torch.max(a)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(5.0, dtype=torch.float32)  # Max value is 5
        torch.testing.assert_close(c.cpu(), expected)

    def test_max_method_no_dim(self):
        """Test that tensor.max() method works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32).to("neuron")

        # Use tensor.max() method
        c = a.max()

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(5.0, dtype=torch.float32)  # Max value is 5
        torch.testing.assert_close(c.cpu(), expected)

    def test_max_function_with_dim(self):
        """Test that torch.max works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each row
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Max along dimension 1 (columns)
        values, indices = torch.max(a, dim=1)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (2,)  # Dimension 1 is reduced
        assert indices.shape == (2,)  # Indices for each row
        expected_values = torch.tensor([3.0, 6.0], dtype=torch.float32)  # Max of each row
        expected_indices = torch.tensor([2, 2], dtype=torch.int64)  # Indices of max values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    def test_max_method_with_dim(self):
        """Test that tensor.max() method works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each column
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Max along dimension 0 (rows)
        values, indices = a.max(dim=0)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (3,)  # Dimension 0 is reduced
        assert indices.shape == (3,)  # Indices for each column
        expected_values = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)  # Max of each column
        expected_indices = torch.tensor([1, 1, 1], dtype=torch.int64)  # Indices of max values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    def test_max_with_keepdim(self):
        """Test max with keepdim parameter."""
        # Create neuron tensor
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Max along dimension 1 with keepdim=True
        values, indices = torch.max(a, dim=1, keepdim=True)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (2, 1)  # Dimension 1 is kept but has size 1
        assert indices.shape == (2, 1)  # Indices with keepdim
        expected_values = torch.tensor([[3.0], [6.0]], dtype=torch.float32)  # Max of each row
        expected_indices = torch.tensor([[2], [2]], dtype=torch.int64)  # Indices of max values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    def test_max_unary_out(self):
        """Test max.unary_out operation."""
        # Create neuron tensors
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).to("neuron")
        out = torch.empty((), dtype=torch.float32).to("neuron")

        # Use torch.max with out parameter
        result = torch.max(a, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.tensor(5.0, dtype=torch.float32)  # Max value is 5
        torch.testing.assert_close(out.cpu(), expected)

    def test_max_dim_out(self):
        """Test max.dim_max operation with output tensors."""
        # Create neuron tensors
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")
        values_out = torch.empty(2, dtype=torch.float32).to("neuron")
        indices_out = torch.empty(2, dtype=torch.int64).to("neuron")

        # Use torch.max with dim and out parameters
        values, indices = torch.max(a, dim=1, out=(values_out, indices_out))

        # Verify result and that out was modified in-place
        assert values is values_out
        assert indices is indices_out
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        expected_values = torch.tensor([3.0, 6.0], dtype=torch.float32)  # Max of each row
        expected_indices = torch.tensor([2, 2], dtype=torch.int64)  # Indices of max values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    def test_max_first_occurrence(self):
        """Test that max returns the first occurrence of the maximum value."""
        # Create neuron tensor with duplicate max values
        a = torch.tensor([[3.0, 3.0, 1.0], [6.0, 6.0, 5.0]], dtype=torch.float32).to("neuron")

        # Max along dimension 1
        values, indices = torch.max(a, dim=1)

        # Verify result - should return the first occurrence of the max value
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        expected_values = torch.tensor([3.0, 6.0], dtype=torch.float32)  # Max of each row
        expected_indices = torch.tensor([0, 0], dtype=torch.int64)  # First occurrence indices
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)
