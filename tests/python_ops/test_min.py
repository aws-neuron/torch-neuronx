"""Test that min operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMinRegistration:
    """Test min operation registration and functionality."""

    def test_min_function_no_dim(self):
        """Test that torch.min works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32).to("neuron")

        # Use torch.min function
        c = torch.min(a)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(1.0, dtype=torch.float32)  # Min value is 1
        torch.testing.assert_close(c.cpu(), expected)

    def test_min_method_no_dim(self):
        """Test that tensor.min() method works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).to("neuron")

        # Use tensor.min() method
        c = a.min()

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(1.0, dtype=torch.float32)  # Min value is 1
        torch.testing.assert_close(c.cpu(), expected)

    @pytest.mark.xfail(reason="dim based torch.min and torch.max are not supported right now")
    def test_min_function_with_dim(self):
        """Test that torch.min works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each row
        a = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=torch.float32).to("neuron")

        # Min along dimension 1 (columns)
        values, indices = torch.min(a, dim=1)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (2,)  # Dimension 1 is reduced
        assert indices.shape == (2,)  # Indices for each row
        expected_values = torch.tensor([1.0, 4.0], dtype=torch.float32)  # Min of each row
        expected_indices = torch.tensor([2, 2], dtype=torch.int32)  # Indices of min values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    @pytest.mark.xfail(reason="dim based torch.min and torch.max are not supported right now")
    def test_min_method_with_dim(self):
        """Test that tensor.min() method works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each column
        a = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=torch.float32).to("neuron")

        # Min along dimension 0 (rows)
        values, indices = a.min(dim=0)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (3,)  # Dimension 0 is reduced
        assert indices.shape == (3,)  # Indices for each column
        expected_values = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)  # Min of each column
        expected_indices = torch.tensor([0, 0, 0], dtype=torch.int32)  # Indices of min values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    @pytest.mark.xfail(reason="dim based torch.min and torch.max are not supported right now")
    def test_min_with_keepdim(self):
        """Test min with keepdim parameter."""
        # Create neuron tensor
        a = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=torch.float32).to("neuron")

        # Min along dimension 1 with keepdim=True
        values, indices = torch.min(a, dim=1, keepdim=True)

        # Verify result
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        assert values.shape == (2, 1)  # Dimension 1 is kept but has size 1
        assert indices.shape == (2, 1)  # Indices with keepdim
        expected_values = torch.tensor([[1.0], [4.0]], dtype=torch.float32)  # Min of each row
        expected_indices = torch.tensor([[2], [2]], dtype=torch.int32)  # Indices of min values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    def test_min_unary_out(self):
        """Test min.unary_out operation."""
        # Create neuron tensors
        a = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32).to("neuron")
        out = torch.empty((), dtype=torch.float32).to("neuron")

        # Use torch.min with out parameter
        result = torch.min(a, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.tensor(1.0, dtype=torch.float32)  # Min value is 1
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.xfail(reason="dim based torch.min and torch.max are not supported right now")
    def test_min_dim_out(self):
        """Test min.dim_min operation with output tensors."""
        # Create neuron tensors
        a = torch.tensor([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=torch.float32).to("neuron")
        values_out = torch.empty(2, dtype=torch.float32).to("neuron")
        indices_out = torch.empty(2, dtype=torch.int32).to("neuron")

        # Use torch.min with dim and out parameters
        values, indices = torch.min(a, dim=1, out=(values_out, indices_out))

        # Verify result and that out was modified in-place
        assert values is values_out
        assert indices is indices_out
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        expected_values = torch.tensor([1.0, 4.0], dtype=torch.float32)  # Min of each row
        expected_indices = torch.tensor([2, 2], dtype=torch.int32)  # Indices of min values
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)

    @pytest.mark.xfail(reason="dim based torch.min and torch.max are not supported right now")
    def test_min_first_occurrence(self):
        """Test that min returns the first occurrence of the minimum value."""
        # Create neuron tensor with duplicate min values
        a = torch.tensor([[1.0, 1.0, 3.0], [4.0, 4.0, 5.0]], dtype=torch.float32).to("neuron")

        # Min along dimension 1
        values, indices = torch.min(a, dim=1)

        # Verify result - should return the first occurrence of the min value
        assert values.device.type == "neuron"
        assert indices.device.type == "neuron"
        expected_values = torch.tensor([1.0, 4.0], dtype=torch.float32)  # Min of each row
        expected_indices = torch.tensor([0, 0], dtype=torch.int32)  # First occurrence indices
        torch.testing.assert_close(values.cpu(), expected_values)
        torch.testing.assert_close(indices.cpu(), expected_indices)
