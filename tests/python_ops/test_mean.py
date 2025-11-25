"""Test that mean operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMeanRegistration:
    """Test mean operation registration and functionality."""

    def test_mean_function_no_dim(self):
        """Test that torch.mean works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 2

        # Use torch.mean function
        c = torch.mean(a)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(2.0, dtype=torch.float32)  # Mean of all 2's is 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_method_no_dim(self):
        """Test that tensor.mean() method works with neuron tensors (no dimension specified)."""
        # Create neuron tensor
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 3

        # Use tensor.mean() method
        c = a.mean()

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == ()  # Scalar tensor
        expected = torch.tensor(3.0, dtype=torch.float32)  # Mean of all 3's is 3
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_function_with_dim(self):
        """Test that torch.mean works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each row
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Mean along dimension 1 (columns)
        c = torch.mean(a, dim=1)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (2,)  # Dimension 1 is reduced
        expected = torch.tensor([2.0, 5.0], dtype=torch.float32)  # Mean of each row
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_method_with_dim(self):
        """Test that tensor.mean() method works with neuron tensors (with dimension specified)."""
        # Create neuron tensor with different values in each column
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Mean along dimension 0 (rows)
        c = a.mean(dim=0)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (3,)  # Dimension 0 is reduced
        expected = torch.tensor([2.5, 3.5, 4.5], dtype=torch.float32)  # Mean of each column
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_with_keepdim(self):
        """Test mean with keepdim parameter."""
        # Create neuron tensor
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")

        # Mean along dimension 1 with keepdim=True
        c = torch.mean(a, dim=1, keepdim=True)

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (2, 1)  # Dimension 1 is kept but has size 1
        expected = torch.tensor([[2.0], [5.0]], dtype=torch.float32)  # Mean of each row
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_with_dtype(self):
        """Test mean with dtype parameter."""
        # Create neuron tensor with float16
        a = torch.ones(16, 16, dtype=torch.float16).to("neuron") * 2

        # Mean with dtype=torch.float32
        c = torch.mean(a, dtype=torch.float32)

        # Verify result
        assert c.device.type == "neuron"
        assert c.dtype == torch.float32
        expected = torch.tensor(2.0, dtype=torch.float32)  # Mean of all 2's is 2
        torch.testing.assert_close(c.cpu(), expected)

    def test_mean_dtype_out(self):
        """Test mean.dtype_out operation."""
        # Create neuron tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron") * 2
        out = torch.empty((), dtype=torch.float32).to("neuron")

        # Use torch.mean with out parameter
        result = torch.mean(a, dtype=torch.float32, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.tensor(2.0, dtype=torch.float32)  # Mean of all 2's is 2
        torch.testing.assert_close(out.cpu(), expected)

    def test_mean_dim_out(self):
        """Test mean.out operation with dimension."""
        # Create neuron tensors
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32).to("neuron")
        out = torch.empty(2, dtype=torch.float32).to("neuron")

        # Use torch.mean with dim and out parameters
        result = torch.mean(a, dim=1, out=out)

        # Verify result and that out was modified in-place
        assert result is out
        assert out.device.type == "neuron"
        expected = torch.tensor([2.0, 5.0], dtype=torch.float32)  # Mean of each row
        torch.testing.assert_close(out.cpu(), expected)

    def test_mean_multiple_dims(self):
        """Test mean with multiple dimensions."""
        # Create neuron tensor
        a = torch.ones(4, 5, 6, dtype=torch.float32).to("neuron") * 2

        # Mean along dimensions 1 and 2
        c = torch.mean(a, dim=(1, 2))

        # Verify result
        assert c.device.type == "neuron"
        assert c.shape == (4,)  # Dimensions 1 and 2 are reduced
        expected = torch.ones(4, dtype=torch.float32) * 2  # Mean of 2's is 2
        torch.testing.assert_close(c.cpu(), expected)
