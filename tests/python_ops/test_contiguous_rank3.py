"""Test cases for RANK3 contiguous kernel - basic cases."""

import os

import pytest
import torch

import torch_neuronx


class TestContiguousRank3:
    """Test cases for RANK3 contiguous operations (collapsed rank = 3)."""

    def test_simple_3d_middle_stride(self):
        """Test basic 3D tensor with strided middle dimension."""
        # Create 3D tensor
        x_cpu = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        x = x_cpu.to("neuron")

        # Stride in middle dimension
        y = x[:, ::2, :]  # Shape: [3, 2, 5]
        assert not y.is_contiguous()

        # This should use RANK3 kernel
        # Collapsed shape will be (3, 2, 5) - cannot collapse further
        z = y.contiguous()
        assert z.is_contiguous()

        # Verify correctness
        expected = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 14.0]],
                [[20.0, 21.0, 22.0, 23.0, 24.0], [30.0, 31.0, 32.0, 33.0, 34.0]],
                [[40.0, 41.0, 42.0, 43.0, 44.0], [50.0, 51.0, 52.0, 53.0, 54.0]],
            ]
        )
        assert torch.allclose(z.cpu(), expected)

    def test_4d_to_rank3_collapse(self):
        """Test 4D tensor that collapses to rank 3."""
        # Create 4D tensor with specific stride pattern
        x_cpu = torch.arange(240, dtype=torch.float32).reshape(2, 3, 4, 10)
        x = x_cpu.to("neuron")

        # Stride in second dimension
        y = x[:, ::2, :, :]  # Shape: [2, 2, 4, 10]
        assert not y.is_contiguous()

        # This should collapse to (2, 2, 40) and use RANK3
        z = y.contiguous()
        assert z.is_contiguous()

        # Verify shape and some values
        assert z.shape == (2, 2, 4, 10)
        expected = x_cpu[:, ::2, :, :].contiguous()
        assert torch.allclose(z.cpu(), expected)

    def test_non_unit_strides_multiple_dims(self):
        """Test tensor with non-unit strides in multiple dimensions."""
        # Create base tensor
        x_cpu = torch.arange(480, dtype=torch.float32).reshape(4, 6, 20)
        x = x_cpu.to("neuron")

        # Multiple strides that prevent full collapse
        y = x[::2, ::3, :]  # Shape: [2, 2, 20]
        assert not y.is_contiguous()

        # Make contiguous
        z = y.contiguous()
        assert z.is_contiguous()

        # Verify
        expected = x_cpu[::2, ::3, :].contiguous()
        assert torch.allclose(z.cpu(), expected)

    def test_rank3_with_offset(self):
        """Test RANK3 operation with storage offset."""
        # Create larger tensor and take a slice
        x_cpu = torch.arange(1000, dtype=torch.float32).reshape(10, 10, 10)
        x = x_cpu.to("neuron")

        # Take a sub-block first
        sub_block = x[2:8, :, :]  # Has storage offset

        # Then stride in middle dimension
        y = sub_block[:, ::2, :]  # Shape: [6, 5, 10]
        assert not y.is_contiguous()
        assert y.storage_offset() > 0

        # Make contiguous
        z = y.contiguous()
        assert z.is_contiguous()
        assert z.storage_offset() == 0

        # Verify
        expected = x_cpu[2:8, ::2, :].contiguous()
        assert torch.allclose(z.cpu(), expected)

    def test_rank3_empty_tensor(self):
        """Test RANK3 operation on empty tensor."""
        # Empty tensor with rank 3 pattern
        x_cpu = torch.empty(0, 4, 5, dtype=torch.float32)
        x = x_cpu.to("neuron")

        # Even with stride, empty stays empty
        y = x[:, ::2, :]
        z = y.contiguous()

        assert z.shape == (0, 2, 5)
        assert z.is_contiguous()

    def test_rank3_single_element_middle_dim(self):
        """Test RANK3 with single element in middle dimension after striding."""
        x_cpu = torch.arange(30, dtype=torch.float32).reshape(3, 2, 5)
        x = x_cpu.to("neuron")

        # Stride that results in single element in middle
        y = x[:, ::2, :]  # Shape: [3, 1, 5]
        assert not y.is_contiguous()

        # Make contiguous
        z = y.contiguous()
        assert z.is_contiguous()

        # Verify
        expected = x_cpu[:, ::2, :].contiguous()
        assert torch.allclose(z.cpu(), expected)

    def test_rank3_performance_characteristics(self):
        """Test to understand RANK3 performance patterns."""
        # Large tensor to see performance benefits
        x_cpu = torch.randn(10, 100, 1000, dtype=torch.float32)
        x = x_cpu.to("neuron")

        # Create RANK3 pattern
        y = x[:, ::10, :]  # Shape: [10, 10, 1000]
        assert not y.is_contiguous()

        # The RANK3 kernel should process this as:
        # for i in range(10):      # outer loop
        #     for j in range(10):  # middle loop
        #         memcpy(1000 elements)  # bulk copy

        z = y.contiguous()
        assert z.is_contiguous()

        # Verify correctness on sample
        assert torch.allclose(z.cpu()[0, 0, :10], x_cpu[0, 0, :10])
