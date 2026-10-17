"""Test cases for rank2 contiguous kernel."""

import os

import pytest
import torch

import torch_neuronx


class TestContiguousRank2:
    """Test cases for rank2 contiguous operations."""

    def test_simple_slice_2d(self):
        """Test contiguous on a simple 2D slice."""
        # Create tensor on CPU first
        x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        # Move to neuron device
        x = x_cpu.to("neuron")

        # Create a non-contiguous slice
        y = x[:, 1:3]  # Select columns 1 and 2
        assert not y.is_contiguous()
        assert y.storage_offset() == 1

        # Make it contiguous
        z = y.contiguous()
        assert z.is_contiguous()

        # Check the result
        z_cpu = z.to("cpu")
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        assert torch.allclose(z_cpu, expected)

    def test_non_contiguous_row_column_slice(self):
        """Test contiguous on non-contiguous slices."""
        x_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        x = x_cpu.to("neuron")

        # Select middle rows and columns - this creates non-contiguous
        y = x[1:3, 1:4]
        assert not y.is_contiguous()
        assert y.storage_offset() > 0

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor([[6.0, 7.0, 8.0], [11.0, 12.0, 13.0]])
        assert torch.allclose(z_cpu, expected)

    def test_strided_slice(self):
        """Test contiguous with strided access."""
        x_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        x = x_cpu.to("neuron")

        # Every other column
        y = x[:, ::2]  # Columns 0, 2, 4
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor(
            [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0], [12.0, 14.0, 16.0], [18.0, 20.0, 22.0]]
        )
        assert torch.allclose(z_cpu, expected)

    def test_transpose(self):
        """Test contiguous after transpose."""
        x_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        x = x_cpu.to("neuron")

        # Transpose creates non-contiguous tensor
        y = x.t()
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
        assert torch.allclose(z_cpu, expected)

    def test_complex_slice(self):
        """Test contiguous with complex slicing patterns."""
        x_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        x = x_cpu.to("neuron")

        # Complex slice: skip first row, take every other column starting from column 1
        y = x[1:, 1::2]  # Rows 1-4, columns 1, 3, 5
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor(
            [[7.0, 9.0, 11.0], [13.0, 15.0, 17.0], [19.0, 21.0, 23.0], [25.0, 27.0, 29.0]]
        )
        assert torch.allclose(z_cpu, expected)

    def test_single_element_slice(self):
        """Test contiguous on single element slices."""
        x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        x = x_cpu.to("neuron")

        # Single column
        y = x[:, 2:3]
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor([[2.0], [6.0], [10.0]])
        assert torch.allclose(z_cpu, expected)

    def test_large_tensor(self):
        """Test contiguous on larger tensors."""
        # Create a larger tensor
        x_cpu = torch.arange(1000, dtype=torch.float32).reshape(100, 10)
        x = x_cpu.to("neuron")

        # Take a slice
        y = x[10:20, 2:8]
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        # Verify a few elements
        z_cpu = z.to("cpu")
        assert z_cpu.shape == (10, 6)
        assert z_cpu[0, 0] == 102.0  # Row 10, col 2 in original
        assert z_cpu[9, 5] == 197.0  # Row 19, col 7 in original

    def test_different_dtypes(self):
        """Test contiguous with different data types."""
        dtypes = [torch.float32, torch.float16, torch.int32]

        for dtype in dtypes:
            if dtype == torch.int32:
                x_cpu = torch.arange(12, dtype=dtype).reshape(3, 4)
            else:
                x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype)

            x = x_cpu.to("neuron")
            y = x[:, 1:3]

            z = y.contiguous()
            assert z.is_contiguous()

            z_cpu = z.to("cpu")
            if dtype == torch.int32:
                expected = torch.tensor([[1, 2], [5, 6], [9, 10]], dtype=dtype)
            else:
                expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]], dtype=dtype)
            assert torch.allclose(z_cpu.float(), expected.float())

    def test_zero_stride_dimension(self):
        """Test contiguous with broadcasting (zero stride)."""
        x_cpu = torch.tensor([[1.0, 2.0, 3.0]])
        x = x_cpu.to("neuron")

        # Expand creates zero strides
        y = x.expand(3, 3)
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()

        z_cpu = z.to("cpu")
        expected = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert torch.allclose(z_cpu, expected)

    def test_empty_tensor(self):
        """Test contiguous on empty tensors."""
        x_cpu = torch.empty(0, 5, dtype=torch.float32)
        x = x_cpu.to("neuron")

        z = x.contiguous()
        assert z.is_contiguous()
        assert z.shape == (0, 5)

    def test_already_contiguous(self):
        """Test that contiguous on already contiguous tensor returns same tensor."""
        x_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        x = x_cpu.to("neuron")

        assert x.is_contiguous()
        z = x.contiguous()

        # Should return the same tensor (no copy needed)
        assert z.data_ptr() == x.data_ptr()
