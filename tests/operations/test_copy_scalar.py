"""Tests for copy_ with scalar source."""

import pytest
import torch


class TestCopyScalar:
    """Test copy_ operation with Python scalar as source."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32])
    def test_copy_scalar_float(self, dtype):
        """Test copy_ with float scalar."""
        dst = torch.empty(1, dtype=dtype, device="neuron")
        dst.copy_(3.14)
        assert torch.allclose(dst.cpu(), torch.tensor([3.14], dtype=dtype))

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8])
    def test_copy_scalar_int(self, dtype):
        """Test copy_ with int scalar."""
        dst = torch.empty(1, dtype=dtype, device="neuron")
        dst.copy_(42)
        assert dst.cpu().item() == 42

    def test_copy_scalar_bool(self):
        """Test copy_ with bool scalar."""
        dst = torch.empty(1, dtype=torch.bool, device="neuron")
        dst.copy_(True)
        assert dst.cpu().item()

        dst.copy_(False)
        assert not dst.cpu().item()

    def test_copy_scalar_dtype_conversion(self):
        """Test copy_ scalar with implicit dtype conversion."""
        # Float scalar to int tensor
        dst = torch.empty(1, dtype=torch.int32, device="neuron")
        dst.copy_(3.7)
        assert dst.cpu().item() == 3  # Truncated

        # Int scalar to float tensor
        dst = torch.empty(1, dtype=torch.float32, device="neuron")
        dst.copy_(42)
        assert dst.cpu().item() == 42.0

    def test_copy_scalar_broadcast(self):
        """Test copy_ scalar broadcasts to fill entire tensor."""
        dst = torch.empty(3, 4, dtype=torch.float32, device="neuron")
        dst.copy_(5.0)
        expected = torch.full((3, 4), 5.0, dtype=torch.float32)
        assert torch.allclose(dst.cpu(), expected)

    def test_copy_scalar_broadcast_int(self):
        """Test copy_ int scalar broadcasts to fill entire tensor."""
        dst = torch.empty(2, 3, dtype=torch.int32, device="neuron")
        dst.copy_(42)
        expected = torch.full((2, 3), 42, dtype=torch.int32)
        assert torch.equal(dst.cpu(), expected)
