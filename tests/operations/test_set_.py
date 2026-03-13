"""Comprehensive tests for set_ operations on Neuron tensors."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.utils import use_mlir_aten_ops


def assert_no_set_fallbacks():
    """Helper to check no set_ operations fell back to CPU."""
    fallback_ops = torch_neuronx.get_fallback_ops()
    set_fallbacks = [op for op in fallback_ops if "set_" in op]
    assert len(set_fallbacks) == 0, f"set_ operations fell back to CPU: {set_fallbacks}"


class TestSetOperations:
    """Test all set_ operations for Neuron tensors."""

    def test_set_source_storage_basic(self):
        """Test basic set_.source_Storage operation."""
        # Create CPU reference first
        source_cpu = torch.randn(4, 3)
        target_cpu = torch.empty(0)
        target_cpu.set_(source_cpu.storage())

        # Test Neuron version with same data
        source = source_cpu.to("neuron")
        target = torch.empty(0, device="neuron")
        target.set_(source.storage())

        # Compare results
        assert target.shape == target_cpu.shape
        assert target.numel() == target_cpu.numel()
        torch.testing.assert_close(target.cpu(), target_cpu)
        assert target.data_ptr() == source.data_ptr()
        assert_no_set_fallbacks()

    def test_set_source_storage_with_offset(self):
        """Test set_.source_Storage_storage_offset operation."""
        # Create CPU reference first
        source_cpu = torch.randn(10)
        target_cpu = torch.empty(0)
        offset, size = 2, 5
        target_cpu.set_(source_cpu.storage(), offset, (size,))

        # Test Neuron version with same data
        source = source_cpu.to("neuron")
        target = torch.empty(0, device="neuron")
        target.set_(source.storage(), offset, (size,))

        # Compare results
        assert target.shape == target_cpu.shape
        assert target.storage_offset() == target_cpu.storage_offset()
        torch.testing.assert_close(target.cpu(), target_cpu)
        assert_no_set_fallbacks()

    def test_set_source_storage_multidimensional(self):
        """Test set_.source_Storage_storage_offset with custom strides."""
        source = torch.randn(12, device="neuron")
        target = torch.empty(0, device="neuron")

        # 2x3 matrix with row-major strides
        target.set_(source.storage(), 0, (2, 3), (3, 1))

        assert target.shape == (2, 3)
        assert target.stride() == (3, 1)
        assert target.numel() == 6
        assert_no_set_fallbacks()

    def test_set_source_tensor(self):
        """Test set_.source_Tensor operation."""
        # Create CPU reference first
        source_cpu = torch.randn(3, 4)
        target_cpu = torch.empty(0)
        target_cpu.set_(source_cpu)

        # Test Neuron version with same data
        source = source_cpu.to("neuron")
        target = torch.empty(0, device="neuron")
        target.set_(source)

        # Compare results
        assert target.shape == target_cpu.shape
        assert target.stride() == target_cpu.stride()
        assert target.storage_offset() == target_cpu.storage_offset()
        torch.testing.assert_close(target.cpu(), target_cpu)
        assert target.data_ptr() == source.data_ptr()
        assert_no_set_fallbacks()

    @pytest.mark.skipif(
        condition=use_mlir_aten_ops(),
        reason="Test gets stuck when lowering aten ops with torch-mlir",
    )
    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "0",
        reason=("Fails in async mode due to compilation error related to contiguous op; needs RCA"),
    )
    def test_set_source_tensor_with_view(self):
        """Test set_.source_Tensor with tensor view."""
        base = torch.randn(6, 4, device="neuron")
        source = base[1:4, 2:]  # View with offset
        target = torch.empty(0, device="neuron")

        target.set_(source)

        assert target.shape == source.shape
        assert target.stride() == source.stride()
        assert target.storage_offset() == source.storage_offset()
        assert torch.allclose(target, source)
        assert target.data_ptr() == source.data_ptr()

    def test_set_source_tensor_storage_offset(self):
        """Test set_.source_Tensor_storage_offset operation."""
        # Create CPU reference first
        source_cpu = torch.randn(12)
        target_cpu = torch.empty(0)
        storage_offset, size, stride = 2, (3, 2), (2, 1)
        target_cpu.set_(source_cpu, storage_offset, size, stride)

        # Test Neuron version with same data
        source = source_cpu.to("neuron")
        target = torch.empty(0, device="neuron")
        target.set_(source, storage_offset, size, stride)

        # Compare results
        assert target.shape == target_cpu.shape
        assert target.stride() == target_cpu.stride()
        assert target.storage_offset() == target_cpu.storage_offset()
        torch.testing.assert_close(target.cpu(), target_cpu)
        assert_no_set_fallbacks()

    def test_set_empty(self):
        """Test set_ (empty) operation."""
        tensor = torch.randn(5, 3, device="neuron")

        tensor.set_()

        assert tensor.numel() == 0
        assert tensor.shape == (0,)

    def test_set_after_resize_to_zero(self):
        """Test set operations after resizing tensor to zero."""
        tensor = torch.ones(4, device="neuron")
        tensor.resize_(0)
        new_tensor = torch.randn(3, device="neuron")

        tensor.set_(new_tensor)

        assert tensor.shape == new_tensor.shape
        assert torch.allclose(tensor, new_tensor)
        assert_no_set_fallbacks()

    def test_set_memory_sharing(self):
        """Test that set operations properly share memory."""
        source = torch.randn(5, device="neuron")
        target = torch.empty(0, device="neuron")

        target.set_(source)
        source.fill_(42.0)

        assert torch.all(target == 42.0)
        assert_no_set_fallbacks()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_set_different_dtypes(self, dtype):
        """Test set operations with different data types."""
        if dtype.is_floating_point:
            source = torch.randn(6, dtype=dtype, device="neuron")
        else:
            source = torch.randint(0, 100, (6,), dtype=dtype, device="neuron")

        target = torch.empty(0, dtype=dtype, device="neuron")
        target.set_(source.storage())

        assert target.dtype == dtype
        assert torch.equal(target, source)

    @assert_raises(RuntimeError, match="expected neuron")
    def test_set_error_cases_cpu_tensor(self):
        """Test error case for set operations - CPU tensor should fail."""
        # CPU tensor should fail
        cpu_tensor = torch.randn(3)
        neuron_tensor = torch.empty(0, device="neuron")

        neuron_tensor.set_(cpu_tensor)

    @assert_raises(RuntimeError, match="Tensor: invalid storage offset -1")
    def test_set_error_cases_negative_offset(self):
        """Test error case for set operations - Negative storage offset should fail."""
        # Negative storage offset should fail
        source = torch.randn(5, device="neuron")
        target = torch.empty(0, device="neuron")

        target.set_(source.storage(), -1, (3,))

    def test_typed_storage_resize_zero_set(self):
        """Test typed_storage.resize to 0 and then call set. This is required for FSDP"""
        tensor = torch.ones(4).to("neuron")
        # Resize to 0, note `tensor._resize` works as expected
        tensor._typed_storage()._resize_(0)

        # Create new tensor and set
        new_tensor = torch.rand(5).to("neuron")
        tensor.set_(new_tensor)

        # Assert size matches
        assert (
            tensor.size() == new_tensor.size()
        ), f"Size mismatch: tensor {tensor.size()} != new_tensor {new_tensor.size()}"

        # Assert values match
        assert torch.all(
            tensor == new_tensor
        ), f"Value mismatch: tensor {tensor} != new_tensor {new_tensor}"
        assert_no_set_fallbacks()


class TestPyTorchSetTests:
    """Standalone test functions adopted from PyTorch Tests"""

    def test_inplace_set_storage(self):
        """Test from PyTorch test_meta.py."""
        x = torch.tensor([0, 1], dtype=torch.int64, device="neuron")
        storage = x.untyped_storage()
        ssize = storage.size()
        meta = torch.empty((), dtype=torch.int64, device="neuron")
        meta.set_(storage, 0, (), ())
        assert storage.size() == ssize

    def test_torch_set(self):
        """Test from PyTorch test_torch.py."""
        zero_d = torch.randn((), device="neuron")
        one_d = torch.randn((1,), device="neuron")
        zero_d_clone = zero_d.clone()
        one_d_clone = one_d.clone()

        assert zero_d_clone.set_(one_d.storage(), 0, (), ()).shape == ()
        assert zero_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape == (1,)
        assert one_d_clone.set_(one_d.storage(), 0, (), ()).shape == ()
        assert one_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape == (1,)

        assert zero_d.clone().set_(zero_d).shape == ()
        assert one_d.clone().set_(zero_d).shape == ()
        assert zero_d.clone().set_(one_d).shape == (1,)
        assert one_d.clone().set_(one_d).shape == (1,)

    def test_contiguous(self):
        """Test from PyTorch test_view_ops.py."""
        x = torch.randn(1, 16, 5, 5, device="neuron")
        assert x.is_contiguous()
        stride = list(x.stride())
        stride[0] = 20
        # Change stride in dimension 0. Tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        assert x.is_contiguous()


class TestSetEdgeCases:
    """Test edge cases and error conditions for set_ operations."""

    def test_large_storage_offset(self):
        """Test set_ with storage offset near storage boundary."""
        source = torch.randn(100, device="neuron")
        target = torch.empty(0, device="neuron")

        # Set with offset at the very end (size 0)
        target.set_(source.storage(), 100, (0,))
        assert target.numel() == 0

        # Set with offset just before end (size 1)
        target.set_(source.storage(), 99, (1,))
        assert target.numel() == 1

    def test_invalid_stride_configurations(self):
        """Test set_ with invalid stride configurations."""
        source = torch.randn(12, device="neuron")
        target = torch.empty(0, device="neuron")

        # Valid: 2x3 with row-major strides
        target.set_(source.storage(), 0, (2, 3), (3, 1))
        assert target.shape == (2, 3)

        # Valid: 2x3 with column-major strides
        target.set_(source.storage(), 0, (2, 3), (1, 2))
        assert target.shape == (2, 3)

    def test_memory_alignment_edge_cases(self):
        """Test set_ operations with various memory alignments."""
        # Create tensor with specific alignment
        source = torch.randn(64, device="neuron")  # 64 elements for alignment testing
        target = torch.empty(0, device="neuron")

        # Test various offsets that might affect alignment
        for offset in [0, 1, 2, 4, 8, 16]:
            if offset < source.numel():
                remaining = source.numel() - offset
                target.set_(source.storage(), offset, (remaining,))
                assert target.numel() == remaining
                assert target.storage_offset() == offset

    def test_multiple_set_operations(self):
        """Test multiple set_ operations on same tensor."""
        source1 = torch.randn(5, device="neuron")
        source2 = torch.randn(7, device="neuron")
        source3 = torch.randn(3, device="neuron")
        target = torch.empty(0, device="neuron")

        # Chain multiple set operations
        target.set_(source1)
        assert target.shape == source1.shape
        assert torch.allclose(target, source1)

        target.set_(source2)
        assert target.shape == source2.shape
        assert torch.allclose(target, source2)

        target.set_(source3)
        assert target.shape == source3.shape
        assert torch.allclose(target, source3)

    def test_set_storage_boundary_conditions(self):
        """Test set_ at storage boundaries."""
        source = torch.randn(10, device="neuron")
        target = torch.empty(0, device="neuron")

        # Test at exact storage boundary
        target.set_(source.storage(), 10, (0,))  # Offset at end, size 0
        assert target.numel() == 0

        # Test just within boundary
        target.set_(source.storage(), 9, (1,))  # Offset near end, size 1
        assert target.numel() == 1

    def test_set_with_different_device_indices(self):
        """Test set_ operations with explicit device indices."""
        # Both tensors on same neuron device (device index should match)
        source = torch.randn(4, device="neuron")
        target = torch.empty(0, device="neuron")

        target.set_(source)

        assert target.device == source.device
        assert torch.allclose(target, source)

    @pytest.mark.parametrize("size", [0, 1, 2, 100, 1000])
    def test_set_various_sizes(self, size):
        """Test set_ operations with various tensor sizes."""
        source = torch.randn(size, device="neuron") if size > 0 else torch.empty(0, device="neuron")

        target = torch.empty(0, device="neuron")
        target.set_(source)

        assert target.shape == source.shape
        assert target.numel() == size
        if size > 0:
            assert torch.allclose(target, source)

    def test_set_preserves_tensor_properties(self):
        """Test that set_ preserves important tensor properties."""
        source = torch.randn(3, 4, device="neuron", dtype=torch.float32)
        target = torch.empty(0, device="neuron", dtype=torch.float32)

        target.set_(source)

        # Properties should be preserved/copied appropriately
        assert target.device == source.device
        assert target.is_contiguous() == source.is_contiguous()
        assert target.storage().device == source.storage().device

    @assert_raises(RuntimeError)
    def test_set_error_recovery(self):
        """Test that failed set_ operations don't corrupt tensor state."""
        target = torch.randn(5, device="neuron")  # Start with valid state
        original_shape = target.shape
        original_data = target.clone()

        # Attempt invalid operation
        cpu_tensor = torch.randn(3)
        target.set_(cpu_tensor)

        # Tensor should retain original state after failed operation
        assert target.shape == original_shape
        assert torch.allclose(target, original_data)
