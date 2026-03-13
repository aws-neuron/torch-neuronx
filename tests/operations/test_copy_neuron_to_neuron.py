import os

import pytest
import torch

import torch_neuronx


def test_copy_neuron_to_neuron_same_device():
    """Test copy between tensors on same Neuron device"""
    cpu_src = torch.randn(4, 4)
    src = torch.empty(4, 4, device="neuron")
    src.copy_(cpu_src)

    dst = torch.empty(4, 4, device="neuron")
    dst.copy_(src, non_blocking=False)

    src_cpu = torch.empty_like(cpu_src)
    dst_cpu = torch.empty_like(cpu_src)
    src_cpu.copy_(src)
    dst_cpu.copy_(dst)
    torch.testing.assert_close(src_cpu, dst_cpu)


def test_copy_self_is_noop():
    """Test that self-copy is a no-op"""
    cpu_tensor = torch.randn(4, 4)
    tensor = torch.empty(4, 4, device="neuron")
    tensor.copy_(cpu_tensor)
    original_data_ptr = tensor.data_ptr()

    tensor.copy_(tensor)

    # Should be same tensor
    assert tensor.data_ptr() == original_data_ptr


def test_copy_neuron_to_neuron_empty():
    """Test copying empty Neuron tensors"""
    src = torch.empty(0, device="neuron")
    dst = torch.empty(0, device="neuron")

    dst.copy_(src)

    assert dst.numel() == 0
    assert dst.shape == (0,)


def test_copy_neuron_to_neuron_scalar():
    """Test copying scalar Neuron tensors"""
    cpu_src = torch.tensor(3.14)
    src = torch.empty((), device="neuron")
    src.copy_(cpu_src)
    dst = torch.empty((), device="neuron")

    dst.copy_(src)

    # Verify value
    dst_cpu = torch.empty(())
    dst_cpu.copy_(dst)
    assert dst_cpu.item() == pytest.approx(3.14)


def test_copy_neuron_to_neuron_different_shapes():
    """Test copying Neuron tensors of different shapes"""
    cpu_src1 = torch.randn(2, 3, 4)
    src1 = torch.empty(2, 3, 4, device="neuron")
    src1.copy_(cpu_src1)
    dst1 = torch.empty(2, 3, 4, device="neuron")

    dst1.copy_(src1)

    # Verify
    src1_cpu = torch.empty_like(cpu_src1)
    dst1_cpu = torch.empty_like(cpu_src1)
    src1_cpu.copy_(src1)
    dst1_cpu.copy_(dst1)
    torch.testing.assert_close(src1_cpu, dst1_cpu)

    # Test with 1D tensor
    cpu_src2 = torch.randn(100)
    src2 = torch.empty(100, device="neuron")
    src2.copy_(cpu_src2)
    dst2 = torch.empty(100, device="neuron")

    dst2.copy_(src2)
    src2_cpu = torch.empty_like(cpu_src2)
    dst2_cpu = torch.empty_like(cpu_src2)
    src2_cpu.copy_(src2)
    dst2_cpu.copy_(dst2)
    torch.testing.assert_close(src2_cpu, dst2_cpu)


def test_copy_neuron_to_neuron_different_src_dst_shape_compatible_not_identical():
    """Test copy with broadcasting support."""
    src = torch.randn(2, 1, 2, device="neuron")
    dst = torch.randn(2, 6, 2, device="neuron")

    dst.copy_(src)

    torch.testing.assert_close(src.expand_as(dst), dst)


def test_copy_neuron_to_neuron_dtype_conversion():
    """Test copy with automatic dtype conversion"""
    # Test fp32 -> fp16
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")
    dst = torch.randn(4, 4, dtype=torch.float16, device="neuron")
    dst.copy_(src)

    torch.testing.assert_close(dst.cpu(), src.cpu().to(torch.float16))

    # Test fp16 -> fp32
    src2 = torch.randn(4, 4, dtype=torch.float16, device="neuron")
    dst2 = torch.randn(4, 4, dtype=torch.float32, device="neuron")

    dst2.copy_(src2)

    torch.testing.assert_close(dst2.cpu(), src2.cpu().to(torch.float32))


def test_copy_neuron_to_neuron_int64():
    """Test neuron int64 copy - MLIR impl should reject int64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    src = torch.randint(0, 100, (4, 4), dtype=torch.int64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.int64, device="neuron")

    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.int64
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_to_fp32():
    """Test neuron fp64 to neuron fp32 copy - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    # Create fp64 on neuron, fp32 on neuron
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float32, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    # Operation still works via non-MLIR implementation
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.cpu().to(torch.float32), dst.cpu())


def test_copy_neuron_to_neuron_fp32_to_fp64():
    """Test neuron fp32 to neuron fp64 copy - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    # Create fp32 on neuron, fp64 on neuron
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float64, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    # Operation still works via non-MLIR implementation
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu().to(torch.float64), dst.cpu())


def test_copy_blocking_uses_mlir():
    """Test non_blocking=False uses MLIR implementation."""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    src = torch.randn(4, 4, device="neuron")
    dst = torch.empty(4, 4, device="neuron")

    impl = CopyMLIRImpl()
    assert impl.can_handle(dst, src)

    dst.copy_(src, non_blocking=False)
    torch_neuronx.synchronize()
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_to_fp64():
    """Test neuron fp64 to neuron fp64 (same dtype)"""
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float64, device="neuron")
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_empty():
    """Test copying empty fp64 tensors"""
    src = torch.empty(0, dtype=torch.float64, device="neuron")
    dst = torch.empty(0, dtype=torch.float64, device="neuron")
    dst.copy_(src)
    assert dst.numel() == 0
    assert dst.dtype == torch.float64
