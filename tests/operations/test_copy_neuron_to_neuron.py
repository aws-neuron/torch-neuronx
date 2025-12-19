import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


def test_copy_neuron_to_neuron_same_device():
    """Test copy between tensors on same Neuron device"""
    # Create on CPU first, then move to Neuron
    cpu_src = torch.randn(4, 4)
    src = torch.empty(4, 4, device="neuron:0")
    src.copy_(cpu_src)  # Use copy_ instead of .to()

    dst = torch.empty(4, 4, device="neuron:0")
    dst.copy_(src)

    # Verify by copying both back to CPU
    src_cpu = torch.empty_like(cpu_src)
    dst_cpu = torch.empty_like(cpu_src)
    src_cpu.copy_(src)
    dst_cpu.copy_(dst)
    torch.testing.assert_close(src_cpu, dst_cpu)


@assert_raises(RuntimeError, match="same Neuron core")
def test_copy_neuron_to_neuron_different_devices_fails():
    """Test that cross-device copy fails"""
    # Skip if only one device available
    if torch_neuronx.device_count() < 2:
        pytest.skip("Need at least 2 Neuron devices")

    cpu_src = torch.randn(4, 4)
    src = torch.empty(4, 4, device="neuron:0")
    src.copy_(cpu_src)
    dst = torch.empty(4, 4, device="neuron:1")

    dst.copy_(src)


def test_copy_self_is_noop():
    """Test that self-copy is a no-op"""
    cpu_tensor = torch.randn(4, 4)
    tensor = torch.empty(4, 4, device="neuron:0")
    tensor.copy_(cpu_tensor)
    original_data_ptr = tensor.data_ptr()

    tensor.copy_(tensor)

    # Should be same tensor
    assert tensor.data_ptr() == original_data_ptr


def test_copy_neuron_to_neuron_empty():
    """Test copying empty Neuron tensors"""
    src = torch.empty(0, device="neuron:0")
    dst = torch.empty(0, device="neuron:0")

    dst.copy_(src)
    assert dst.numel() == 0
    assert dst.shape == (0,)


def test_copy_neuron_to_neuron_scalar():
    """Test copying scalar Neuron tensors"""
    cpu_src = torch.tensor(3.14)
    src = torch.empty((), device="neuron:0")
    src.copy_(cpu_src)
    dst = torch.empty((), device="neuron:0")

    dst.copy_(src)

    # Verify value
    dst_cpu = torch.empty(())
    dst_cpu.copy_(dst)
    assert dst_cpu.item() == pytest.approx(3.14)


def test_copy_neuron_to_neuron_different_shapes():
    """Test copying Neuron tensors of different shapes"""
    cpu_src1 = torch.randn(2, 3, 4)
    src1 = torch.empty(2, 3, 4, device="neuron:0")
    src1.copy_(cpu_src1)
    dst1 = torch.empty(2, 3, 4, device="neuron:0")

    dst1.copy_(src1)

    # Verify
    src1_cpu = torch.empty_like(cpu_src1)
    dst1_cpu = torch.empty_like(cpu_src1)
    src1_cpu.copy_(src1)
    dst1_cpu.copy_(dst1)
    torch.testing.assert_close(src1_cpu, dst1_cpu)

    # Test with 1D tensor
    cpu_src2 = torch.randn(100)
    src2 = torch.empty(100, device="neuron:0")
    src2.copy_(cpu_src2)
    dst2 = torch.empty(100, device="neuron:0")

    dst2.copy_(src2)
    src2_cpu = torch.empty_like(cpu_src2)
    dst2_cpu = torch.empty_like(cpu_src2)
    src2_cpu.copy_(src2)
    dst2_cpu.copy_(dst2)
    torch.testing.assert_close(src2_cpu, dst2_cpu)
