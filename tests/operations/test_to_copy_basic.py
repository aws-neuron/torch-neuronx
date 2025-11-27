import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


def test_to_copy_cpu_to_neuron():
    """Test .to() for CPU to Neuron transfer"""
    cpu_tensor = torch.randn(4, 4)
    neuron_tensor = cpu_tensor.to("neuron:0")

    assert neuron_tensor.device.type == "neuron"
    assert neuron_tensor.device.index == 0
    assert neuron_tensor.shape == cpu_tensor.shape

    # Verify data
    back_to_cpu = neuron_tensor.to("cpu")
    torch.testing.assert_close(cpu_tensor, back_to_cpu)


def test_to_copy_neuron_to_cpu():
    """Test .to() for Neuron to CPU transfer"""
    neuron_tensor = torch.randn(4, 4).to("neuron:0")
    cpu_tensor = neuron_tensor.to("cpu")

    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.shape == neuron_tensor.shape


def test_to_copy_dtype_conversion():
    """Test .to() with dtype conversion"""
    cpu_float = torch.randn(4, 4, dtype=torch.float32)
    neuron_int = cpu_float.to(device="neuron:0", dtype=torch.int32)

    assert neuron_int.dtype == torch.int32
    assert neuron_int.device.type == "neuron"

    # Verify conversion
    expected = cpu_float.to(torch.int32)
    result = neuron_int.to("cpu")
    torch.testing.assert_close(expected, result)


def test_to_copy_non_contiguous_cpu_to_neuron():
    """Test .to() handles non-contiguous CPU tensors"""
    # Create a non-contiguous CPU tensor
    cpu_tensor = torch.randn(4, 4).transpose(0, 1)
    assert not cpu_tensor.is_contiguous()

    # Should handle non-contiguous by making it contiguous internally
    neuron_tensor = cpu_tensor.to("neuron:0")

    assert neuron_tensor.device.type == "neuron"
    assert neuron_tensor.shape == cpu_tensor.shape

    # Verify data is correct
    back_to_cpu = neuron_tensor.to("cpu")
    torch.testing.assert_close(cpu_tensor, back_to_cpu)


def test_to_copy_no_op_same_device():
    """Test that .to() with same device returns same tensor"""
    cpu_tensor = torch.randn(4, 4)

    # Same device, dtype, layout - should return same object
    result = cpu_tensor.to(device="cpu")
    assert result is cpu_tensor

    # Same device but different dtype - should create new tensor
    result_int = cpu_tensor.to(device="cpu", dtype=torch.int32)
    assert result_int is not cpu_tensor
    assert result_int.dtype == torch.int32


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") != "1",
    reason="Error message for Neuron execution only",
)
@assert_raises(
    RuntimeError, match="Neuron tensors only support contiguous or preserve memory format"
)
def test_to_copy_memory_format_error():
    """Test that unsupported memory formats are rejected with canonical message"""
    tensor = torch.randn(4, 4, 4, 4)

    # Should fail for non-supported formats
    tensor.to("neuron:0", memory_format=torch.channels_last)


def test_to_copy_combined_dtype_and_device():
    """Test .to() with both dtype and device change"""
    cpu_float = torch.randn(4, 4, dtype=torch.float32)

    # Change both device and dtype to bfloat16
    neuron_bfloat = cpu_float.to(device="neuron:0", dtype=torch.bfloat16)

    assert neuron_bfloat.device.type == "neuron"
    assert neuron_bfloat.dtype == torch.bfloat16

    # Verify data
    expected = cpu_float.to(dtype=torch.bfloat16)
    result = neuron_bfloat.to("cpu")
    torch.testing.assert_close(expected, result)


def test_to_copy_empty_tensor():
    """Test .to() with empty tensors"""
    empty_cpu = torch.empty(0)
    empty_neuron = empty_cpu.to("neuron:0")

    assert empty_neuron.device.type == "neuron"
    assert empty_neuron.numel() == 0

    # And back
    back_to_cpu = empty_neuron.to("cpu")
    assert back_to_cpu.device.type == "cpu"
    assert back_to_cpu.numel() == 0


def test_to_copy_scalar_tensor():
    """Test .to() with scalar tensors"""
    scalar_cpu = torch.tensor(3.14)
    scalar_neuron = scalar_cpu.to("neuron:0")

    assert scalar_neuron.device.type == "neuron"
    assert scalar_neuron.dim() == 0

    # Verify value
    result = scalar_neuron.to("cpu")
    assert result.item() == pytest.approx(3.14)
