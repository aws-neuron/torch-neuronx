"""CPU fallback implementation for generic element-wise contiguous copy"""

import torch

import torch_neuronx._C as _C


def contiguous_generic_kernel(
    src, dst, shape, src_strides, dst_strides, src_storage_offset=0, non_blocking=False
):
    """
    Generic element-wise copy kernel using CPU fallback.

    This implementation copies the tensor to CPU, performs the strided access
    operation there (where it's efficient), and copies back to device.

    Args:
        src: Source tensor (1D view of full storage)
        dst: Destination tensor (1D view, contiguous)
        shape: Original tensor shape
        src_strides: Source strides
        dst_strides: Destination strides (should be contiguous)
        src_storage_offset: Storage offset for the source tensor
    """
    # Allocate CPU buffer for the entire storage
    storage_size = src.untyped_storage().size() // src.element_size()
    cpu_buffer = torch.empty(storage_size, dtype=src.dtype, device="cpu")

    # Copy entire storage from Neuron to CPU using raw NRT operations
    _C._nrt_copy_raw_to_cpu(src, cpu_buffer, non_blocking)

    # Create strided view on CPU buffer matching source layout
    # Use the provided storage offset
    # Create a strided view of the CPU buffer that matches the source tensor
    cpu_strided = cpu_buffer.as_strided(
        size=shape, stride=src_strides, storage_offset=src_storage_offset
    )

    # Make contiguous copy on CPU (this is very efficient on CPU)
    cpu_contiguous = cpu_strided.contiguous()

    # Copy contiguous result back to Neuron device
    # The destination tensor is already allocated and contiguous
    _C._nrt_copy_cpu_to_raw(cpu_contiguous, dst, non_blocking)

    return dst
