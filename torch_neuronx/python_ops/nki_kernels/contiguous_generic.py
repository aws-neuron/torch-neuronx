"""CPU fallback implementation for generic element-wise contiguous copy"""

import torch

import torch_neuronx._C as _C


def contiguous_generic_kernel(src, dst, shape, src_strides, dst_strides, src_storage_offset=0):
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
    # Calculate total size in bytes
    element_size = src.element_size()
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    total_bytes = total_elements * element_size

    # Allocate CPU buffer for the entire storage
    # We need to copy the entire storage because the tensor might use any part of it
    storage_size = src.untyped_storage().size() // element_size
    storage_bytes = storage_size * element_size

    # Create CPU tensor with same storage size
    cpu_buffer = torch.empty(storage_size, dtype=src.dtype, device="cpu")

    # Copy entire storage from Neuron to CPU using raw NRT operations
    # This avoids any PyTorch tensor operations that might trigger recursion
    nrt_handle = _C._get_nrt_tensor_capsule(src)
    _C._nrt_copy_raw_to_cpu(
        nrt_handle,
        cpu_buffer,
        0,  # Copy from start of storage
        storage_bytes,
    )

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
    dst_nrt_handle = _C._get_nrt_tensor_capsule(dst)
    _C._nrt_copy_cpu_to_raw(
        cpu_contiguous,
        dst_nrt_handle,
        0,  # Destination is contiguous, so offset is 0
        total_bytes,
    )

    return dst
