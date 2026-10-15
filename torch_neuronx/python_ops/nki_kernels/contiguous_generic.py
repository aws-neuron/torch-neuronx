"""CPU fallback implementation for generic element-wise contiguous copy"""

import torch

import torch_neuronx._C as _C


def contiguous_generic_kernel(
    src, dst, shape, src_strides, dst_strides, src_storage_offset=0, non_blocking=False
):
    """Copy an arbitrary strided tensor through a CPU fallback.

    Args:
        src (torch.Tensor): One-dimensional view of the complete source storage.
        dst (torch.Tensor): Contiguous destination tensor on a Neuron device.
        shape (Sequence[int]): Logical source shape.
        src_strides (Sequence[int]): Logical source strides.
        dst_strides (Sequence[int]): Destination strides retained for interface compatibility.
        src_storage_offset (int, optional): Source storage offset. Defaults to ``0``.
        non_blocking (bool, optional): Allow asynchronous raw copies. Defaults to ``False``.

    Returns:
        torch.Tensor: The supplied ``dst`` after scheduling the contiguous copy back.

    Raises:
        RuntimeError: If native copy device, layout, size, or bounds validation fails.

    Note:
        This CPU fallback mutates ``dst``. With ``non_blocking=True``, its final
        CPU-to-Neuron device copy may still be pending when the function returns.
    """
    # Report this as a CPU fallback — the contiguous op was supposed to run on
    # device but no NKI kernel could handle this stride pattern.
    _C._log_offloaded_op("aten::contiguous")

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
