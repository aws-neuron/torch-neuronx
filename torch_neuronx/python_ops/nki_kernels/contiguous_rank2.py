"""NKI kernel for rank-2 contiguous copy"""

import neuronxcc.nki.typing as nt
import torch
from neuronxcc import nki

from torch_neuronx.nki_hop import nki_op, wrap_nki


@wrap_nki
@nki.jit
def _traced_kernel(
    src, dst: nt.mutable_tensor, outer_size, inner_size, src_outer_stride, storage_offset
):
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl

    # src and dst are already 1D tensors from the caller
    # Loop over the outer dimension
    for i in nl.sequential_range(outer_size):
        # Calculate source position accounting for storage offset and stride
        src_start = storage_offset + i * src_outer_stride

        # Calculate destination position (destination is contiguous)
        dst_start = i * inner_size

        # Copy inner_size elements from source to destination
        nisa.dma_copy(
            dst=dst[dst_start : dst_start + inner_size],
            src=src[src_start : src_start + inner_size],
        )


@nki_op("nki_kernels::contiguous_rank2", mutates_args={})
def contiguous_rank2_kernel(
    src: torch.Tensor,
    dst: torch.Tensor,
    shape: list[int],
    src_strides: list[int],
    block_size: int,
    storage_offset: int,
) -> torch.Tensor:
    """
    Wrapper for RANK2 kernel with debug logging and parameter adaptation.

    Args:
        src: Source tensor (1D view of full storage)
        dst: Destination tensor (1D view)
        shape: Collapsed shape (2 elements)
        src_strides: Source strides (2 elements)
        block_size: Number of contiguous elements (should equal shape[1])
        storage_offset: Storage offset of the original tensor
    """
    # Validate parameters
    if len(shape) != 2:
        raise ValueError(f"Expected rank-2 collapsed shape, got {len(shape)} dimensions")

    if block_size != shape[1]:
        raise ValueError(
            f"block_size ({block_size}) should equal inner dimension size ({shape[1]})"
        )

    # Extract parameters
    outer_size = shape[0]
    inner_size = shape[1]
    src_outer_stride = src_strides[0]

    # Handle unsupported dtypes by treating as uint8
    src_dtype = src.dtype
    dst_dtype = dst.dtype
    element_size_multiplier = 1

    # List of dtypes supported by NKI dma_copy
    supported_dtypes = {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.uint32,
        torch.int16,
        torch.uint16,
        torch.int8,
        torch.uint8,
        torch.bool,
    }

    # Add float8 types if they exist
    if hasattr(torch, "float8_e4m3fn"):
        supported_dtypes.add(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        supported_dtypes.add(torch.float8_e5m2)

    # Check if we need to convert to uint8
    needs_conversion = src_dtype not in supported_dtypes or dst_dtype not in supported_dtypes

    if needs_conversion:
        # Get element size for the dtype
        element_size = src.element_size()
        if element_size > 1:
            element_size_multiplier = element_size
            src = src.view(torch.uint8)
            dst = dst.view(torch.uint8)

            # Adjust all size/stride parameters
            inner_size *= element_size_multiplier
            src_outer_stride *= element_size_multiplier
            storage_offset *= element_size_multiplier

    # Call the traced kernel
    return _traced_kernel(
        src,
        dst,
        outer_size,
        inner_size,
        src_outer_stride,
        storage_offset,
    )
