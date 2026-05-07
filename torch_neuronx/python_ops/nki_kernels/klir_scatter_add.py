"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Scatter-add operation using gather-accumulate-scatter pattern.

This kernel scatters values from src and adds them to input based on indices in index.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from neuronxcc.private_nkl.utils.kernel_helpers import div_ceil, get_program_sharding_info


def _validate_scatter_add_inputs(input, dim, index, src, num_shards):
    """Validate inputs for scatter_add operation."""
    assert (
        len(input.shape) == 2
    ), f"scatter_add only supports 2D tensors, got input shape {input.shape}"
    assert (
        len(index.shape) == 1
    ), f"scatter_add expects 1D index tensor, got index shape {index.shape}"
    assert len(src.shape) == 2, f"scatter_add only supports 2D tensors, got src shape {src.shape}"
    assert dim == 0, f"scatter_add currently only supports dim=0, got dim={dim}"

    n, d = input.shape
    k, d_src = src.shape

    assert (
        index.shape[0] == k
    ), f"Index and src must have same dimension 0, got index {index.shape[0]}, src {k}"
    assert d == d_src, f"Dimension mismatch: input has {d}, src has {d_src}"
    assert (
        d % num_shards == 0
    ), f"dimension 1 ({d}) must be divisible by num_shards ({num_shards}) for LNC sharding"


@nki.jit
def scatter_add(input, dim, index, src):
    """
    Scatter-add from src into input based on indices.
    Equivalent to PyTorch's input.scatter_add(dim=dim, index=index, src=src).

    This operation performs: input[index[i], j] += src[i, j] for all i, j

    Note: This is an in-place operation that modifies the input tensor directly.
    Uses gather-add-scatter pattern to handle accumulation.

    Args:
        input: (N, D) - destination tensor to accumulate into (modified in-place)
        dim: int - dimension along which to scatter (must be 0)
        index: (K,) - 1D tensor of indices
        src: (K, D) - source values to scatter

    Returns:
        input: (N, D) - the input tensor with scattered values added (same object)

    Restrictions:
        - Input and src tensors must be 2D
        - Index tensor must be 1D
        - dim must be 0
        - Indices within a tile of 128 rows should be unique for correctness
    """
    k, d = src.shape

    _, num_shards, shard_id = get_program_sharding_info()

    _validate_scatter_add_inputs(input, dim, index, src, num_shards)

    k_tile_size = nl.tile_size.pmax
    d_tile_size = nl.tile_size.psum_fmax * 4
    num_k_tiles = div_ceil(k, k_tile_size)

    # LNC Sharding: shard over dimension 1 to avoid write conflicts
    d_per_shard = d // num_shards
    d_offset = shard_id * d_per_shard
    num_d_tiles_per_shard = div_ceil(d_per_shard, d_tile_size)

    for k_tile_idx in nl.affine_range(num_k_tiles):
        k_valid = min(k_tile_size, k - k_tile_idx * k_tile_size)

        # Load index tile
        idx_tile = nl.ndarray((k_valid, 1), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_tile,
            src=index.ap(pattern=[[1, k_valid], [1, 1]], offset=k_tile_idx * k_tile_size),
        )

        for local_d_tile_idx in nl.affine_range(num_d_tiles_per_shard):
            d_tile_offset = d_offset + local_d_tile_idx * d_tile_size
            d_valid = min(d_tile_size, d_per_shard - local_d_tile_idx * d_tile_size)

            # Load source tile
            src_tile = nl.ndarray((k_valid, d_valid), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_tile,
                src=src.ap(
                    pattern=[[d, k_valid], [1, d_valid]],
                    offset=d_tile_offset + k_tile_idx * k_tile_size * d,
                ),
            )

            # Gather existing values from input at indirect indices
            existing_tile = nl.ndarray((k_valid, d_valid), dtype=input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=existing_tile,
                src=input.ap(
                    pattern=[[d, k_valid], [1, d_valid]],
                    offset=d_tile_offset,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
            )

            # Add in SBUF
            result_tile = nl.ndarray((k_valid, d_valid), dtype=input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result_tile, data1=existing_tile, data2=src_tile, op=nl.add)

            # Scatter write back
            nisa.dma_copy(
                dst=input.ap(
                    pattern=[[d, k_valid], [1, d_valid]],
                    offset=d_tile_offset,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=result_tile,
            )

    return input
