"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Gather operation using indirect load - MIGRATED TO NEW FRONTEND.

This kernel gathers rows from input based on indices in index.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from neuronxcc.private_nkl.utils.kernel_helpers import div_ceil, get_program_sharding_info
from nki.isa.constants import oob_mode


def _validate_gather_inputs(input, dim, index, num_shards):
    """Validate inputs for gather operation."""
    assert len(input.shape) == 2, f"gather only supports 2D tensors, got input shape {input.shape}"
    assert len(index.shape) == 1, f"gather expects 1D index tensor, got index shape {index.shape}"
    assert dim == 0, f"gather currently only supports dim=0, got dim={dim}"

    n, d = input.shape
    k = index.shape[0]
    assert (
        k % num_shards == 0
    ), f"index size ({k}) must be divisible by num_shards ({num_shards}) for LNC sharding"


@nki.jit
def gather(input, dim, index):
    """
    Gather from input based on indices in index using indirect load.
    Equivalent to PyTorch's torch.gather(input, dim=dim, index=index).

    Args:
        input: (N, D) - source tensor to gather from
        dim: int - dimension along which to gather (must be 0)
        index: (K,) - 1D tensor of indices

    Returns:
        output: (K, D) - gathered result where output[i, :] = input[index[i], :]
        Output dtype matches input.dtype

    Example:
        input shape: (4096, 2880)
        dim: 0
        index shape: (16384,)
        output shape: (16384, 2880)

    Restrictions:
        - Input tensor must be 2D
        - Index tensor must be 1D
        - dim must be 0
    """
    k = index.shape[0]
    d = input.shape[1]
    dtype = input.dtype

    _, num_shards, shard_id = get_program_sharding_info()

    _validate_gather_inputs(input, dim, index, num_shards)

    k_tile_size = nl.tile_size.pmax
    d_tile_size = nl.tile_size.psum_fmax * 4
    num_d_tiles = div_ceil(d, d_tile_size)

    # LNC Sharding Strategy: Shard over output dimension 0 (not dimension 1)
    # Rationale: Gathering full rows ensures DMA transfers are large enough to saturate bandwidth.
    # Each core processes complete rows for its assigned row range, maximizing DMA efficiency.
    k_per_shard = k // num_shards
    k_offset = shard_id * k_per_shard
    num_k_tiles_per_shard = div_ceil(k_per_shard, k_tile_size)

    output = nl.ndarray((k, d), dtype=dtype, buffer=nl.shared_hbm)

    for local_tile_idx in nl.affine_range(num_k_tiles_per_shard):
        tile_k_offset = k_offset + local_tile_idx * k_tile_size

        # Calculate valid elements for this k tile
        k_valid = min(k_tile_size, k_per_shard - local_tile_idx * k_tile_size)

        # Load index tile - create destination buffer first, then use dma_copy
        idx_tile = nl.ndarray((k_valid, 1), dtype=index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_tile, src=index.ap(pattern=[[1, k_valid], [1, 1]], offset=tile_k_offset)
        )

        for d_tile_idx in nl.affine_range(num_d_tiles):
            # Calculate valid elements for this d tile
            d_valid = min(d_tile_size, d - d_tile_idx * d_tile_size)

            # Load data tile using Vector Dynamic Access
            # idx_tile contains the dynamic row indices
            data_tile = nl.ndarray((k_valid, d_valid), dtype=dtype, buffer=nl.sbuf)

            nisa.dma_copy(
                dst=data_tile,
                src=input.ap(
                    pattern=[[d, k_valid], [1, d_valid]],
                    offset=d_tile_idx * d_tile_size,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
            )

            # Store output using dma_copy instead of nl.store
            nisa.dma_copy(
                dst=output[nl.ds(tile_k_offset, k_valid), nl.ds(d_tile_idx * d_tile_size, d_valid)],
                src=data_tile,
            )

    return output
