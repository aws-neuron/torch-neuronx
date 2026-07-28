"""
Generic Dynamic Matmul Kernel: [dynamic, K] @ [K, N] -> [dynamic, N]

Simplified dynamic matmul with single block size = 128 (one m-tile per dynamic iteration).
Takes lhs [m_static, k] (not transposed), does internal transpose.
num_m_tiles is a scalar [1,1] tensor indicating how many 128-row tiles to process.
Works for any K and N that are multiples of 128 (k) and 512 (n).
"""

import nki
import nki.isa as nisa
import nki.language as nl


def static_matmul_block(
    lhsT_stripes_in_block,  # noqa: N803
    rhs_stripes_in_block,
    result_accum,
    tile_size_m,
    tile_size_k,
    tile_size_n,
    tiles_in_block_k,
    tiles_in_block_n,
):
    """
    Static matmul for one m-tile (128 rows) across all k-blocks and n-tiles within one n-block.
    tiles_in_block_m is always 1 here (single block size = 128).
    """
    # Single m-tile (tiles_in_block_m = 1)
    for tn in nl.affine_range(tiles_in_block_n):
        n_start_local = tn * tile_size_n

        psum_tile = nl.ndarray((tile_size_m, tile_size_n), nl.float32, nl.psum)
        nisa.memset(psum_tile, 0.0)

        for tk_local in nl.affine_range(tiles_in_block_k):
            lhsT_stripe = lhsT_stripes_in_block[tk_local]  # noqa: N806
            rhs_stripe = rhs_stripes_in_block[tk_local]

            if lhsT_stripe is not None and rhs_stripe is not None:
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_size_m, 0:tile_size_n],
                    stationary=lhsT_stripe[0:tile_size_k, 0:tile_size_m],
                    moving=rhs_stripe[0:tile_size_k, n_start_local : n_start_local + tile_size_n],
                )

        nisa.tensor_tensor(
            dst=result_accum[0:tile_size_m, n_start_local : n_start_local + tile_size_n],
            data1=result_accum[0:tile_size_m, n_start_local : n_start_local + tile_size_n],
            data2=psum_tile[0:tile_size_m, 0:tile_size_n],
            op=nl.add,
        )


@nki.jit()
def dynamic_m_matmul_kernel(lhs, rhs, num_m_tiles):
    """
    Dynamic matmul: lhs[m_static, K] @ rhs[K, N] -> res[m_static, N]

    Args:
        lhs: [m_static, k=2048] on HBM
        rhs: [k=2048, n=4096] on HBM
        num_m_tiles: [1, 1] int32 tensor — number of 128-row tiles to process
    """
    m_static, k_max = lhs.shape
    k_max_, n_max = rhs.shape
    assert k_max == k_max_

    # Hyperparams
    tile_size_m = 128
    tile_size_k = 128
    tile_size_n = 512
    tiles_in_block_k = 8  # k_block = 1024
    tiles_in_block_n = 2  # n_block = 1024
    block_size_n = tiles_in_block_n * tile_size_n  # 1024

    num_tile_k = k_max // tile_size_k  # 16
    num_block_k = num_tile_k // tiles_in_block_k  # 2
    num_tile_n = n_max // tile_size_n  # 8
    num_block_n = num_tile_n // tiles_in_block_n  # 4

    # Output buffer
    res = nl.ndarray((m_static, n_max), lhs.dtype, nl.shared_hbm)

    # Load num_m_tiles into register
    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    # Track dynamic offset (in units of elements, i.e. m_tile_idx * tile_size_m * stride)
    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    # Pre-allocate SBUF tensors outside dynamic loop
    result_accum = nl.ndarray((tile_size_m, block_size_n), nl.float32, nl.sbuf)
    nisa.memset(result_accum, 0.0)

    lhsT_stripes = []  # noqa: N806
    rhs_stripes = []
    for _ in range(tiles_in_block_k):
        lhsT_s = nl.ndarray((tile_size_k, tile_size_m), nl.bfloat16, nl.sbuf)  # noqa: N806
        nisa.memset(lhsT_s, 0.0)
        lhsT_stripes.append(lhsT_s)
        rhs_s = nl.ndarray((tile_size_k, block_size_n), nl.bfloat16, nl.sbuf)
        nisa.memset(rhs_s, 0.0)
        rhs_stripes.append(rhs_s)

    # Dynamic loop: each iteration processes one 128-row m-tile
    for _ in nl.dynamic_range(0, num_m_tiles_reg, 1):
        # Static loop over n-blocks
        for n_block in nl.affine_range(num_block_n):
            nisa.memset(result_accum, 0.0)

            # Static loop over k-blocks
            for k_block in nl.sequential_range(num_block_k):
                for tk_local in range(tiles_in_block_k):
                    tk_global = k_block * tiles_in_block_k + tk_local

                    # Load lhs tile [128, 128] and transpose to [128, 128]
                    lhs_tile = nl.ndarray((tile_size_m, tile_size_k), nl.bfloat16, nl.sbuf)
                    nisa.dma_copy(
                        dst=lhs_tile[0:tile_size_m, 0:tile_size_k],
                        src=lhs.ap(
                            pattern=[[k_max, tile_size_m], [1, tile_size_k]],
                            offset=tk_global * tile_size_k,
                            scalar_offset=m_tile_start,
                            indirect_dim=0,
                        ),
                    )
                    temp_transpose = nl.ndarray((tile_size_k, tile_size_m), nl.bfloat16, nl.psum)
                    nisa.nc_transpose(dst=temp_transpose, data=lhs_tile)
                    nisa.tensor_copy(
                        dst=lhsT_stripes[tk_local][0:tile_size_k, 0:tile_size_m],
                        src=temp_transpose,
                    )

                    # Load rhs stripe [128, 1024]
                    nisa.dma_copy(
                        dst=rhs_stripes[tk_local][0:tile_size_k, 0:block_size_n],
                        src=rhs[
                            tk_global * tile_size_k : (tk_global + 1) * tile_size_k,
                            n_block * block_size_n : (n_block + 1) * block_size_n,
                        ],
                    )

                # Compute matmul for this k-block
                static_matmul_block(
                    lhsT_stripes,
                    rhs_stripes,
                    result_accum,
                    tile_size_m,
                    tile_size_k,
                    tile_size_n,
                    tiles_in_block_k,
                    tiles_in_block_n,
                )

            # Write result to HBM
            nisa.dma_copy(
                dst=res.ap(
                    pattern=[[n_max, tile_size_m], [1, block_size_n]],
                    offset=n_block * block_size_n,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=result_accum[0:tile_size_m, 0:block_size_n],
            )

        # Advance m offset by 128 rows
        # (in row-major: 128 * k_max elements for lhs, 128 * n_max for res)
        nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m)

    return res
