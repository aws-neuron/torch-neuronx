"""Dynamic-M matmul lhs[M,K] @ rhs[K,N] -> [M,N]. M runtime-dynamic (num_m_tiles),
K any positive size, N a multiple of num_programs (LNC shard). Weight-resident + LNC-sharded.

- weight-resident: n-block loop outer; each n-block's rhs loaded once, reused across M-tiles.
- LNC shard: grid=(num_programs,); core prog_id owns N columns [prog_id*shard_n, ...),
  loads only its rhs columns, writes disjoint output.
- ragged tiling: K in 128-tiles (final k-tile may be < 128); shard_n in 512-segments
  (short final n-tile, widths need not be 128-mult), grouped 2/n-block.
- adaptive n-block width: block_size_n (1024/512/256/128) shrinks with K so the resident
  rhs fits SBUF; a function of K alone, so compile_key=(K,N) still keys it.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .adapter_utils import mm_block_size_n


def div_ceil(n, d):
    return (n + d - 1) // d


def static_matmul_block(
    lhsT_stripes_in_block,  # noqa: N803
    rhs_stripes_in_block,
    result_accum,
    psum_tiles,
    tile_size_m,
    k_widths_in_block,
    block_local_tiles,
    tiles_this_k,
):
    """One m-tile across the k-tiles + n-tiles of one n-block. block_local_tiles =
    [(local_n_offset, n_width)] per n-tile; k_widths_in_block[tk] = contraction width
    of k-tile tk (128, or < 128 for the final ragged k-tile); tiles_this_k = valid
    k-tiles in this (maybe partial) k-block; psum_tiles allocated by the caller in-body."""
    for tn in range(len(block_local_tiles)):
        n_local = block_local_tiles[tn][0]
        n_width = block_local_tiles[tn][1]

        psum_tile = psum_tiles[tn]
        nisa.memset(psum_tile, 0.0)

        for tk_local in nl.affine_range(tiles_this_k):
            k_this = k_widths_in_block[tk_local]
            lhsT_stripe = lhsT_stripes_in_block[tk_local]  # noqa: N806
            rhs_stripe = rhs_stripes_in_block[tk_local]

            nisa.nc_matmul(
                dst=psum_tile[0:tile_size_m, 0:n_width],
                stationary=lhsT_stripe[0:k_this, 0:tile_size_m],
                moving=rhs_stripe[0:k_this, n_local : n_local + n_width],
            )

        nisa.tensor_tensor(
            dst=result_accum[0:tile_size_m, n_local : n_local + n_width],
            data1=result_accum[0:tile_size_m, n_local : n_local + n_width],
            data2=psum_tile[0:tile_size_m, 0:n_width],
            op=nl.add,
        )


@nki.jit()
def dynamic_m_matmul_kernel(lhs, rhs, num_m_tiles):
    """lhs[m_static,K] @ rhs[K,N] -> res[m_static,N]. num_m_tiles [1,1] int32 = number of
    128-row m-tiles. K,N any positive size (ragged final k-tile / n-tile); N divisible
    by num_programs (LNC shard)."""
    m_static, k_max = lhs.shape
    k_max_, n_max = rhs.shape
    assert k_max == k_max_

    # SPMD LNC shard over N columns (num_progs trace-time const; prog_id only in offsets).
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    assert n_max % num_progs == 0, "N must be divisible by num_programs for the LNC shard"
    shard_n = n_max // num_progs
    n_shard_offset = prog_id * shard_n

    tile_size_m = 128
    tile_size_k = 128
    tiles_in_block_k = 4  # transpose(<=4) + accumulation(<=tiles_in_block_n) PSUM banks <= 8

    num_tile_k = div_ceil(k_max, tile_size_k)  # any count; final k-tile may be < 128
    num_block_k = div_ceil(num_tile_k, tiles_in_block_k)  # final k-block may be partial

    # Per-k-tile contraction width: 128, except a ragged final tile (k_max % 128).
    k_tile_widths = []
    for t in range(num_tile_k):
        k_tile_widths.append(min(tile_size_k, k_max - t * tile_size_k))

    # fn of K -> compile_key=(K,N) keys it (recompile-free).
    block_size_n = mm_block_size_n(num_tile_k)
    tile_size_n = min(512, block_size_n)  # PSUM free dim <= 512
    tiles_in_block_n = block_size_n // tile_size_n

    # n-tiles over the per-core shard: tile_size_n-wide segments with a short final tile.
    n_tile_list = []  # (local_n_start, n_width)
    n_off = 0
    while n_off < shard_n:
        n_w = min(tile_size_n, shard_n - n_off)
        n_tile_list.append((n_off, n_w))
        n_off += n_w

    # n-blocks: group tiles_in_block_n consecutive n-tiles; final block may be partial.
    n_block_list = []  # (block_start_local, block_width, [(local_offset, width), ...])
    i = 0
    while i < len(n_tile_list):
        group = n_tile_list[i : i + tiles_in_block_n]
        block_start = group[0][0]
        block_width = 0
        block_local_tiles = []
        for gi in range(len(group)):
            w = group[gi][1]
            block_local_tiles.append((block_width, w))
            block_width += w
        n_block_list.append((block_start, block_width, block_local_tiles))
        i += tiles_in_block_n
    num_block_n = len(n_block_list)
    max_block_width = 0  # size rhs_resident to actual, not block_size_n
    for nb in n_block_list:
        if nb[1] > max_block_width:
            max_block_width = nb[1]

    res = nl.ndarray((m_static, n_max), lhs.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])

    # Top-level SBUF scratch (only PSUM must be in-body); distinct buffer per k-tile.
    lhsT_stripes = []  # noqa: N806
    lhs_tiles = []
    for _ in range(tiles_in_block_k):
        s = nl.ndarray((tile_size_k, tile_size_m), nl.bfloat16, nl.sbuf)
        nisa.memset(s, 0.0)
        lhsT_stripes.append(s)
        t = nl.ndarray((tile_size_m, tile_size_k), nl.bfloat16, nl.sbuf)
        nisa.memset(t, 0.0)
        lhs_tiles.append(t)

    # Resident rhs stripes for one n-block across all K, sized to actual max width.
    rhs_resident = []
    for _ in range(num_tile_k):
        rhs_s = nl.ndarray((tile_size_k, max_block_width), nl.bfloat16, nl.sbuf)
        nisa.memset(rhs_s, 0.0)
        rhs_resident.append(rhs_s)

    # Per-n-block m offset + loop count, top-level so each fori_loop body dominates them.
    m_tile_starts = []
    m_tile_regs = []
    for _ in range(num_block_n):
        m_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
        nisa.memset(m_start, 0.0)
        m_tile_starts.append(m_start)
        reg = nisa.register_alloc()
        nisa.register_load(reg, num_m_tiles_sbuf)
        m_tile_regs.append(reg)

    # n-block OUTER (unrolled): load its rhs stripes once, then the M-loop reuses them.
    # body is defined inline + passed straight to fori_loop (the tracer forbids calling
    # inner functions), so it closes over the per-n-block loop vars — safe because
    # fori_loop traces body immediately, so B023 is per-file-ignored for this kernel.
    for n_block in range(num_block_n):
        block_start = n_block_list[n_block][0]
        block_width = n_block_list[n_block][1]
        block_local_tiles = n_block_list[n_block][2]
        col_offset = n_shard_offset + block_start
        for tk_global in range(num_tile_k):
            k_this = k_tile_widths[tk_global]
            nisa.dma_copy(
                dst=rhs_resident[tk_global][0:k_this, 0:block_width],
                src=rhs.ap(
                    pattern=[[n_max, k_this], [1, block_width]],
                    offset=tk_global * tile_size_k * n_max + col_offset,
                ),
            )
        m_tile_start = m_tile_starts[n_block]

        def body(_i):
            # PSUM + result_accum in-body, fresh per iteration (NCC_IBIR092 / live-range).
            temp_transposes = []
            for _ in range(tiles_in_block_k):
                temp_transposes.append(nl.ndarray((tile_size_k, tile_size_m), nl.bfloat16, nl.psum))
            psum_tiles = []
            for _ in range(len(block_local_tiles)):
                psum_tiles.append(nl.ndarray((tile_size_m, tile_size_n), nl.float32, nl.psum))
            result_accum = nl.ndarray((tile_size_m, block_width), nl.float32, nl.sbuf)
            nisa.memset(result_accum, 0.0)

            for k_block in range(num_block_k):
                tiles_this_k = min(tiles_in_block_k, num_tile_k - k_block * tiles_in_block_k)
                for tk_local in range(tiles_this_k):
                    tk_global = k_block * tiles_in_block_k + tk_local
                    k_this = k_tile_widths[tk_global]
                    nisa.dma_copy(
                        dst=lhs_tiles[tk_local][0:tile_size_m, 0:k_this],
                        src=lhs.ap(
                            pattern=[[k_max, tile_size_m], [1, k_this]],
                            offset=tk_global * tile_size_k,
                            scalar_offset=m_tile_start,
                            indirect_dim=0,
                        ),
                    )
                    nisa.nc_transpose(
                        dst=temp_transposes[tk_local][0:k_this, 0:tile_size_m],
                        data=lhs_tiles[tk_local][0:tile_size_m, 0:k_this],
                    )
                    nisa.tensor_copy(
                        dst=lhsT_stripes[tk_local][0:k_this, 0:tile_size_m],
                        src=temp_transposes[tk_local][0:k_this, 0:tile_size_m],
                    )

                static_matmul_block(  # reads resident rhs for this k-block (no reload)
                    lhsT_stripes,
                    rhs_resident[
                        k_block * tiles_in_block_k : k_block * tiles_in_block_k + tiles_this_k
                    ],
                    result_accum,
                    psum_tiles,
                    tile_size_m,
                    k_tile_widths[
                        k_block * tiles_in_block_k : k_block * tiles_in_block_k + tiles_this_k
                    ],
                    block_local_tiles,
                    tiles_this_k,
                )

            nisa.dma_copy(
                dst=res.ap(
                    pattern=[[n_max, tile_size_m], [1, block_width]],
                    offset=col_offset,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=result_accum[0:tile_size_m, 0:block_width],
            )
            nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m)

        nl.fori_loop(0, m_tile_regs[n_block], body)

    return res
