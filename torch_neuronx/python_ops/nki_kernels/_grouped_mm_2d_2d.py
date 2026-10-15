# ruff: noqa: N806
"""Vendored copy of neuronxcc grouped_mm_2d_2d kernel.

Fixes deprecated ``neuronxcc.nki`` import of ``div_ceil`` that the NKI
compiler now rejects.  Remove this file once the upstream kernel is fixed.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit(mode="trace")
def grouped_mm_2d_2d(a, b, offs):
    D1, T = a.shape
    T_, D2 = b.shape
    (G,) = offs.shape
    assert T == T_, f"T dimension mismatch: a has {T=}, b has {T_=}"
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype=}, {b.dtype=}"
    dtype = a.dtype

    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)

    PROG_D2 = D2 // num_progs
    TILE_T = nl.tile_size.pmax
    TILE_D1 = nl.tile_size.gemm_stationary_fmax
    TILE_D2 = nl.tile_size.gemm_moving_fmax
    TILE_C = nl.tile_size.pmax

    assert T % TILE_T == 0, f"{T=} must be divisible by {TILE_T=}"
    assert D2 % num_progs == 0, f"{D2=} must be divisible by {num_progs=}"

    r = nl.ndarray((G, D1, D2), dtype, nl.shared_hbm)

    offs_tile = nl.ndarray((TILE_C, G), offs.dtype, nl.sbuf)
    offs_ap = offs.ap([[0, TILE_C], [1, G]])
    nisa.dma_copy(offs_tile, offs_ap)

    comparison_tile = nl.ndarray((TILE_C, G), offs.dtype, nl.sbuf)

    chunk_to_group_tiles = []
    for chunk_outer in range(div_ceil(T // TILE_T, TILE_C)):
        MASKED_TILE_C = min(TILE_C, T // TILE_T - chunk_outer * TILE_C)

        chunk_start_tile = nl.ndarray((MASKED_TILE_C, G), offs.dtype, nl.sbuf)
        nisa.iota(
            chunk_start_tile, [[0, G]], chunk_outer * TILE_C * TILE_T, channel_multiplier=TILE_T
        )

        offs_tile_ap = offs_tile.ap([[G, MASKED_TILE_C], [1, G]])
        comparison_tile_ap = comparison_tile.ap([[G, MASKED_TILE_C], [1, G]])
        nisa.tensor_tensor(comparison_tile_ap, offs_tile_ap, chunk_start_tile, nl.less_equal)

        chunk_to_group_tile = nl.ndarray((MASKED_TILE_C, 1), offs.dtype, nl.sbuf)
        nisa.tensor_reduce(chunk_to_group_tile, nl.add, comparison_tile_ap, 1)
        chunk_to_group_tiles.append(chunk_to_group_tile)

    zero = nl.ndarray((TILE_D1, PROG_D2), dtype, nl.sbuf)
    nisa.memset(zero, 0)
    for g in range(G):
        for i in range(div_ceil(D1, TILE_D1)):
            MASKED_TILE_D1 = min(TILE_D1, D1 - i * TILE_D1)
            r_ap = r.ap(
                [[D2, MASKED_TILE_D1], [1, PROG_D2]],
                g * D1 * D2 + i * TILE_D1 * D2 + prog_id * PROG_D2,
            )
            zero_ap = zero.ap([[PROG_D2, MASKED_TILE_D1], [1, PROG_D2]])
            nisa.dma_copy(r_ap, zero_ap)

    for k in range(T // TILE_T):
        chunk_outer, chunk_inner = divmod(k, TILE_C)
        chunk_to_group_tile = chunk_to_group_tiles[chunk_outer]
        chunk_to_group_tile_ap = chunk_to_group_tile.ap([[1, 1], [1, 1]], chunk_inner)

        b_tile = nl.ndarray((TILE_T, PROG_D2), dtype, nl.sbuf)
        b_ap = b.ap([[D2, TILE_T], [1, PROG_D2]], k * TILE_T * D2 + prog_id * PROG_D2)
        nisa.dma_copy(b_tile, b_ap)

        for i in range(div_ceil(D1, TILE_D1)):
            MASKED_TILE_D1 = min(TILE_D1, D1 - i * TILE_D1)

            a_T_tile = nl.ndarray((TILE_T, MASKED_TILE_D1), dtype, nl.sbuf)
            a_T_tile_ap = a_T_tile.ap(
                [[MASKED_TILE_D1, TILE_T], [1, 1], [1, 1], [1, MASKED_TILE_D1]]
            )
            a_ap = a.ap(
                [[T, MASKED_TILE_D1], [1, 1], [1, 1], [1, TILE_T]], i * TILE_D1 * T + k * TILE_T
            )
            nisa.dma_transpose(a_T_tile_ap, a_ap)

            r_tile_sbuf = nl.ndarray((MASKED_TILE_D1, PROG_D2), dtype, nl.sbuf)
            r_ap = r.ap(
                [[D2, MASKED_TILE_D1], [1, PROG_D2]],
                i * TILE_D1 * D2 + prog_id * PROG_D2,
                scalar_offset=chunk_to_group_tile_ap,
            )
            nisa.dma_copy(r_tile_sbuf, r_ap, oob_mode=nisa.oob_mode.skip)

            for j in range(div_ceil(PROG_D2, TILE_D2)):
                MASKED_TILE_D2 = min(TILE_D2, PROG_D2 - j * TILE_D2)
                r_tile_psum = nl.ndarray((MASKED_TILE_D1, MASKED_TILE_D2), nl.float32, nl.psum)
                b_tile_ap = b_tile.ap([[PROG_D2, TILE_T], [1, MASKED_TILE_D2]], j * TILE_D2)
                nisa.nc_matmul(r_tile_psum, a_T_tile, b_tile_ap)

                r_tile_sbuf_ap = r_tile_sbuf.ap(
                    [[PROG_D2, MASKED_TILE_D1], [1, MASKED_TILE_D2]], j * TILE_D2
                )
                nisa.tensor_tensor(r_tile_sbuf_ap, r_tile_sbuf_ap, r_tile_psum, nl.add)

            nisa.dma_copy(r_ap, r_tile_sbuf, oob_mode=nisa.oob_mode.skip)

    return r
