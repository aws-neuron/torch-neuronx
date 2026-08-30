"""Dynamic-K matmul: out[M,N] = x_eᵀ @ dout, contracting over K.

K is the dynamic contraction dim (via num_k_tiles); M, N are baked. Operands are
K-major (x_e=[K,M], dout=[K,N]) so K is the partition/loop dim with a dynamic count
and static free strides — one NEFF compiled at K=128 serves any 128-multiple K.

M and N are baked, so neither one needs tile alignment. Full tiles keep their
``affine_range`` loops, and the kernel peels a short final tile at trace time. The tile
helpers are at module level, because a traced body cannot call a kernel-local function.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def _accumulate_tile(
    dout,
    out,
    x_tile,
    d_tile,
    n_stride,
    tile_k,
    m_start,
    m_w,
    n_start,
    n_w,
    k_off,
):
    """``out[m_start:+m_w, n_start:+n_w] += x_tile.T @ dout[k_off:+tile_k, n_start:+n_w]``.

    Each call allocates PSUM, so the live range ends with the tile (NCC_IBIR092). PSUM
    cannot stay live across ``fori_loop`` iterations. Therefore the sum goes back to HBM.
    """
    nisa.dma_copy(
        dst=d_tile[0:tile_k, 0:n_w],
        src=dout.ap(
            pattern=[[n_stride, tile_k], [1, n_w]],
            offset=n_start,
            scalar_offset=k_off,
            indirect_dim=0,
        ),
    )
    psum = nl.ndarray((m_w, n_w), nl.float32, nl.psum)
    nisa.nc_matmul(
        dst=psum[0:m_w, 0:n_w],
        stationary=x_tile[0:tile_k, 0:m_w],
        moving=d_tile[0:tile_k, 0:n_w],
    )
    rt = nl.ndarray((m_w, n_w), out.dtype, nl.sbuf)
    nisa.dma_copy(
        dst=rt[0:m_w, 0:n_w],
        src=out[m_start : m_start + m_w, n_start : n_start + n_w],
    )
    nisa.tensor_tensor(
        dst=rt[0:m_w, 0:n_w],
        data1=rt[0:m_w, 0:n_w],
        data2=psum[0:m_w, 0:n_w],
        op=nl.add,
    )
    nisa.dma_copy(
        dst=out[m_start : m_start + m_w, n_start : n_start + n_w],
        src=rt[0:m_w, 0:n_w],
    )


def _accumulate_row_band(
    x_e,
    dout,
    out,
    x_tile,
    d_tile,
    m_stride,
    n_stride,
    tile_k,
    tile_n,
    m_start,
    m_w,
    num_full_n,
    tail_n,
    n_tail_start,
    k_off,
):
    """Load the x stripe of one m-tile. Then add every n-tile of that output row band."""
    # x_e[k_off:+tile_k, m_start:+m_w]: partition stride m_stride (static), base k_off.
    nisa.dma_copy(
        dst=x_tile[0:tile_k, 0:m_w],
        src=x_e.ap(
            pattern=[[m_stride, tile_k], [1, m_w]],
            offset=m_start,
            scalar_offset=k_off,
            indirect_dim=0,
        ),
    )
    if num_full_n:
        for nb in nl.affine_range(num_full_n):
            _accumulate_tile(
                dout=dout,
                out=out,
                x_tile=x_tile,
                d_tile=d_tile,
                n_stride=n_stride,
                tile_k=tile_k,
                m_start=m_start,
                m_w=m_w,
                n_start=nb * tile_n,
                n_w=tile_n,
                k_off=k_off,
            )
    if tail_n:
        _accumulate_tile(
            dout=dout,
            out=out,
            x_tile=x_tile,
            d_tile=d_tile,
            n_stride=n_stride,
            tile_k=tile_k,
            m_start=m_start,
            m_w=m_w,
            n_start=n_tail_start,
            n_w=tail_n,
            k_off=k_off,
        )


@nki.jit()
def dynamic_k_matmul_kernel(x_e, dout, num_k_tiles):
    """out[M,N] = x_eᵀ @ dout, contracting the K partition dim.

    M and N accept any positive size. Both are baked, so this kernel peels a short tile.

    Args:
        x_e: [K_compile, M] bf16 on HBM (first num_k_tiles*128 rows used)
        dout: [K_compile, N] bf16 on HBM
        num_k_tiles: [1, 1] int32 — number of 128-row K tiles to contract
    Returns:
        out: [M, N] bf16 on HBM
    """
    K_compile, M = x_e.shape  # noqa: N806
    K_compile_, N = dout.shape  # noqa: N806
    assert K_compile == K_compile_

    tile_k, tile_m, tile_n = 128, 128, 512

    # M and N are baked, so a short final tile has a trace-time constant width.
    num_full_m, tail_m = divmod(M, tile_m)
    num_full_n, tail_n = divmod(N, tile_n)
    m_tail_start = num_full_m * tile_m
    n_tail_start = num_full_n * tile_n

    out = nl.ndarray((M, N), x_e.dtype, nl.shared_hbm)

    # Zero-init output; we accumulate into it across K tiles.
    z = nl.ndarray((tile_m, N), x_e.dtype, nl.sbuf)
    nisa.memset(z, 0.0)
    for mi in range(num_full_m):
        nisa.dma_copy(dst=out[mi * tile_m : (mi + 1) * tile_m, :], src=z)
    if tail_m:
        nisa.dma_copy(dst=out[m_tail_start:M, :], src=z[0:tail_m, :])

    nt_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sbuf, src=num_k_tiles)
    nt_reg = nisa.register_alloc()
    nisa.register_load(nt_reg, nt_sbuf)

    k_off = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)  # dynamic K-row offset
    nisa.memset(k_off, 0.0)

    x_tile = nl.ndarray((tile_k, tile_m), x_e.dtype, nl.sbuf)
    d_tile = nl.ndarray((tile_k, tile_n), dout.dtype, nl.sbuf)

    def body(_i):
        if num_full_m:
            for mb in nl.affine_range(num_full_m):
                _accumulate_row_band(
                    x_e=x_e,
                    dout=dout,
                    out=out,
                    x_tile=x_tile,
                    d_tile=d_tile,
                    m_stride=M,
                    n_stride=N,
                    tile_k=tile_k,
                    tile_n=tile_n,
                    m_start=mb * tile_m,
                    m_w=tile_m,
                    num_full_n=num_full_n,
                    tail_n=tail_n,
                    n_tail_start=n_tail_start,
                    k_off=k_off,
                )
        if tail_m:
            _accumulate_row_band(
                x_e=x_e,
                dout=dout,
                out=out,
                x_tile=x_tile,
                d_tile=d_tile,
                m_stride=M,
                n_stride=N,
                tile_k=tile_k,
                tile_n=tile_n,
                m_start=m_tail_start,
                m_w=tail_m,
                num_full_n=num_full_n,
                tail_n=tail_n,
                n_tail_start=n_tail_start,
                k_off=k_off,
            )
        nisa.tensor_scalar(k_off, k_off, nl.add, tile_k)

    return out
