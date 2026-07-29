"""Dynamic-K matmul: out[M,N] = x_eᵀ @ dout, contracting over K.

K is the dynamic contraction dim (via num_k_tiles); M, N are baked. Operands are
K-major (x_e=[K,M], dout=[K,N]) so K is the partition/loop dim with a dynamic count
and static free strides — one NEFF compiled at K=128 serves any 128-multiple K.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def dynamic_k_matmul_kernel(x_e, dout, num_k_tiles):
    """out[M,N] = x_eᵀ @ dout, contracting the K partition dim.

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
    num_m_tiles = M // tile_m  # static
    num_n_tiles = N // tile_n  # static

    out = nl.ndarray((M, N), x_e.dtype, nl.shared_hbm)

    # Zero-init output; we accumulate into it across K tiles.
    z = nl.ndarray((tile_m, N), x_e.dtype, nl.sbuf)
    nisa.memset(z, 0.0)
    for mi in range(num_m_tiles):
        nisa.dma_copy(dst=out[mi * tile_m : (mi + 1) * tile_m, :], src=z)

    nt_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nt_sbuf, src=num_k_tiles)
    nt_reg = nisa.register_alloc()
    nisa.register_load(nt_reg, nt_sbuf)

    k_off = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)  # dynamic K-row offset
    nisa.memset(k_off, 0.0)

    x_tile = nl.ndarray((tile_k, tile_m), x_e.dtype, nl.sbuf)
    d_tile = nl.ndarray((tile_k, tile_n), dout.dtype, nl.sbuf)

    for _ in nl.dynamic_range(0, nt_reg, 1):
        for mb in nl.affine_range(num_m_tiles):
            m_start = mb * tile_m
            # x_e[k_off:+128, m_start:+128]: partition stride M (static), base k_off.
            nisa.dma_copy(
                dst=x_tile[0:tile_k, 0:tile_m],
                src=x_e.ap(
                    pattern=[[M, tile_k], [1, tile_m]],
                    offset=m_start,
                    scalar_offset=k_off,
                    indirect_dim=0,
                ),
            )
            for nb in nl.affine_range(num_n_tiles):
                n_start = nb * tile_n
                # dout[k_off:+128, n_start:+512]
                nisa.dma_copy(
                    dst=d_tile[0:tile_k, 0:tile_n],
                    src=dout.ap(
                        pattern=[[N, tile_k], [1, tile_n]],
                        offset=n_start,
                        scalar_offset=k_off,
                        indirect_dim=0,
                    ),
                )
                psum = nl.ndarray((tile_m, tile_n), nl.float32, nl.psum)
                nisa.nc_matmul(
                    dst=psum[0:tile_m, 0:tile_n],
                    stationary=x_tile[0:tile_k, 0:tile_m],
                    moving=d_tile[0:tile_k, 0:tile_n],
                )
                # read-modify-write accumulate into HBM out tile
                rt = nl.ndarray((tile_m, tile_n), x_e.dtype, nl.sbuf)
                nisa.dma_copy(
                    dst=rt[0:tile_m, 0:tile_n],
                    src=out[m_start : m_start + tile_m, n_start : n_start + tile_n],
                )
                nisa.tensor_tensor(
                    dst=rt[0:tile_m, 0:tile_n],
                    data1=rt[0:tile_m, 0:tile_n],
                    data2=psum[0:tile_m, 0:tile_n],
                    op=nl.add,
                )
                nisa.dma_copy(
                    dst=out[m_start : m_start + tile_m, n_start : n_start + tile_n],
                    src=rt[0:tile_m, 0:tile_n],
                )
        nisa.tensor_scalar(k_off, k_off, nl.add, tile_k)

    return out
