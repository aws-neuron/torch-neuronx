"""Exp Activation Kernel: output = exp(input)

Dynamic m-dim with single block size = 128.
Uses nisa.activation with op=nl.exp.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def exp_kernel(input_tensor, num_m_tiles):
    """
    Exp activation: output = exp(input)

    Args:
        input_tensor: [m_static, hidden] bf16 on HBM
        num_m_tiles: [1, 1] int32 — number of 128-row tiles to process
    Returns:
        output: [m_static, hidden] bf16 on HBM
    """
    m_static, hidden = input_tensor.shape

    tile_size_m = 128
    tile_size_h = 512
    # H is a compile-time constant (baked into the NEFF), so the free-dim tiling
    # is resolved in Python at trace time. Cover the full width with 512-wide
    # tiles plus a final short tile of width `hidden % 512` — arbitrary H, no
    # padding. (Contrast M: runtime-dynamic via num_m_tiles + dynamic_range.)
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((m_static, hidden), input_tensor.dtype, nl.shared_hbm)

    # Load num_m_tiles
    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    for _ in nl.dynamic_range(0, num_m_tiles_reg, 1):
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]
            # Load input tile
            in_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=in_tile[0:tile_size_m, 0:h_width],
                src=input_tensor.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # Apply exp activation
            out_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
            nisa.activation(
                dst=out_tile[0:tile_size_m, 0:h_width],
                data=in_tile[0:tile_size_m, 0:h_width],
                op=nl.exp,
            )

            # Store result
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=out_tile[0:tile_size_m, 0:h_width],
            )

        nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m)

    return result
