"""
Free-dim Concat Kernel: output = cat([input_a, input_b], dim=free)

Concatenates [m_static, H1] and [m_static, H2] along the free dim into
[m_static, H1+H2]. Dynamic m-dim via num_m_tiles; H1/H2 baked and tiled at trace
time. Pure data movement — each input tile is copied to its column range in the
output ([0:H1] for a, [H1:H1+H2] for b).
"""

import nki
import nki.isa as nisa
import nki.language as nl


def _h_segments(hidden, tile_size_h):
    """Split `hidden` into tile_size_h-wide segments (final one short) — resolved
    at trace time since H is a compile-time constant."""
    starts, widths = [], []
    off = 0
    while off < hidden:
        w = min(tile_size_h, hidden - off)
        starts.append(off)
        widths.append(w)
        off += w
    return starts, widths


@nki.jit()
def cat_kernel(input_a, input_b, num_m_tiles):
    """
    Free-dim concat: output[:, :H1] = input_a, output[:, H1:] = input_b

    Args:
        input_a: [m_static, H1] bf16 on HBM
        input_b: [m_static, H2] bf16 on HBM
        num_m_tiles: [1, 1] int32 — number of 128-row tiles to copy
    Returns:
        output: [m_static, H1 + H2] bf16 on HBM
    """
    m_static, H1 = input_a.shape  # noqa: N806
    _, H2 = input_b.shape  # noqa: N806
    H_out = H1 + H2  # noqa: N806

    tile_size_m = 128
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H1/H2 are compile-time constants, so each input's segments
    # resolve at trace time.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_a.dtype), 2)
    tile_size_h = 4096 // _dtype_bytes
    a_starts, a_widths = _h_segments(H1, tile_size_h)
    b_starts, b_widths = _h_segments(H2, tile_size_h)

    result = nl.ndarray((m_static, H_out), input_a.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    def body(_i):
        # input_a -> result[:, 0:H1]
        for seg in range(len(a_starts)):
            h_start, h_width = a_starts[seg], a_widths[seg]
            a_tile = nl.ndarray((tile_size_m, h_width), input_a.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=a_tile[0:tile_size_m, 0:h_width],
                src=input_a.ap(
                    pattern=[[H1, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[H_out, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=a_tile[0:tile_size_m, 0:h_width],
            )
        # input_b -> result[:, H1:H1+H2]
        for seg in range(len(b_starts)):
            h_start, h_width = b_starts[seg], b_widths[seg]
            b_tile = nl.ndarray((tile_size_m, h_width), input_b.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=b_tile[0:tile_size_m, 0:h_width],
                src=input_b.ap(
                    pattern=[[H2, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[H_out, tile_size_m], [1, h_width]],
                    offset=H1 + h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=b_tile[0:tile_size_m, 0:h_width],
            )
        nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m)

    nl.fori_loop(0, num_m_tiles_reg, body)

    return result
