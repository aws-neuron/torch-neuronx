"""where Kernel: output = where(condition, input_a, input_b)  (torch.where.self)

Elementwise select: out = condition ? input_a : input_b. input_a / input_b share
shape and (float) dtype. Dynamic m-dim with single block size = 128; H is baked
and tiled at trace time (dtype-sized tile width + a final short tile).

The condition is a bool (1-byte) mask, but a 1-byte HBM tensor cannot be DMA'd
with the per-tile dynamic offset (only the first, offset-0 tile reads correctly;
every later tile's byte address is miscomputed). So the adapter hands the
condition in as an int16 view ([m, H//2], zero-copy); the DMA then uses a 2-byte
element (correct dynamic offset), and we reinterpret it back to uint8 [m, H] in
SBUF before selecting. H is therefore required to be even (enforced in the
adapter's can_handle).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def where_kernel(condition, input_a, input_b, num_m_tiles):
    """
    where: output = condition ? input_a : input_b, elementwise.

    Args:
        condition: [m_static, hidden // 2] int16 on HBM — a bool[m_static, hidden]
                   mask viewed as int16 (2 bool bytes per element)
        input_a:   [m_static, hidden] on HBM (selected where condition is true)
        input_b:   [m_static, hidden] on HBM (selected where condition is false)
        num_m_tiles: [1, 1] int32 — number of 128-row tiles per program
    Returns:
        output: [m_static, hidden], input_a.dtype, on HBM
    """
    m_static, hidden = input_a.shape
    hidden_half = condition.shape[1]

    tile_size_m = 128
    # Programs process interleaved row tiles; the adapter equalizes loop counts.
    # `condition` is a half-width int16 view, so its row stride is hidden_half.
    num_programs = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    program_row_offset = program_id * tile_size_m * hidden
    program_row_offset_cond = program_id * tile_size_m * hidden_half
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H compile-time constant -> resolved at trace time; tile_size_h-
    # wide tiles plus a final short tile of `hidden % tile_size_h` — arbitrary H.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_a.dtype), 2)
    tile_size_h = min(hidden, 4096 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    def body(_i):
        # Load the whole condition row as int16 (2-byte -> correct dynamic offset),
        # then reinterpret to uint8 [128, hidden] in SBUF — same bytes, so column j
        # of the uint8 view is condition bit j. Sliced per h-segment below.
        cond16 = nl.ndarray((tile_size_m, hidden_half), nl.int16, nl.sbuf)
        nisa.dma_copy(
            dst=cond16[0:tile_size_m, 0:hidden_half],
            src=condition.ap(
                pattern=[[hidden_half, tile_size_m], [1, hidden_half]],
                offset=program_row_offset_cond,
                scalar_offset=m_tile_start,
                indirect_dim=0,
            ),
        )
        cond8 = cond16.view(nl.uint8)

        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]

            a_tile = nl.ndarray((tile_size_m, h_width), input_a.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=a_tile[0:tile_size_m, 0:h_width],
                src=input_a.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=program_row_offset + h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            b_tile = nl.ndarray((tile_size_m, h_width), input_b.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=b_tile[0:tile_size_m, 0:h_width],
                src=input_b.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=program_row_offset + h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # Exact select, dtype pinned to input_a. `nl.*` rather than the usual
            # `nisa.*`-into-a-preallocated-dst because nisa has no data-mask select
            # (affine_select/range_select are index-affine with a scalar false value).
            out_tile = nl.where(
                cond8[0:tile_size_m, h_start : h_start + h_width],
                a_tile[0:tile_size_m, 0:h_width],
                b_tile[0:tile_size_m, 0:h_width],
                dtype=input_a.dtype,
            )

            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=program_row_offset + h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
                src=out_tile[0:tile_size_m, 0:h_width],
            )

        nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m * num_programs)

    nl.fori_loop(0, num_m_tiles_reg, body)

    return result
