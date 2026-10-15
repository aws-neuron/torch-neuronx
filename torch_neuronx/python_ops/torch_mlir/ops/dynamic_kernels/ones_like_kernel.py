"""ones_like Kernel: output = 1 (same shape/dtype as input).

Dynamic m-dim with single block size = 128. The output value is independent of
the input data — each tile is filled with 1 via nisa.memset — but input_tensor
still defines the [m_static, hidden] shape/dtype baked into the NEFF, and the
runtime num_m_tiles drives the dynamic-M loop. H is a compile-time constant,
tiled at trace time (a dtype-sized tile + a final short tile), so arbitrary H
needs no padding. (Contrast M: runtime-dynamic via num_m_tiles + fori_loop.)

The adapter relies on the operand being unread: it passes the output buffer in
arg0's place rather than padding/copying the caller's tensor (elementwise.py's
``reads_input``).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def ones_like_kernel(input_tensor, num_m_tiles):
    """Create a tensor of ones over a runtime-dynamic row dimension.

    Args:
        input_tensor (tensor): ``[m_static, hidden]`` shape and dtype prototype in HBM.
        num_m_tiles (tensor): Int32 ``[1, 1]`` count of 128-row tiles per program.

    Returns:
        tensor: HBM tensor with the prototype shape and dtype whose first
            ``128 * num_programs * num_m_tiles[0]`` rows contain ones.

    Note:
        ``num_programs`` is the launch-grid width. Later rows are left unwritten, and the
        input values are not read, so the adapter may pass the output buffer as arg0.
    """
    m_static, hidden = input_tensor.shape

    tile_size_m = 128
    # Programs process interleaved row tiles; the adapter equalizes loop counts.
    # Only the store is offset — the operand is never read.
    num_programs = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    program_row_offset = program_id * tile_size_m * hidden
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H compile-time constant -> resolved at trace time; tile_size_h-
    # wide tiles plus a final short tile of `hidden % tile_size_h` — arbitrary H.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_tensor.dtype), 2)
    tile_size_h = min(hidden, 4096 // _dtype_bytes)
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

    def body(_i):
        """Fill one dynamic row-tile iteration with ones.

        Args:
            _i (int): Unused loop iteration index.
        """
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]

            # Fill the output tile with 1 (dtype inferred from the tile).
            out_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
            nisa.memset(out_tile[0:tile_size_m, 0:h_width], 1.0)

            # Store result
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
