"""
Gather Kernel: output[i, :] = source[indices[i], :] for i in [0, num_tokens)

Uses vector_offset for indirect DMA — loads a tile of indices and uses them
as per-row offsets in a single DMA operation.
Dynamic m-dim with single block size = 128.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def gather_kernel(source, indices, num_m_tiles):
    """
    Gather rows by index: output[i, :] = source[indices[i], :]

    Args:
        source: [T, H] bf16 on HBM — full token buffer
        indices: [T, 1] int32 on HBM — token indices (first num_tokens valid)
        num_m_tiles: [1, 1] int32 — number of 128-row output tiles
    Returns:
        output: [T, H] bf16 on HBM (first num_tokens rows valid)
    """
    T, H = source.shape  # noqa: N806

    tile_size_m = 128
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H compile-time constant -> resolved at trace time; tile_size_h-
    # wide tiles plus a final short tile of `H % tile_size_h` — arbitrary H.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(source.dtype), 2)
    tile_size_h = min(H, 4096 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < H:
        _w = min(tile_size_h, H - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((T, H), source.dtype, nl.shared_hbm)

    # Load num_m_tiles
    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    # Output row offset tracker
    out_row_offset = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(out_row_offset, 0.0)

    for _ in nl.dynamic_range(0, num_m_tiles_reg, 1):
        # Load 128 indices for this tile
        idx_tile = nl.ndarray((tile_size_m, 1), nl.int32, nl.sbuf)
        nisa.dma_copy(
            dst=idx_tile[0:tile_size_m, 0:1],
            src=indices.ap(
                pattern=[[1, tile_size_m], [1, 1]],
                offset=0,
                scalar_offset=out_row_offset,
                indirect_dim=0,
            ),
        )

        # Gather rows using vector_offset — one DMA per h-tile gathers all 128 rows
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]
            data_tile = nl.ndarray((tile_size_m, h_width), source.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=data_tile[0:tile_size_m, 0:h_width],
                src=source.ap(
                    pattern=[[H, tile_size_m], [1, h_width]],
                    offset=h_start,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
            )

            # Store to contiguous output
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[H, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=out_row_offset,
                    indirect_dim=0,
                ),
                src=data_tile[0:tile_size_m, 0:h_width],
            )

        nisa.tensor_scalar(out_row_offset, out_row_offset, nl.add, tile_size_m)

    return result
