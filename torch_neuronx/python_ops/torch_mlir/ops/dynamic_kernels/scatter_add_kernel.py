"""
Scatter-Add Kernel: output[indices[i], :] += data[i, :] for i in [0, num_tokens)

Uses vector_offset for indirect DMA reads/writes.
Manual read-modify-write (load target, add in SBUF, write back) to avoid
dst_rmw_op inside dynamic loop (compiler limitation).
Dynamic m-dim with single block size = 128.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def scatter_add_kernel(data, indices, output_buffer, num_m_tiles):
    """
    Scatter-add: output_buffer[indices[i], :] += data[i, :]

    Args:
        data: [T, H] bf16 on HBM — source data (first num_tokens rows valid)
        indices: [T, 1] int32 on HBM — target indices (first num_tokens valid)
        output_buffer: [T, H] bf16 on HBM — accumulation target
        num_m_tiles: [1, 1] int32
    Returns:
        output_buffer: [T, H] bf16 on HBM (updated)
    """
    _T, H = data.shape  # noqa: N806

    tile_size_m = 128
    tile_size_h = 512
    # H is compile-time constant: tile the free dim in Python, 512-wide tiles plus
    # a final short tile of width `H % 512` — arbitrary H, no padding.
    h_starts, h_widths = [], []
    _off = 0
    while _off < H:
        _w = min(tile_size_h, H - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    # Load num_m_tiles
    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    data_row_offset = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(data_row_offset, 0.0)

    for _ in nl.dynamic_range(0, num_m_tiles_reg, 1):
        # Load 128 indices
        idx_tile = nl.ndarray((tile_size_m, 1), nl.int32, nl.sbuf)
        nisa.dma_copy(
            dst=idx_tile[0:tile_size_m, 0:1],
            src=indices.ap(
                pattern=[[1, tile_size_m], [1, 1]],
                offset=0,
                scalar_offset=data_row_offset,
                indirect_dim=0,
            ),
        )

        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]
            # Load data tile (contiguous source rows)
            src_tile = nl.ndarray((tile_size_m, h_width), data.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=src_tile[0:tile_size_m, 0:h_width],
                src=data.ap(
                    pattern=[[H, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=data_row_offset,
                    indirect_dim=0,
                ),
            )

            # Load target rows using vector_offset (indirect read)
            target_tile = nl.ndarray((tile_size_m, h_width), data.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=target_tile[0:tile_size_m, 0:h_width],
                src=output_buffer.ap(
                    pattern=[[H, tile_size_m], [1, h_width]],
                    offset=h_start,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
            )

            # Add in SBUF
            sum_tile = nl.ndarray((tile_size_m, h_width), data.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=sum_tile[0:tile_size_m, 0:h_width],
                data1=target_tile[0:tile_size_m, 0:h_width],
                data2=src_tile[0:tile_size_m, 0:h_width],
                op=nl.add,
            )

            # Write back using vector_offset (indirect write)
            nisa.dma_copy(
                dst=output_buffer.ap(
                    pattern=[[H, tile_size_m], [1, h_width]],
                    offset=h_start,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=sum_tile[0:tile_size_m, 0:h_width],
            )

        nisa.tensor_scalar(data_row_offset, data_row_offset, nl.add, tile_size_m)

    nisa.dma_copy(output_buffer, output_buffer)
    return output_buffer
