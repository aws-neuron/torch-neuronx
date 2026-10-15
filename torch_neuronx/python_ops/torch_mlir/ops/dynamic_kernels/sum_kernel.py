"""
Free-dim Sum Reduction Kernel: output[:, 0] = sum(input, axis=free)

Reduces the last (free) dim of a [m_static, hidden] tensor to [m_static, 1].
Dynamic m-dim via num_m_tiles; hidden baked and tiled at trace time. Partials
accumulate in fp32 across free-dim segments, then cast back to the input dtype.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def sum_kernel(input_tensor, num_m_tiles):
    """Sum each input row over its free dimension.

    Args:
        input_tensor (nki.language.ndarray): Input with shape ``[m_static, hidden]``.
        num_m_tiles (nki.language.ndarray): Number of 128-row tiles to reduce.

    Returns:
        nki.language.ndarray: Row sums with shape ``[m_static, 1]`` in the input dtype.

    Note:
        Segment partials accumulate in fp32 before conversion to the input dtype.
        Rows after the requested tiles are not initialized.
    """
    m_static, hidden = input_tensor.shape

    tile_size_m = 128
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. hidden is a compile-time constant, so this tiling resolves at
    # trace time: tile_size_h-wide segments plus a final short tile of
    # `hidden % tile_size_h`. Each segment is reduced then accumulated in fp32.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_tensor.dtype), 2)
    tile_size_h = min(hidden, 4096 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((m_static, 1), input_tensor.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    def body(_i):
        # fp32 accumulator over the free-dim segments for this 128-row tile.
        """Reduce one 128-row tile over the free dimension.

        Args:
            _i (int): Dynamic loop iteration index, accepted but not inspected.
        """
        acc = nl.ndarray((tile_size_m, 1), nl.float32, nl.sbuf)
        nisa.memset(acc, 0.0)
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]
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
            seg_sum = nl.ndarray((tile_size_m, 1), nl.float32, nl.sbuf)
            nisa.tensor_reduce(
                dst=seg_sum[0:tile_size_m, 0:1],
                data=in_tile[0:tile_size_m, 0:h_width],
                op=nl.add,
                axis=[1],
            )
            nisa.tensor_tensor(
                dst=acc[0:tile_size_m, 0:1],
                data1=acc[0:tile_size_m, 0:1],
                data2=seg_sum[0:tile_size_m, 0:1],
                op=nl.add,
            )

        out_tile = nl.ndarray((tile_size_m, 1), input_tensor.dtype, nl.sbuf)
        nisa.tensor_copy(dst=out_tile[0:tile_size_m, 0:1], src=acc[0:tile_size_m, 0:1])
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[1, tile_size_m], [1, 1]],
                offset=0,
                scalar_offset=m_tile_start,
                indirect_dim=0,
            ),
            src=out_tile[0:tile_size_m, 0:1],
        )

        nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, tile_size_m)

    nl.fori_loop(0, num_m_tiles_reg, body)

    return result
