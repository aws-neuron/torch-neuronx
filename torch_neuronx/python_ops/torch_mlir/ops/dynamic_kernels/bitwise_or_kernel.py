"""bitwise_or Kernel: output = input_a | input_b  (torch.bitwise_or.Tensor)

Elementwise bitwise OR. The target workload uses it to combine bool attention
masks (bool = 0/1, so bitwise OR == logical OR). A 1-byte (bool) tensor cannot be
DMA'd with the per-tile dynamic offset (only the offset-0 tile reads correctly),
so the adapter passes the operands as int16 views ([m, H//2], zero-copy) and
views the int16 output back to bool. Bitwise OR is bit-parallel, so OR-ing the
int16-packed pairs is exactly OR-ing the underlying bools — no unpacking needed.
Requires an even H (enforced in the adapter). Dynamic m-dim, block size = 128; the
(halved) free dim is baked and tiled at trace time.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def bitwise_or_kernel(input_a, input_b, num_m_tiles):
    """Apply bitwise OR to int16-packed Boolean-mask operands.

    Args:
        input_a (tensor): Int16 view of the first ``bool[m_static, H]`` HBM operand.
        input_b (tensor): Int16 view of the second ``bool[m_static, H]`` HBM operand.
        num_m_tiles (tensor): Int32 ``[1, 1]`` count of 128-row tiles per program.

    Returns:
        tensor: Packed int16 result for the caller to view as Boolean values.

    Note:
        Packing is bit-exact for Boolean bytes and requires an even hidden width.
        Rows outside the selected tiles remain unspecified in the static-capacity result.
    """
    m_static, hidden = input_a.shape

    tile_size_m = 128
    # Programs process interleaved row tiles; the adapter equalizes loop counts.
    num_programs = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    program_row_offset = program_id * tile_size_m * hidden
    # Free-dim tile width ~4 KB/partition (DMA saturation knee), same rule as
    # add/sub/mul; the operands are int16 views, so the element size is fixed at 2.
    tile_size_h = min(hidden, 4096 // 2)
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
        """Apply packed bitwise OR for one interleaved row tile.

        Args:
            _i (int): Loop index required by ``fori_loop`` and otherwise unused.
        """
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

            out_tile = nl.ndarray((tile_size_m, h_width), input_a.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=out_tile[0:tile_size_m, 0:h_width],
                data1=a_tile[0:tile_size_m, 0:h_width],
                data2=b_tile[0:tile_size_m, 0:h_width],
                op=nl.bitwise_or,
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
