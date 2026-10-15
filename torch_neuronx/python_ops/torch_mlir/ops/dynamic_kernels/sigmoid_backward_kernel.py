"""
Sigmoid Backward Kernel: grad_input = grad_output * output * (1 - output)

Two separate inputs (grad_output, output=sigmoid(x)), dynamic m-dim with single
block size = 128. Mirrors the eager decomposition
``grad_output * (output * (1 - output))`` — no fp32 upcast (the reference
decomposition computes in the input dtype).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def sigmoid_backward_kernel(grad_output, output, num_m_tiles):
    """Compute sigmoid input gradients over dynamic row tiles.

    Args:
        grad_output (nki.language.ndarray): Gradient with respect to the sigmoid output.
        output (nki.language.ndarray): Sigmoid output with shape ``[m_static, hidden]``.
        num_m_tiles (nki.language.ndarray): Number of 128-row tiles to process.

    Returns:
        nki.language.ndarray: Tensor with the gradient shape and dtype whose first
            ``128 * num_m_tiles[0]`` rows contain ``grad_output * output * (1 - output)``.

    Note:
        Later rows are left unwritten. Arithmetic remains in the input dtype without an
        fp32 upcast.
    """
    m_static, hidden = grad_output.shape

    tile_size_m = 128
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H compile-time constant -> resolved at trace time; tile_size_h-
    # wide tiles plus a final short tile of `hidden % tile_size_h` — arbitrary H.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(grad_output.dtype), 2)
    tile_size_h = min(hidden, 4096 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((m_static, hidden), grad_output.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    def body(_i):
        """Compute one dynamic row-tile iteration.

        Args:
            _i (int): Loop iteration index accepted but not used.
        """
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]

            go_tile = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=go_tile[0:tile_size_m, 0:h_width],
                src=grad_output.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            out_s_tile = nl.ndarray((tile_size_m, h_width), output.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=out_s_tile[0:tile_size_m, 0:h_width],
                src=output.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # one_minus = 1 - output  via  (output * -1) + 1
            one_minus = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.tensor_scalar(
                one_minus[0:tile_size_m, 0:h_width],
                out_s_tile[0:tile_size_m, 0:h_width],
                nl.multiply,
                -1.0,
                op1=nl.add,
                operand1=1.0,
            )

            # dsig = output * (1 - output)
            dsig = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=dsig[0:tile_size_m, 0:h_width],
                data1=out_s_tile[0:tile_size_m, 0:h_width],
                data2=one_minus[0:tile_size_m, 0:h_width],
                op=nl.multiply,
            )

            # grad_input = grad_output * dsig
            out_tile = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=out_tile[0:tile_size_m, 0:h_width],
                data1=go_tile[0:tile_size_m, 0:h_width],
                data2=dsig[0:tile_size_m, 0:h_width],
                op=nl.multiply,
            )

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

    nl.fori_loop(0, num_m_tiles_reg, body)

    return result
