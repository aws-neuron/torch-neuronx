"""
SiLU Backward Kernel: output = grad_output * silu_dx(input)

where silu_dx(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                  = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

This is the elementwise backward of silu: given grad_output (upstream gradient)
and the original input to silu, computes the gradient w.r.t. the input.

Dynamic m-dim with single block size = 128; arbitrary H (free dim tiled at trace
time in 512-wide tiles plus a short remainder). Uses nisa.activation with
op=nl.silu_dx for the derivative, then nisa.tensor_tensor multiply with grad_output.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit()
def silu_bwd_kernel(grad_output, input_tensor, num_m_tiles):
    """Compute SiLU input gradients over dynamic row tiles.

    Args:
        grad_output (nki.language.ndarray): Upstream gradient with shape ``[m_static, hidden]``.
        input_tensor (nki.language.ndarray): Original SiLU input with the same shape.
        num_m_tiles (nki.language.ndarray): Number of 128-row tiles to process.

    Returns:
        nki.language.ndarray: Static-shaped ``grad_output * silu_dx(input_tensor)``
        in the ``grad_output`` dtype.

    Note:
        Rows after the requested tiles are not initialized.
    """
    m_static, hidden = grad_output.shape

    tile_size_m = 128
    tile_size_h = 512
    # H is a compile-time constant: tile the free dim in Python, 512-wide tiles plus
    # a final short tile of width `hidden % 512` — arbitrary H, no padding.
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(_off)
        h_widths.append(_w)
        _off += _w

    result = nl.ndarray((m_static, hidden), grad_output.dtype, nl.shared_hbm)

    # Load num_m_tiles
    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])
    num_m_tiles_reg = nisa.register_alloc()
    nisa.register_load(num_m_tiles_reg, num_m_tiles_sbuf)

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    def body(_i):
        """Compute SiLU gradients for one 128-row tile.

        Args:
            _i (int): Dynamic loop iteration index, accepted but unused.
        """
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]

            # Load grad_output tile
            grad_tile = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.dma_copy(
                dst=grad_tile[0:tile_size_m, 0:h_width],
                src=grad_output.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # Load input tile (original input to silu forward)
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

            # Compute silu_dx(input) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            silu_dx_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
            nisa.activation(
                dst=silu_dx_tile[0:tile_size_m, 0:h_width],
                data=in_tile[0:tile_size_m, 0:h_width],
                op=nl.silu_dx,
            )

            # Multiply: output = grad_output * silu_dx(input)
            out_tile = nl.ndarray((tile_size_m, h_width), grad_output.dtype, nl.sbuf)
            nisa.tensor_tensor(
                dst=out_tile[0:tile_size_m, 0:h_width],
                data1=grad_tile[0:tile_size_m, 0:h_width],
                data2=silu_dx_tile[0:tile_size_m, 0:h_width],
                op=nl.multiply,
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

    nl.fori_loop(0, num_m_tiles_reg, body)

    return result
