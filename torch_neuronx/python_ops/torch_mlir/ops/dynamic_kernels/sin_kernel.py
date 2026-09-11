"""Sin Activation Kernel: output = sin(input)

Dynamic m-dim with single block size = 128.

The hardware ``nl.sin`` activation is a bare polynomial approximation valid only
near ``[-pi, pi]`` — it has NO argument range reduction, so for |x| beyond a few
radians it diverges catastrophically (e.g. sin(20) returns ~780). NKI exposes no
floor/round/mod op, but casting fp32 -> int32 rounds to nearest, which gives us
range reduction into ``[-pi, pi]`` in fp32 before the activation:

    k  = round(x / (2*pi))                 # fp32 -> int32 (round-to-nearest) -> fp32
    xr = x - 2*pi * k                      # in [-pi, pi]
    y  = sin(xr)                           # == sin(x)

fp32 compute (inputs/outputs stay in the caller dtype) keeps the reduction from
losing precision on small |x|.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

_TWO_PI = 2.0 * math.pi
_INV_TWO_PI = 1.0 / (2.0 * math.pi)


@nki.jit()
def sin_kernel(input_tensor, num_m_tiles):
    """
    Sin activation with range reduction: output = sin(input), any input range.

    Args:
        input_tensor: [m_static, hidden] on HBM
        num_m_tiles: [1, 1] int32 — number of 128-row tiles to process
    Returns:
        output: [m_static, hidden] on HBM
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

    def body(_i):
        for seg in range(len(h_starts)):
            h_start = h_starts[seg]
            h_width = h_widths[seg]
            # Load input tile (compute the whole reduction in fp32)
            in_tile = nl.ndarray((tile_size_m, h_width), nl.float32, nl.sbuf)
            nisa.dma_copy(
                dst=in_tile[0:tile_size_m, 0:h_width],
                src=input_tensor.ap(
                    pattern=[[hidden, tile_size_m], [1, h_width]],
                    offset=h_start,
                    scalar_offset=m_tile_start,
                    indirect_dim=0,
                ),
            )

            # Range reduction: k = round(x / 2pi) via fp32 -> int32 (round-to-nearest) -> fp32
            scaled = nl.ndarray((tile_size_m, h_width), nl.float32, nl.sbuf)
            nisa.tensor_scalar(
                dst=scaled[0:tile_size_m, 0:h_width],
                data=in_tile[0:tile_size_m, 0:h_width],
                op0=nl.multiply,
                operand0=_INV_TWO_PI,
            )
            k_int = nl.ndarray((tile_size_m, h_width), nl.int32, nl.sbuf)
            nisa.tensor_copy(
                dst=k_int[0:tile_size_m, 0:h_width], src=scaled[0:tile_size_m, 0:h_width]
            )
            k_f = nl.ndarray((tile_size_m, h_width), nl.float32, nl.sbuf)
            nisa.tensor_copy(dst=k_f[0:tile_size_m, 0:h_width], src=k_int[0:tile_size_m, 0:h_width])
            # k2 = 2pi * k
            k2 = nl.ndarray((tile_size_m, h_width), nl.float32, nl.sbuf)
            nisa.tensor_scalar(
                dst=k2[0:tile_size_m, 0:h_width],
                data=k_f[0:tile_size_m, 0:h_width],
                op0=nl.multiply,
                operand0=_TWO_PI,
            )
            # xr = x - 2pi*k  (in [-pi, pi])
            xr = nl.ndarray((tile_size_m, h_width), nl.float32, nl.sbuf)
            nisa.tensor_tensor(
                dst=xr[0:tile_size_m, 0:h_width],
                data1=in_tile[0:tile_size_m, 0:h_width],
                data2=k2[0:tile_size_m, 0:h_width],
                op=nl.subtract,
            )

            # sin(xr) == sin(x)
            out_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
            nisa.activation(
                dst=out_tile[0:tile_size_m, 0:h_width],
                data=xr[0:tile_size_m, 0:h_width],
                op=nl.sin,
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
