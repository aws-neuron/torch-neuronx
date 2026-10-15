"""div: output = input_a / other  (plan #38 ``div.out`` / ``div.Tensor``).

One kernel per divisor form, since they differ in what is DMA'd per tile: a full
[m_static, hidden] tensor, a [m_static, 1] column broadcast along the free dim (the plan's
``[512,4] / [512,1]``), or a scalar the host replicates into a [128, 1] fp32 column.

There is no divide ALU op here ("unsupported operator 'divide'"), so every form computes
``a * reciprocal(b)`` with the reciprocal in fp32 -- which is what the static lowering does,
and the results are bit-identical to it.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .kernel_utils import _DTYPE_BYTES, _TILE_M, free_dim_tiles, run_dynamic_m_loop


def _div_tensor(dst, in_tiles, aux, h_width):
    """The reciprocal tile stays fp32 (where nisa.reciprocal computes anyway): rounding it to
    the operand dtype first would put two roundings in the result instead of one."""
    recip = nl.ndarray((_TILE_M, h_width), nl.float32, nl.sbuf)
    nisa.reciprocal(recip, in_tiles[1])
    nisa.tensor_tensor(dst=dst, data1=in_tiles[0], data2=recip, op=nl.multiply)


def _div_column(dst, in_tiles, aux, h_width):
    """tensor_scalar's operand0 is per-partition, so a [128, 1] tile broadcasts along the
    free dim."""
    recip = nl.ndarray((_TILE_M, 1), nl.float32, nl.sbuf)
    nisa.reciprocal(recip, in_tiles[1])
    nisa.tensor_scalar(dst, in_tiles[0], nl.multiply, recip)


def _div_baked_column(dst, in_tiles, aux, h_width):
    """``aux`` is the reciprocal computed once outside the loop: a scalar divisor never
    changes."""
    nisa.tensor_scalar(dst, in_tiles[0], nl.multiply, aux)


@nki.jit()
def div_tensor_kernel(input_a, input_b, ctrl):
    """Two [m_static, hidden] operands of one dtype, traced at 128 rows, run at any M >= 128.
    ctrl: [[full tiles], [tail row = M - 128]]."""
    m_static, hidden = input_a.shape
    h_starts, h_widths = free_dim_tiles(hidden, input_a.dtype)
    result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

    live_bytes = 3 * _DTYPE_BYTES.get(input_a.dtype, 2) + 4  # a, b, dst + fp32 reciprocal
    run_dynamic_m_loop(
        ctrl,
        (input_a, input_b),
        result,
        input_a.dtype,
        hidden,
        h_starts,
        h_widths,
        _div_tensor,
        live_bytes,
    )
    return result


@nki.jit()
def div_row_kernel(input_a, divisor, ctrl):
    """divisor is [m_static, 1] of the same dtype, broadcast along the free dim."""
    m_static, hidden = input_a.shape
    h_starts, h_widths = free_dim_tiles(hidden, input_a.dtype)
    result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

    live_bytes = 2 * _DTYPE_BYTES.get(input_a.dtype, 2)  # the [128, 1] column is negligible
    run_dynamic_m_loop(
        ctrl,
        (input_a,),
        result,
        input_a.dtype,
        hidden,
        h_starts,
        h_widths,
        _div_column,
        live_bytes,
        column=divisor,
    )
    return result


@nki.jit()
def div_scalar_kernel(input_a, divisor, ctrl):
    """divisor is the host's [128, 1] fp32 replication of a Python scalar."""
    m_static, hidden = input_a.shape
    h_starts, h_widths = free_dim_tiles(hidden, input_a.dtype)
    result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

    div_tile = nl.ndarray((_TILE_M, 1), nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=div_tile, src=divisor)
    recip = nl.ndarray((_TILE_M, 1), nl.float32, nl.sbuf)
    nisa.reciprocal(recip, div_tile)  # once for the whole kernel

    live_bytes = 2 * _DTYPE_BYTES.get(input_a.dtype, 2)
    run_dynamic_m_loop(
        ctrl,
        (input_a,),
        result,
        input_a.dtype,
        hidden,
        h_starts,
        h_widths,
        _div_baked_column,
        live_bytes,
        aux=recip,
    )
    return result
