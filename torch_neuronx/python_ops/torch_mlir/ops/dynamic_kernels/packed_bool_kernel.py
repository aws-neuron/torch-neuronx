"""Binary bool op on int16-packed masks: output = input_a <alu_op> input_b.

A 1-byte tensor cannot be DMA'd with a per-tile dynamic offset (only the offset-0 tile reads
correctly), so the adapter passes int16 views ([m, H//2], zero-copy) and views the result back
to bool. Bool bytes are 0/1 and bitwise ops are bit-parallel, so the packed op is exactly the
elementwise one -- no unpacking. Needs an even H (enforced in the adapter).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .kernel_utils import free_dim_tiles, run_dynamic_m_loop

# Two int16 operand tiles plus the int16 destination, per (slot, h-segment) element.
_LIVE_BYTES = 3 * 2


def _packed_binary(dst, in_tiles, aux, h_width):
    """dst = a <alu_op> b on the packed masks."""
    nisa.tensor_tensor(dst=dst, data1=in_tiles[0], data2=in_tiles[1], op=aux)


def make_packed_bool_kernel(alu_op):
    """``alu_op`` must be a bitwise op (it acts per byte) and is default-bound, not captured:
    NKI can't resolve closure free variables."""

    @nki.jit()
    def packed_bool_kernel(input_a, input_b, ctrl, alu_op=alu_op):
        """Two [m_static, hidden] int16 HBM operands -- bool[m_static, 2*hidden] views."""
        m_static, hidden = input_a.shape
        h_starts, h_widths = free_dim_tiles(hidden, input_a.dtype)

        result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

        run_dynamic_m_loop(
            ctrl,
            (input_a, input_b),
            result,
            input_a.dtype,
            hidden,
            h_starts,
            h_widths,
            _packed_binary,
            _LIVE_BYTES,
            aux=alu_op,
        )
        return result

    return packed_bool_kernel
