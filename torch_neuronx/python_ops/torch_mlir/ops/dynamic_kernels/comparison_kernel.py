"""ge/eq/lt/ne/le: output = (input <cmp> other), a bool mask.

``other`` is a scalar (runtime threshold operand) or a same-shape tensor. One NEFF per
(op, overload, H, dtype) covers every M, and every threshold on the scalar path.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .kernel_utils import _DTYPE_BYTES, _TILE_M, free_dim_tiles, run_dynamic_m_loop


def _cmp_scalar(dst, in_tiles, aux, h_width):
    """``aux`` = (cmp_op, [128, 1] fp32 threshold): tensor_scalar's operand0 is per-partition,
    which is why the host replicates the scalar into a column."""
    nisa.tensor_scalar(dst, in_tiles[0], aux[0], aux[1])


def _cmp_tensor(dst, in_tiles, aux, h_width):
    """dst = a <cmp_op> b, both full width."""
    nisa.tensor_tensor(dst=dst, data1=in_tiles[0], data2=in_tiles[1], op=aux)


def make_comparison_kernel(cmp_op):
    """``cmp_op`` is default-bound, not captured: NKI can't resolve closure free variables."""

    @nki.jit()
    def comparison_kernel(input_tensor, threshold, ctrl, cmp_op=cmp_op):
        """[m_static, hidden] HBM traced at 128 rows, run at any M >= 128. threshold: [128, 1]
        fp32 replication of the scalar. ctrl: [[full tiles], [tail row = M - 128]]."""
        m_static, hidden = input_tensor.shape
        h_starts, h_widths = free_dim_tiles(hidden, input_tensor.dtype)

        result = nl.ndarray((m_static, hidden), nl.bool_, nl.shared_hbm)

        thr = nl.ndarray((_TILE_M, 1), nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=thr, src=threshold)

        live_bytes = _DTYPE_BYTES.get(input_tensor.dtype, 2) + 1  # operand + 1-byte mask
        run_dynamic_m_loop(
            ctrl,
            (input_tensor,),
            result,
            nl.bool_,
            hidden,
            h_starts,
            h_widths,
            _cmp_scalar,
            live_bytes,
            aux=(cmp_op, thr),
        )
        return result

    return comparison_kernel


def make_binary_mask_kernel(cmp_op):
    """The ``.Tensor`` overload: a second full-width operand instead of the threshold column.
    Also serves logical_and on float operands."""

    @nki.jit()
    def binary_mask_kernel(input_a, input_b, ctrl, cmp_op=cmp_op):
        """Two [m_static, hidden] operands of one dtype, traced at 128 rows, run at any M."""
        m_static, hidden = input_a.shape
        h_starts, h_widths = free_dim_tiles(hidden, input_a.dtype)

        result = nl.ndarray((m_static, hidden), nl.bool_, nl.shared_hbm)

        live_bytes = 2 * _DTYPE_BYTES.get(input_a.dtype, 2) + 1
        run_dynamic_m_loop(
            ctrl,
            (input_a, input_b),
            result,
            nl.bool_,
            hidden,
            h_starts,
            h_widths,
            _cmp_tensor,
            live_bytes,
            aux=cmp_op,
        )
        return result

    return binary_mask_kernel


_CMP_OPS = {
    "ge": nl.greater_equal,
    "eq": nl.equal,
    "lt": nl.less,
    "ne": nl.not_equal,
    "le": nl.less_equal,
}
