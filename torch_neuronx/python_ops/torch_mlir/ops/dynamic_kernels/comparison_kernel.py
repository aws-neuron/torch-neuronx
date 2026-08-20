"""ge/lt/ne/le: output = (input <cmp> other), a bool mask.

M is runtime and unpadded (see the tail tile), and the threshold is a runtime
operand, so one NEFF per (op, H, dtype) covers every M and every scalar.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_TILE_M = 128
_TILE_BYTES = 4096  # free-dim tile size per partition: the DMA saturation knee
_DTYPE_BYTES = {nl.bfloat16: 2, nl.float16: 2, nl.float32: 4}


def _emit_cmp_tile(input_tensor, result, thr, row, hidden, h_starts, h_widths, cmp_op):
    """One 128-row tile at runtime row offset ``row``: load, compare, store the mask.
    Module-level because NKI rejects calls to functions nested inside a jit kernel."""
    for seg in range(len(h_starts)):
        h_start, h_width = h_starts[seg], h_widths[seg]
        in_tile = nl.ndarray((_TILE_M, h_width), input_tensor.dtype, nl.sbuf)
        nisa.dma_copy(
            dst=in_tile,
            src=input_tensor.ap(
                pattern=[[hidden, _TILE_M], [1, h_width]],
                offset=h_start,
                scalar_offset=row,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        out_tile = nl.ndarray((_TILE_M, h_width), nl.bool_, nl.sbuf)
        nisa.tensor_scalar(out_tile, in_tile, cmp_op, thr)
        nisa.dma_copy(
            dst=result.ap(
                pattern=[[hidden, _TILE_M], [1, h_width]],
                offset=h_start,
                scalar_offset=row,
                indirect_dim=0,
            ),
            src=out_tile,
            dge_mode=nisa.dge_mode.hwdge,
        )


def make_comparison_kernel(cmp_op):
    """Build the dynamic-M comparison kernel for ALU op ``cmp_op``. ``cmp_op`` is
    default-bound, not captured: NKI's tracer can't resolve closure free variables."""

    @nki.jit()
    def comparison_kernel(input_tensor, threshold, ctrl, cmp_op=cmp_op):
        """input_tensor: [m_static, hidden] HBM, compiled at 128 rows, run at any M >= 128.
        threshold: [128, 1] fp32 — replicated, since nisa.tensor_scalar needs operand0's
        partitions to match the dst tile's. ctrl: [[full tiles], [tail row = M - 128]]."""
        m_static, hidden = input_tensor.shape

        # Split the free dim per SPMD program; without it every core would process the
        # whole tensor. Row interleaving would be the alternative but needs M padded to
        # 128 * num_programs, since unequal per-program trip counts hang the device.
        num_progs = nl.num_programs(axes=0)
        prog_id = nl.program_id(axis=0)
        if hidden % num_progs == 0:
            h_span = hidden // num_progs
            h_base = prog_id * h_span
        else:
            h_span = hidden
            h_base = 0

        # hidden is compile-time, so the free-dim tiling resolves here: full-width
        # tiles plus a final short one, i.e. arbitrary H with no padding.
        tile_size_h = min(h_span, _TILE_BYTES // _DTYPE_BYTES.get(input_tensor.dtype, 2))
        h_starts, h_widths = [], []
        _off = 0
        while _off < h_span:
            _w = min(tile_size_h, h_span - _off)
            h_starts.append(h_base + _off)
            h_widths.append(_w)
            _off += _w

        result = nl.ndarray((m_static, hidden), nl.bool_, nl.shared_hbm)

        thr = nl.ndarray((_TILE_M, 1), nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=thr, src=threshold)

        num_full_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=num_full_sbuf, src=ctrl[0])
        num_full_reg = nisa.register_alloc()
        nisa.register_load(num_full_reg, num_full_sbuf)

        tail_row = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tail_row, src=ctrl[1])

        m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
        nisa.memset(m_tile_start, 0.0)

        def body(_i):
            _emit_cmp_tile(
                input_tensor, result, thr, m_tile_start, hidden, h_starts, h_widths, cmp_op
            )
            nisa.tensor_scalar(m_tile_start, m_tile_start, nl.add, _TILE_M)

        nl.fori_loop(0, num_full_reg, body)

        # Tail tile at row M - 128, run unconditionally: it covers the remainder when
        # M % 128 != 0 and repeats the last whole tile otherwise, which is harmless
        # because the compare is idempotent. Keeps every access inside [0, M) -> no pad.
        _emit_cmp_tile(input_tensor, result, thr, tail_row, hidden, h_starts, h_widths, cmp_op)

        return result

    return comparison_kernel


_CMP_OPS = {
    "ge": nl.greater_equal,
    "lt": nl.less,
    "ne": nl.not_equal,
    "le": nl.less_equal,
}
