"""Dtype-cast copy kernel: output = input.to(out_dtype).

Backs both ``aten::_to_copy`` and ``aten::copy_``. H and both dtypes are baked, the
row count is a runtime value, so a seq-length change reuses one NEFF.

* **No host padding.** After ``M // 128`` full tiles the loop is followed by one
  straight-line *backward-aligned overlapping* tail tile at row ``M - 128``. A
  cast/copy is idempotent, so re-doing rows is harmless, and every access stays
  inside ``[0, M)`` — which is what lets ``copy_`` write ``self`` in place at any
  M >= 128. A *short* tail tile instead would need a runtime DMA count
  (P492035150 / NKI-2418), which only non-idempotent kernels really need.
* **Same-dtype copies skip the conversion** entirely.
* **LNC sharding splits the free dim** (exp_kernel's flavor): row-interleaving
  needs equal per-program trip counts, i.e. the padding this kernel avoids.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# The nl dtypes are their own names ("bfloat16" etc.), so str(dtype) keys in.
_DTYPE_BYTES = {
    nl.bfloat16: 2,
    nl.float16: 2,
    nl.float32: 4,
}

_TILE_M = 128

# Free-dim tile width in bytes/partition: the DMA saturation knee.
_TILE_H_BYTES = 4096


def _h_segments(h_span, tile_size_h, h_base):
    """Split ``h_span`` into tile_size_h-wide segments (final one short) based at
    ``h_base``. H is a compile-time constant, so this resolves at trace time."""
    starts, widths = [], []
    off = 0
    while off < h_span:
        w = min(tile_size_h, h_span - off)
        starts.append(h_base + off)
        widths.append(w)
        off += w
    return starts, widths


def _emit_tile(input_tensor, result, hidden, h_starts, h_widths, row_off, out_dtype, convert):
    """Convert one _TILE_M-row tile whose first row is the runtime value ``row_off``.

    Module-level because NKI only allows a nested function as a loop body, and both
    the loop body and the straight-line tail need this.
    """
    for seg in range(len(h_starts)):
        h_start = h_starts[seg]
        h_width = h_widths[seg]
        in_tile = nl.ndarray((_TILE_M, h_width), input_tensor.dtype, nl.sbuf)
        nisa.dma_copy(
            dst=in_tile[0:_TILE_M, 0:h_width],
            src=input_tensor.ap(
                pattern=[[hidden, _TILE_M], [1, h_width]],
                offset=h_start,
                scalar_offset=row_off,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

        if convert:
            # Goes via FP32 internally, so every bf16/fp16/fp32 pair is exact up to
            # the target's precision.
            out_tile = nl.ndarray((_TILE_M, h_width), out_dtype, nl.sbuf)
            nisa.tensor_copy(
                dst=out_tile[0:_TILE_M, 0:h_width],
                src=in_tile[0:_TILE_M, 0:h_width],
            )
        else:
            out_tile = in_tile

        nisa.dma_copy(
            dst=result.ap(
                pattern=[[hidden, _TILE_M], [1, h_width]],
                offset=h_start,
                scalar_offset=row_off,
                indirect_dim=0,
            ),
            src=out_tile[0:_TILE_M, 0:h_width],
            dge_mode=nisa.dge_mode.hwdge,
        )


def make_cast_kernel(out_dtype):
    """Build a dynamic-row cast kernel for a target dtype.

    Args:
        out_dtype (NKI dtype): Result dtype to bind into the kernel.

    Returns:
        Callable: JIT-compiled cast kernel.

    Note:
        The default-bound dtype is a trace-time constant that ``wrap_nki`` folds into
        the NEFF; a plain closure fails tracing with an unbound variable.
    """

    @nki.jit()
    def cast_kernel(input_tensor, num_rows, out_dtype=out_dtype):
        """Cast runtime-selected rows of a tensor to a bound dtype.

        Args:
            input_tensor (tensor): Source ``[m_static, hidden]`` tensor in HBM.
            num_rows (tensor): Int32 ``[1, 1]`` runtime row count, at least 128.
            out_dtype (NKI dtype, optional): Result dtype. Defaults to the dtype bound
                by ``make_cast_kernel``.

        Returns:
            tensor: ``[m_static, hidden]`` HBM tensor in ``out_dtype``.

        Note:
            A same-dtype copy skips conversion. A backward-aligned tail repeats rows
            rather than requiring an unsupported runtime DMA count. Only the first
            ``num_rows`` rows of the static-shaped result are initialized.
        """
        m_static, hidden = input_tensor.shape

        # Each SPMD program takes a contiguous hidden/num_programs slice; without it
        # every core moves the whole tensor -- 2x the HBM traffic at LNC=2. Equal
        # widths keep the per-core basic-block counts identical (NCC_IXGM002). Only
        # the free dim can be split: unequal per-program trip counts hang.
        num_progs = nl.num_programs(0)
        prog_id = nl.program_id(0)
        if hidden % num_progs == 0:
            h_span = hidden // num_progs
            h_base = prog_id * h_span
        else:
            h_span = hidden
            h_base = 0

        # Sized by the WIDER dtype so neither the load nor the store tile exceeds the knee.
        elem_bytes = max(
            _DTYPE_BYTES[str(input_tensor.dtype)],
            _DTYPE_BYTES[str(out_dtype)],
        )
        h_starts, h_widths = _h_segments(h_span, _TILE_H_BYTES // elem_bytes, h_base)

        convert = str(input_tensor.dtype) != str(out_dtype)

        result = nl.ndarray((m_static, hidden), out_dtype, nl.shared_hbm)

        num_rows_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=num_rows_sbuf, src=num_rows[0])

        # M // 128, on device because M is a runtime value.
        num_full = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="num_full_tiles")
        nisa.tensor_scalar(dst=num_full, data=num_rows_sbuf, op0=nl.right_shift, operand0=7)
        num_full_reg = nisa.register_alloc()
        nisa.register_load(num_full_reg, num_full)

        row_cursor = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="row_cursor")
        nisa.memset(row_cursor, 0.0)

        def body(_i):
            """Process one dynamic row-tile iteration.

            Args:
                _i (int): Loop iteration index.
            """
            _emit_tile(
                input_tensor, result, hidden, h_starts, h_widths, row_cursor, out_dtype, convert
            )
            nisa.tensor_scalar(row_cursor, row_cursor, nl.add, _TILE_M)

        nl.fori_loop(0, num_full_reg, body)

        # Unconditional tail tile at row M - 128: covers the M % 128 ragged rows with
        # no short DMA, and re-does the last full tile when M is a whole multiple.
        tail_row = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="tail_row")
        nisa.tensor_scalar(dst=tail_row, data=num_rows_sbuf, op0=nl.add, operand0=-_TILE_M)
        _emit_tile(input_tensor, result, hidden, h_starts, h_widths, tail_row, out_dtype, convert)

        return result

    return cast_kernel


# One kernel per target dtype, built at import: each gets its own @nki_op in
# ``wrappers`` (registration is global and must happen once per name).
cast_kernel_bf16 = make_cast_kernel(nl.bfloat16)
cast_kernel_fp16 = make_cast_kernel(nl.float16)
cast_kernel_fp32 = make_cast_kernel(nl.float32)
