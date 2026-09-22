"""Strided-view -> contiguous copy kernel with a dynamic row count.

Backs ``aten::copy_`` / ``aten::_to_copy`` whose source is a strided view -- the case the
C++ prologue used to absorb, baking the row count into the NEFF and recompiling per token
count. ``dyn_stride``/``width``/``fanout`` arrive as tensor SHAPES so they bake themselves
through the input specs (as ``hidden`` does in ``cast_kernel``); ``rows`` and the view's
first element offset ride in ``meta``'s VALUES, so one NEFF covers every row count and
every storage offset -- the two ``.chunk(2)`` halves share it.

* ``base`` is a FLAT [N, 1] view, so ``scalar_offset`` steps in ELEMENTS: that is what
  makes the in-row offset a runtime value. Folding it into the bound tensor's
  storage_offset instead trips NRT's "input tensor is too small" check.
* The tail tile is backward-aligned and overlapping, so no short DMA is needed (a runtime
  DMA count is unsupported, NKI-2418). A copy is idempotent, so re-doing rows is harmless.
* ``tile_m`` rows per tile is baked too: 128 fills the partition dim, and ``rows < 128``
  uses 1 so the tail's ``rows - tile_m`` offset stays non-negative. Same body either way.
* No LNC sharding: splitting needs equal per-program trip counts, and the writes are
  idempotent, so every core redundantly running the whole copy is correct.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_DTYPE_BYTES = {
    nl.bfloat16: 2,
    nl.float16: 2,
    nl.float32: 4,
    nl.int32: 4,
}

_TILE_M = 128

# Free-dim tile width in bytes/partition: the DMA saturation knee (cast_kernel's value).
_TILE_H_BYTES = 4096

# The free-dim `for seg` loop is unrolled statically by NKI (cap 10000, Reduce.h
# kMaxUnrollIters). Cap the segment count well under that; wider copies are declined
# upstream and fall back to the default lowering.
_MAX_H_SEGMENTS = 8192


def width_within_unroll_budget(width, elem_bytes):
    """False when ceil(width / knee) exceeds the unroll budget -> adapter declines."""
    knee = _TILE_H_BYTES // elem_bytes
    return -(-width // knee) <= _MAX_H_SEGMENTS


def _h_segments(width, elem_bytes):
    """Split ``width`` into knee-wide segments (final one short)."""
    tile_h = _TILE_H_BYTES // elem_bytes
    starts, widths = [], []
    off = 0
    while off < width:
        w = min(tile_h, width - off)
        starts.append(off)
        widths.append(w)
        off += w
    return starts, widths


def _emit_tile(
    base,
    result,
    dyn_stride,
    width,
    fanout,
    tile_m,
    h_starts,
    h_widths,
    in_cur,
    out_cur,
    dt,
    convert,
):
    """Copy one tile_m-row tile. ``in_cur`` is a runtime ELEMENT offset into ``base``,
    ``out_cur`` a runtime ROW offset into ``result``. Module-level because NKI allows a
    nested function only as a loop body, and the tail tile needs this too."""
    for seg in range(len(h_starts)):
        h_start = h_starts[seg]
        w = h_widths[seg]

        in_tile = nl.ndarray((tile_m, w), base.dtype, nl.sbuf)
        nisa.dma_copy(
            dst=in_tile[0:tile_m, 0:w],
            src=base.ap(
                pattern=[[dyn_stride, tile_m], [1, w]],
                offset=h_start,
                scalar_offset=in_cur,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )

        if convert:
            out_tile = nl.ndarray((tile_m, w), dt, nl.sbuf)
            nisa.tensor_copy(dst=out_tile[0:tile_m, 0:w], src=in_tile[0:tile_m, 0:w])
        else:
            out_tile = in_tile

        # fanout > 1 implies width == 1, so there is exactly one segment and w is 1.
        if fanout > 1:
            store_src = nl.broadcast_to(out_tile[0:tile_m, 0:1], (tile_m, fanout))
            store_w = fanout
        else:
            store_src = out_tile[0:tile_m, 0:w]
            store_w = w

        nisa.dma_copy(
            dst=result.ap(
                pattern=[[width * fanout, tile_m], [1, store_w]],
                offset=h_start * fanout,
                scalar_offset=out_cur,
                indirect_dim=0,
            ),
            src=store_src,
            dge_mode=nisa.dge_mode.hwdge,
        )


def make_strided_copy_kernel(out_dtype, tile_m=_TILE_M):
    """Build a strided-copy kernel whose result dtype is ``out_dtype``; None keeps the
    source dtype (the passthrough variant, and the only one that can move int32).
    ``tile_m`` is the rows per tile. Both are default-bound so the tracer reads them as
    local constants -- a closure fails to trace."""

    @nki.jit()
    def strided_copy_kernel(
        base, stride_proto, width_proto, fan_proto, meta, out_dtype=out_dtype, tile_m=tile_m
    ):
        """
        Gather a strided view into a contiguous result.

        Args:
            base:         [N, 1]           flat view of the source storage
            stride_proto: [1, dyn_stride]  shape-only: source row stride
            width_proto:  [1, width]       shape-only: contiguous run length
            fan_proto:    [1, fanout]      shape-only: stride-0 fan-out (1 = none)
            meta:         [4, 1] int32     [rows // tile_m, in_start,
                                            in_start + (rows - tile_m) * dyn_stride,
                                            rows - tile_m]
        Returns:
            result: [rows_static, width * fanout] on HBM
        """
        dyn_stride = stride_proto.shape[1]
        width = width_proto.shape[1]
        fanout = fan_proto.shape[1]
        out_width = width * fanout
        rows_static = base.shape[0] // dyn_stride

        if out_dtype is None:
            dt = base.dtype
            convert = False
        else:
            dt = out_dtype
            convert = str(base.dtype) != str(out_dtype)

        # Sized by the WIDER dtype so neither the load nor the store tile exceeds the knee.
        elem_bytes = max(_DTYPE_BYTES[str(base.dtype)], _DTYPE_BYTES[str(dt)])
        h_starts, h_widths = _h_segments(width, elem_bytes)

        result = nl.ndarray((rows_static, out_width), dt, nl.shared_hbm)

        # One HBM read per scalar: a single-partition tensor_copy needs src partition start
        # in {0, 32, 64, 96}, so slicing row k out of one SBUF tile is not allowed.
        trips = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="trips")
        nisa.dma_copy(dst=trips, src=meta[0])
        in_cur = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="in_cursor")
        nisa.dma_copy(dst=in_cur, src=meta[1])
        in_tail = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="in_tail")
        nisa.dma_copy(dst=in_tail, src=meta[2])
        out_tail = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="out_tail")
        nisa.dma_copy(dst=out_tail, src=meta[3])

        out_cur = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="out_cursor")
        nisa.memset(out_cur, 0.0)

        # Touch each proto so it is not a dead parameter the compiler could drop, which
        # would leave execute_compiled_graph binding more inputs than the NEFF has.
        touch = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name="proto_touch")
        nisa.dma_copy(dst=touch, src=stride_proto[0, 0:1])
        nisa.dma_copy(dst=touch, src=width_proto[0, 0:1])
        nisa.dma_copy(dst=touch, src=fan_proto[0, 0:1])

        trips_reg = nisa.register_alloc()
        nisa.register_load(trips_reg, trips)

        def body(_i):
            _emit_tile(
                base,
                result,
                dyn_stride,
                width,
                fanout,
                tile_m,
                h_starts,
                h_widths,
                in_cur,
                out_cur,
                dt,
                convert,
            )
            nisa.tensor_scalar(in_cur, in_cur, nl.add, tile_m * dyn_stride)
            nisa.tensor_scalar(out_cur, out_cur, nl.add, tile_m)

        nl.fori_loop(0, trips_reg, body)

        # Backward-aligned tail: covers the rows % tile_m ragged rows with no short DMA,
        # and re-does the last full tile when rows is a whole multiple.
        _emit_tile(
            base,
            result,
            dyn_stride,
            width,
            fanout,
            tile_m,
            h_starts,
            h_widths,
            in_tail,
            out_tail,
            dt,
            convert,
        )

        return result

    return strided_copy_kernel


# One kernel per target dtype, built at import: each gets its own @nki_op in ``wrappers``
# (registration is global and must happen once per name).
strided_copy_kernel_same = make_strided_copy_kernel(None)
strided_copy_kernel_bf16 = make_strided_copy_kernel(nl.bfloat16)
strided_copy_kernel_fp16 = make_strided_copy_kernel(nl.float16)
strided_copy_kernel_fp32 = make_strided_copy_kernel(nl.float32)
# Sub-tile rows. Only same-dtype reaches it: copy_ requires matching dtypes for a strided
# source, and contiguous cannot change dtype.
strided_copy_kernel_same_row = make_strided_copy_kernel(None, tile_m=1)
