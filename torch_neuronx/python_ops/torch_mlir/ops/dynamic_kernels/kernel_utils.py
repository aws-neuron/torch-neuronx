"""Shared skeleton for the no-pad dynamic elementwise kernels: bake H, split the free dim
per program, walk a runtime row count with an unrolled fori_loop, close with an overlapping
tail tile. A kernel supplies only ``compute(dst, in_tiles, aux, h_width)``, which must be
module-level -- NKI rejects calls to functions nested inside a jit kernel.
"""

import nki.isa as nisa
import nki.language as nl

_TILE_M = 128
_TILE_BYTES = 4096  # free-dim tile size per partition: the DMA saturation knee
_DTYPE_BYTES = {nl.bfloat16: 2, nl.float16: 2, nl.float32: 4, nl.int16: 2, nl.bool_: 1}

# Row tiles per trip. Each trip pays a ~7us back-edge fence that does not scale with the
# data it moves, so fatter trips is the only lever. Swept 1..32: 8 is the knee (see the PR
# tables). Power of two -- the trip-count split shifts.
_UNROLL_MAX = 8

# Per-partition SBUF ceiling for one trip (trn2 has 192 KB). Phase batching keeps every
# slot's tiles live, so a large H drops the unroll factor instead of overflowing SBUF.
_TRIP_SBUF_BUDGET = 128 * 1024


def free_dim_tiles(hidden, dtype):
    """(h_starts, h_widths) for this program's slice of a ``hidden``-wide row: full tiles plus
    a final short one, resolved at trace time since H is compile-time. Splitting the FREE dim
    (not rows) is what keeps M unpadded -- unequal per-program trip counts hang the device.
    """
    num_progs = nl.num_programs(axes=0)
    prog_id = nl.program_id(axis=0)
    if hidden % num_progs == 0:
        h_span = hidden // num_progs
        h_base = prog_id * h_span
    else:
        h_span = hidden
        h_base = 0

    tile_size_h = min(h_span, _TILE_BYTES // _DTYPE_BYTES.get(dtype, 2))
    h_starts, h_widths = [], []
    off = 0
    while off < h_span:
        width = min(tile_size_h, h_span - off)
        h_starts.append(h_base + off)
        h_widths.append(width)
        off += width
    return h_starts, h_widths


def unroll_factor(h_widths, live_bytes_per_element):
    """Largest power-of-two unroll whose live tiles fit the budget. ``live_bytes_per_element``
    counts every tile one (slot, h-segment) holds: operands, destination, scratch. Depends only
    on compile-time values, so both programs pick the same factor (NCC_IXGM002)."""
    per_slot = len(h_widths) * max(h_widths) * live_bytes_per_element
    unroll, log2_unroll = 1, 0
    while unroll * 2 <= _UNROLL_MAX and (unroll * 2) * per_slot <= _TRIP_SBUF_BUDGET:
        unroll *= 2
        log2_unroll += 1
    return unroll, log2_unroll


def trip_stages(unroll, log2_unroll):
    """[(width, shift, mask)], one fori_loop per stage: ``n >> log2(unroll)`` full-width trips,
    then one trip per leftover bit, then a width-1 mop-up. Loop entry costs a fence even at
    zero trips, so the cascade stops at width 4 rather than continuing to 2.
    """
    if unroll == 1:
        return [(1, 0, 0)]
    if unroll <= 4:
        return [(unroll, log2_unroll, 0), (1, 0, unroll - 1)]
    stages, width, shift = [], unroll, log2_unroll
    while width >= 4:
        stages.append((width, shift, 0 if width == unroll else 1))
        width //= 2
        shift -= 1
    stages.append((1, 0, 3))  # the width-4 stage took bit 2; bits 0-1 remain
    return stages


def loop_control(ctrl):
    """Whole-tile count (M // 128), tail row offset (M - 128) and the row cursor, read out of
    the [2, 1] int32 ``ctrl`` operand. Runtime values: that is what keeps M out of the NEFF."""
    num_full_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_full_sbuf, src=ctrl[0])

    tail_row = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tail_row, src=ctrl[1])

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)
    return num_full_sbuf, tail_row, m_tile_start


def row_cursors(m_tile_start, unroll):
    """Per-slot row cursors. Slot k > 0 needs its own scalar: a compile-time ``offset=``
    cannot leave the traced extent, so only a runtime cursor addresses later tiles."""
    cursors = [m_tile_start]
    for k in range(1, unroll):
        cursors.append(nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"row_start_{k}"))
    return cursors


def stage_count_reg(num_full_sbuf, shift, mask, tag):
    """This stage's trip count as a loop register, shifted and masked on device."""
    count_sbuf = num_full_sbuf
    if shift > 0:
        shifted = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"n_shr{tag}")
        nisa.tensor_scalar(dst=shifted, data=count_sbuf, op0=nl.right_shift, operand0=shift)
        count_sbuf = shifted
    if mask > 0:
        masked = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"n_msk{tag}")
        nisa.tensor_scalar(dst=masked, data=count_sbuf, op0=nl.bitwise_and, operand0=mask)
        count_sbuf = masked
    count_reg = nisa.register_alloc()
    nisa.register_load(count_reg, count_sbuf)
    return count_reg


def _load_tile(src, row, hidden, h_start, h_width):
    tile = nl.ndarray((_TILE_M, h_width), src.dtype, nl.sbuf)
    nisa.dma_copy(
        dst=tile,
        src=src.ap(
            pattern=[[hidden, _TILE_M], [1, h_width]],
            offset=h_start,
            scalar_offset=row,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return tile


def _load_column(src, row):
    """[128, 1] tile of a [m_static, 1] operand at row offset ``row`` (row stride 1)."""
    tile = nl.ndarray((_TILE_M, 1), src.dtype, nl.sbuf)
    nisa.dma_copy(
        dst=tile,
        src=src.ap(pattern=[[1, _TILE_M], [1, 1]], scalar_offset=row, indirect_dim=0),
        dge_mode=nisa.dge_mode.hwdge,
    )
    return tile


def emit_trip(
    width,
    row_start,
    inputs,
    result,
    out_dtype,
    hidden,
    h_starts,
    h_widths,
    compute,
    aux=None,
    column=None,
):
    """Emit ``width`` consecutive 128-row tiles: every load, then every compute, then every
    store. Per-slot ordering instead lets the SBUF allocator fold the slots onto one buffer
    pair, which caps pipelining at 2-deep. ``column`` is an optional [m_static, 1] operand
    broadcast along the free dim.

    Plain loops only: NKI's parser rejects comprehensions and starred unpacking in a kernel.
    """
    in_tiles = []
    for slot in range(width):
        col = None
        if column is not None:
            col = _load_column(column, row_start[slot])
        seg_tiles = []
        for seg in range(len(h_starts)):
            tiles = []
            for src_idx in range(len(inputs)):
                tiles.append(
                    _load_tile(
                        inputs[src_idx], row_start[slot], hidden, h_starts[seg], h_widths[seg]
                    )
                )
            if col is not None:
                tiles.append(col)
            seg_tiles.append(tiles)
        in_tiles.append(seg_tiles)

    out_tiles = []
    for slot in range(width):
        seg_tiles = []
        for seg in range(len(h_starts)):
            dst = nl.ndarray((_TILE_M, h_widths[seg]), out_dtype, nl.sbuf)
            compute(dst, in_tiles[slot][seg], aux, h_widths[seg])
            seg_tiles.append(dst)
        out_tiles.append(seg_tiles)

    for slot in range(width):
        for seg in range(len(h_starts)):
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[hidden, _TILE_M], [1, h_widths[seg]]],
                    offset=h_starts[seg],
                    scalar_offset=row_start[slot],
                    indirect_dim=0,
                ),
                src=out_tiles[slot][seg],
                dge_mode=nisa.dge_mode.hwdge,
            )


def run_dynamic_m_loop(
    ctrl,
    inputs,
    result,
    out_dtype,
    hidden,
    h_starts,
    h_widths,
    compute,
    live_bytes_per_element,
    aux=None,
    column=None,
):
    """Walk every row of a runtime M. The tail tile runs unconditionally at row M - 128: it
    covers the remainder, and when M % 128 == 0 it just rewrites the last tile (these ops are
    idempotent). Every access stays inside [0, M), so nothing is padded and ``out=`` can back
    the result."""
    unroll, log2_unroll = unroll_factor(h_widths, live_bytes_per_element)
    num_full_sbuf, tail_row, m_tile_start = loop_control(ctrl)
    row_start = row_cursors(m_tile_start, unroll)

    stages = trip_stages(unroll, log2_unroll)
    for tag in range(len(stages)):
        width = stages[tag][0]
        count_reg = stage_count_reg(num_full_sbuf, stages[tag][1], stages[tag][2], tag)

        # width default-bound, not captured: later stages rebind it.
        def body(_i, width=width):
            for slot in range(1, width):
                nisa.tensor_scalar(
                    dst=row_start[slot],
                    data=m_tile_start,
                    op0=nl.add,
                    operand0=slot * _TILE_M,
                )
            emit_trip(
                width,
                row_start,
                inputs,
                result,
                out_dtype,
                hidden,
                h_starts,
                h_widths,
                compute,
                aux,
                column,
            )
            nisa.tensor_scalar(
                dst=m_tile_start, data=m_tile_start, op0=nl.add, operand0=width * _TILE_M
            )

        nl.fori_loop(0, count_reg, body)

    emit_trip(
        1,
        [tail_row],
        inputs,
        result,
        out_dtype,
        hidden,
        h_starts,
        h_widths,
        compute,
        aux,
        column,
    )
