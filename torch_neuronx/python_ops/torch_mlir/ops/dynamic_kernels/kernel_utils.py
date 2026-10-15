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
    """Split a hidden dimension into this program's compile-time tiles.

    Args:
        hidden (int): Width of each input row.
        dtype (object): NKI element dtype used to determine the byte-limited tile width.

    Returns:
        tuple: Lists of tile start offsets and corresponding widths.

    Note:
        The free dimension, rather than rows, is split so runtime row counts remain unpadded;
        unequal per-program trip counts can hang the Neuron device.
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
    """Choose a power-of-two unroll factor using the SBUF budget.

    Args:
        h_widths (Sequence): Compile-time widths of the hidden-dimension segments.
        live_bytes_per_element (int): Bytes held per element for operands, output, and scratch.

    Returns:
        tuple: Selected unroll factor and its base-two logarithm.

    Note:
        Only compile-time values participate, so every program selects the same factor. The
        base factor remains one even when a single slot exceeds the trip budget.
    """
    per_slot = len(h_widths) * max(h_widths) * live_bytes_per_element
    unroll, log2_unroll = 1, 0
    while unroll * 2 <= _UNROLL_MAX and (unroll * 2) * per_slot <= _TRIP_SBUF_BUDGET:
        unroll *= 2
        log2_unroll += 1
    return unroll, log2_unroll


def trip_stages(unroll, log2_unroll):
    """Build the staged loop widths, shifts, and masks for an unroll factor.

    Args:
        unroll (int): Power-of-two maximum number of row tiles per trip.
        log2_unroll (int): Base-two logarithm of ``unroll``.

    Returns:
        list: ``(width, shift, mask)`` tuples for full trips, leftover bits, and mop-up.

    Note:
        Each stage pays a loop-entry fence even for zero trips, so the cascade stops at width
        four before a width-one mop-up.
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
    """Load runtime row-loop controls and initialize the row cursor.

    Args:
        ctrl (nl.NkiTensor): ``[2, 1]`` int32 tensor containing the full-tile count and tail row.

    Returns:
        tuple: SBUF tensors for the full-tile count, tail row, and zeroed row cursor.

    Note:
        Keeping these values runtime-driven prevents the row count from being baked into the NEFF.
    """
    num_full_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_full_sbuf, src=ctrl[0])

    tail_row = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tail_row, src=ctrl[1])

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)
    return num_full_sbuf, tail_row, m_tile_start


def row_cursors(m_tile_start, unroll):
    """Allocate one runtime row cursor for each unrolled slot.

    Args:
        m_tile_start (nl.NkiTensor): Cursor for the first row-tile slot.
        unroll (int): Number of cursors to return.

    Returns:
        list: The original cursor followed by allocated cursors for later slots.

    Note:
        Later slots require runtime scalars because a traced compile-time ``offset`` cannot
        leave its extent.
    """
    cursors = [m_tile_start]
    for k in range(1, unroll):
        cursors.append(nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"row_start_{k}"))
    return cursors


def stage_count_reg(num_full_sbuf, shift, mask, tag):
    """Load a shifted and masked stage trip count into a register.

    Args:
        num_full_sbuf (nl.NkiTensor): Runtime count of full row tiles.
        shift (int): Right-shift applied before loading the count.
        mask (int): Bit mask applied after shifting.
        tag (object): Identifier used to name intermediate tensors.

    Returns:
        nisa.VirtualRegister: Register containing the stage trip count.
    """
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
    """Emit loads, computation, and stores for consecutive row tiles.

    Args:
        width (int): Number of 128-row slots in the trip.
        row_start (Sequence): Runtime row cursor for each slot.
        inputs (Sequence): Full-width source tensors.
        result (nl.NkiTensor): Destination tensor in shared HBM modified in place.
        out_dtype (object): NKI dtype for destination tiles.
        hidden (int): Full hidden width and row stride.
        h_starts (Sequence): Hidden-segment start offsets.
        h_widths (Sequence): Hidden-segment widths.
        compute (Callable): Module-level function that fills one destination tile.
        aux (object, optional): Auxiliary value passed to ``compute``. Defaults to ``None``.
        column (nl.NkiTensor, optional): Column operand broadcast over the free dimension.
            Defaults to ``None``.

    Note:
        All slots load before any compute or store so buffers remain independently
        pipelineable. Plain loops avoid NKI parser rejection of comprehensions and starred
        unpacking.
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
    """Process every row for a runtime-sized leading dimension.

    Args:
        ctrl (nl.NkiTensor): Runtime full-tile count and tail-row control tensor.
        inputs (Sequence): Full-width source tensors.
        result (nl.NkiTensor): Destination tensor in shared HBM modified in place.
        out_dtype (object): NKI dtype for destination tiles.
        hidden (int): Full hidden width and row stride.
        h_starts (Sequence): Hidden-segment start offsets.
        h_widths (Sequence): Hidden-segment widths.
        compute (Callable): Module-level function that fills one destination tile.
        live_bytes_per_element (int): Per-element live storage used to choose unroll.
        aux (object, optional): Auxiliary value passed to ``compute``. Defaults to ``None``.
        column (nl.NkiTensor, optional): Column operand broadcast over the free dimension.
            Defaults to ``None``.

    Note:
        The unconditional tail starts at ``M - 128``; exact multiples idempotently rewrite
        the last tile. Every access stays within ``[0, M)``, so no padding is required and
        ``out=`` can back ``result``.
    """
    unroll, log2_unroll = unroll_factor(h_widths, live_bytes_per_element)
    num_full_sbuf, tail_row, m_tile_start = loop_control(ctrl)
    row_start = row_cursors(m_tile_start, unroll)

    stages = trip_stages(unroll, log2_unroll)
    for tag in range(len(stages)):
        width = stages[tag][0]
        count_reg = stage_count_reg(num_full_sbuf, stages[tag][1], stages[tag][2], tag)

        # width default-bound, not captured: later stages rebind it.
        def body(_i, width=width):
            """Emit one staged trip and advance the shared row cursor.

            Args:
                _i (int): Loop iteration index supplied by ``nl.fori_loop``; unused.
                width (int, optional): Number of row-tile slots emitted by this stage.
                    Defaults to the current stage width.
            """
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
