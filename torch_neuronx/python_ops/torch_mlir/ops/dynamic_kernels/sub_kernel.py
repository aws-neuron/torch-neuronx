"""
Elementwise Sub Kernel: output = input_a - input_b

Two separate inputs, dynamic m-dim with single block size = 128.

The M loop is unrolled and the body is batched by phase (all loads, then all
subtracts, then all stores) so the DMA engine stays busy across a trip instead of
stalling on each tile's subtract. Each trip pays a fixed ~7us back-edge fence that
does not scale with the data it moves, so folding more row tiles into one trip
is the only lever against that overhead.

M-dim LNC sharding is owned by the adapter (elementwise.py): the launch grid is
`logical_neuron_cores` wide and each program processes an interleaved set of
128-row tiles via `program_row_offset` + a stride of `tile_size_m * num_programs`.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Max row tiles folded into one trip of the dynamic M loop. Power of two: the
# trip-count split below uses shift/AND.
_UNROLL = 8

# Per-partition SBUF ceiling for one trip's live tiles (trn2 has 192 KB). A trip
# holds a_tile + b_tile + out_tile per (slot, h-segment) at once, so large H drops
# the unroll factor instead of overflowing SBUF. Held at 128 KB: raising it to push
# H<=4096 shapes to unroll=8 regressed the analogous add sweep (the extra
# width-8->4->1 trip stages each cost a ~7us fence that the low-M shapes with few
# m-tiles can't amortize).
_TRIP_SBUF_BUDGET = 128 * 1024


def _run_sub_loop(
    count_reg,
    tiles_per_trip,
    input_a,
    input_b,
    result,
    hidden,
    tile_size_m,
    num_programs,
    h_starts,
    h_widths,
    m_tile_start,
    row_start,
):
    """One nl.fori_loop that subtracts `tiles_per_trip` interleaved 128-row tiles per trip.

    Slot 0 addresses rows straight off the carried `m_tile_start`; slot k>0 off its
    own `row_start[k]` = cursor + k*(tile_size_m*num_programs), so the slots stay
    independent and honor the adapter's row-interleaved LNC sharding.

    The body is batched by phase -- every load, then every subtract, then every store --
    so the DMA engine can burst back-to-back loads/stores while the Vector engine
    runs the subtracts, instead of load-sub-store per slot serializing the pipeline.
    """
    slot_stride = tile_size_m * num_programs

    def body(_i):
        for slot in range(1, tiles_per_trip):
            nisa.tensor_scalar(
                dst=row_start[slot],
                data=m_tile_start,
                op0=nl.add,
                operand0=slot * slot_stride,
            )

        # Phase 1: load every (slot, h-segment) a_tile and b_tile.
        a_tiles = []
        b_tiles = []
        for slot in range(tiles_per_trip):
            a_seg = []
            b_seg = []
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                a_tile = nl.ndarray((tile_size_m, h_width), input_a.dtype, nl.sbuf)
                nisa.dma_copy(
                    dst=a_tile[0:tile_size_m, 0:h_width],
                    src=input_a.ap(
                        pattern=[[hidden, tile_size_m], [1, h_width]],
                        offset=h_starts[seg],
                        scalar_offset=row_start[slot],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )
                b_tile = nl.ndarray((tile_size_m, h_width), input_b.dtype, nl.sbuf)
                nisa.dma_copy(
                    dst=b_tile[0:tile_size_m, 0:h_width],
                    src=input_b.ap(
                        pattern=[[hidden, tile_size_m], [1, h_width]],
                        offset=h_starts[seg],
                        scalar_offset=row_start[slot],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )
                a_seg.append(a_tile)
                b_seg.append(b_tile)
            a_tiles.append(a_seg)
            b_tiles.append(b_seg)

        # Phase 2: subtract every tile.
        out_tiles = []
        for slot in range(tiles_per_trip):
            o_seg = []
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                out_tile = nl.ndarray((tile_size_m, h_width), input_a.dtype, nl.sbuf)
                nisa.tensor_tensor(
                    dst=out_tile[0:tile_size_m, 0:h_width],
                    data1=a_tiles[slot][seg][0:tile_size_m, 0:h_width],
                    data2=b_tiles[slot][seg][0:tile_size_m, 0:h_width],
                    op=nl.subtract,
                )
                o_seg.append(out_tile)
            out_tiles.append(o_seg)

        # Phase 3: store every tile.
        for slot in range(tiles_per_trip):
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                nisa.dma_copy(
                    dst=result.ap(
                        pattern=[[hidden, tile_size_m], [1, h_width]],
                        offset=h_starts[seg],
                        scalar_offset=row_start[slot],
                        indirect_dim=0,
                    ),
                    src=out_tiles[slot][seg][0:tile_size_m, 0:h_width],
                    dge_mode=nisa.dge_mode.hwdge,
                )

        nisa.tensor_scalar(
            dst=m_tile_start,
            data=m_tile_start,
            op0=nl.add,
            operand0=tiles_per_trip * slot_stride,
        )

    nl.fori_loop(0, count_reg, body)


@nki.jit()
def sub_kernel(input_a, input_b, num_m_tiles):
    """
    Elementwise sub: output = input_a - input_b

    Args:
        input_a: [m_static, hidden] bf16 on HBM
        input_b: [m_static, hidden] bf16 on HBM
        num_m_tiles: [1, 1] int32, number of 128-row tiles per program
    Returns:
        output: [m_static, hidden] bf16 on HBM
    """
    m_static, hidden = input_a.shape

    tile_size_m = 128
    # Programs process interleaved row tiles; the adapter equalizes loop counts.
    num_programs = nl.num_programs(axes=0)
    program_id = nl.program_id(axis=0)
    program_row_offset = program_id * tile_size_m * hidden
    # Free-dim tile width ~4 KB/partition (DMA saturation knee): bf16/fp16 -> 2048,
    # fp32 -> 1024. H compile-time constant -> resolved at trace time; tile_size_h-
    # wide tiles plus a final short tile of `hidden % tile_size_h` — arbitrary H.
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_a.dtype), 2)
    tile_size_h = min(hidden, 4096 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < hidden:
        _w = min(tile_size_h, hidden - _off)
        h_starts.append(program_row_offset + _off)
        h_widths.append(_w)
        _off += _w

    # Largest power-of-two unroll whose live tiles fit the budget. One trip holds
    # a_tile + b_tile + out_tile per (slot, h-segment): 3 tiles per slot-segment.
    # Compile-time only, so every program picks the same factor and emits the same
    # basic blocks (NCC symmetry).
    _trip_bytes = 3 * len(h_starts) * tile_size_h * _dtype_bytes
    unroll = 1
    log2_unroll = 0
    while unroll * 2 <= _UNROLL and 2 * unroll * _trip_bytes <= _TRIP_SBUF_BUDGET:
        unroll *= 2
        log2_unroll += 1

    result = nl.ndarray((m_static, hidden), input_a.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    # Slot 0 reads the carried cursor directly; slot k>0 needs its own scalar.
    row_start = [m_tile_start]
    for k in range(1, unroll):
        row_start.append(nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"row_start_{k}"))

    # Trip-count split, on device because num_m_tiles is a runtime value: stage 0
    # runs n>>log2(unroll) full-width trips, each later wide stage runs 0 or 1 trips
    # (one bit of the leftover), a final width-1 stage mops up, and every stage shares
    # the carried cursor so each resumes where the last stopped. Each stage is another
    # fori_loop, and loop entry costs a fence (~7us) even at zero trips, so a stage of
    # width w pays off only while (w-1)/2 > 1 -- hence stages stop at 4 instead of 2.
    widths, shifts, masks = [], [], []
    if unroll <= 4:
        widths.append(unroll)
        shifts.append(log2_unroll)
        masks.append(0)
        if unroll > 1:
            widths.append(1)
            shifts.append(0)
            masks.append(unroll - 1)
    else:
        _w, _s = unroll, log2_unroll
        while _w >= 4:
            widths.append(_w)
            shifts.append(_s)
            masks.append(0 if _w == unroll else 1)
            _w //= 2
            _s -= 1
        widths.append(1)
        shifts.append(0)
        # The width-4 stage above consumed bit 2, so bits 0-1 are what is left.
        masks.append(3)

    for stage in range(len(widths)):
        count_sbuf = num_m_tiles_sbuf
        if shifts[stage] > 0:
            shifted = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"n_shr{stage}")
            nisa.tensor_scalar(
                dst=shifted, data=count_sbuf, op0=nl.right_shift, operand0=shifts[stage]
            )
            count_sbuf = shifted
        if masks[stage] > 0:
            masked = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"n_msk{stage}")
            nisa.tensor_scalar(
                dst=masked, data=count_sbuf, op0=nl.bitwise_and, operand0=masks[stage]
            )
            count_sbuf = masked

        count_reg = nisa.register_alloc()
        nisa.register_load(count_reg, count_sbuf)
        _run_sub_loop(
            count_reg,
            widths[stage],
            input_a,
            input_b,
            result,
            hidden,
            tile_size_m,
            num_programs,
            h_starts,
            h_widths,
            m_tile_start,
            row_start,
        )

    return result
