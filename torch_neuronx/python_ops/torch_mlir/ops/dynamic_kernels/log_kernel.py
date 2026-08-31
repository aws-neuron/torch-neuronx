"""Log Activation Kernel: output = log(input)

Dynamic m-dim with single block size = 128.
Uses nisa.activation with op=nl.log.

Structurally identical to the exp kernel (unary activation, unsharded on M): the M
loop is unrolled with a device-side multi-stage trip-count split, the body is
batched by phase (all loads, then all activations, then all stores), the hidden
dim is SPMD-split across LNC programs, and the indirect-offset DMAs use hardware
DGE. See optimization_tips #1, #5, #13, #14, #15.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Max row tiles folded into one trip of the dynamic M loop. Every trip pays a fixed
# ~7us back-edge fence -- the next trip's DMA addresses come off the loop-carried row
# cursor, so all in-flight DMAs must land before it advances -- and that cost does not
# scale with the data a trip moves, so fatter trips is the only lever. Must stay a
# power of two: the trip-count split below uses shift/AND. Held at 8: raising to 16
# regressed the sweep (the extra 16->8->4->1 trip stage costs a ~7us fence the low-M
# shapes can't fill, and the low-H shapes are DMA-starved from the H-split, not from
# too few tiles per trip -- more unroll doesn't help them).
_UNROLL = 8

# Per-partition SBUF ceiling for one trip's live tiles (trn2 has 192 KB). A trip holds
# an input and an output tile per (slot, h-segment) pair at once, so large H drops the
# unroll factor instead of overflowing SBUF.
_TRIP_SBUF_BUDGET = 128 * 1024


def _run_log_loop(
    count_reg,
    tiles_per_trip,
    input_tensor,
    result,
    hidden,
    tile_size_m,
    h_starts,
    h_widths,
    m_tile_start,
    row_start,
):
    """One nl.fori_loop that logs `tiles_per_trip` consecutive 128-row tiles per trip.

    Slot 0 addresses rows straight off the carried `m_tile_start`, slot k>0 off its own
    `row_start[k]` = cursor + k*128, so the slots stay independent of each other.

    The body is batched by phase -- every load, then every activation, then every store.
    """

    def body(_i):
        for slot in range(1, tiles_per_trip):
            nisa.tensor_scalar(
                dst=row_start[slot],
                data=m_tile_start,
                op0=nl.add,
                operand0=slot * tile_size_m,
            )

        # Phase 1: load every (slot, h-segment) tile.
        in_tiles = []
        for slot in range(tiles_per_trip):
            seg_tiles = []
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                in_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
                nisa.dma_copy(
                    dst=in_tile[0:tile_size_m, 0:h_width],
                    src=input_tensor.ap(
                        pattern=[[hidden, tile_size_m], [1, h_width]],
                        offset=h_starts[seg],
                        scalar_offset=row_start[slot],
                        indirect_dim=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )
                seg_tiles.append(in_tile)
            in_tiles.append(seg_tiles)

        # Phase 2: log every tile.
        out_tiles = []
        for slot in range(tiles_per_trip):
            seg_tiles = []
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                out_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
                nisa.activation(
                    dst=out_tile[0:tile_size_m, 0:h_width],
                    data=in_tiles[slot][seg][0:tile_size_m, 0:h_width],
                    op=nl.log,
                )
                seg_tiles.append(out_tile)
            out_tiles.append(seg_tiles)

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
            operand0=tiles_per_trip * tile_size_m,
        )

    nl.fori_loop(0, count_reg, body)


@nki.jit()
def log_kernel(input_tensor, num_m_tiles):
    """
    Log activation: output = log(input)

    Args:
        input_tensor: [m_static, hidden] bf16 on HBM
        num_m_tiles: [1, 1] int32 — number of 128-row tiles to process
    Returns:
        output: [m_static, hidden] bf16 on HBM
    """
    m_static, hidden = input_tensor.shape

    tile_size_m = 128

    # Give each SPMD program a contiguous hidden/num_programs slice: the launch grid is
    # logical_neuron_cores wide, so without this every core reads and writes the whole
    # tensor -- 2x the HBM traffic at LNC=2. Equal widths keep the per-core basic-block
    # counts identical as NCC_IXGM002 requires. Only the free dim can be split this way;
    # M's trip count is a runtime value, so splitting it would desymmetrize those counts.
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    if hidden % num_progs == 0:
        h_span = hidden // num_progs
        h_base = prog_id * h_span
    else:
        h_span = hidden
        h_base = 0

    # ~8 KB/partition per tile, the DMA-throughput saturation knee. H is a compile-time
    # constant, so the tiling resolves in Python at trace time: full-width tiles plus a
    # final short one, i.e. arbitrary H with no padding (M stays runtime-dynamic).
    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_tensor.dtype), 2)
    tile_size_h = min(h_span, 8192 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < h_span:
        _w = min(tile_size_h, h_span - _off)
        h_starts.append(h_base + _off)
        h_widths.append(_w)
        _off += _w

    # Largest power-of-two unroll whose live tiles fit the budget. Compile-time only, so
    # both SPMD programs pick the same factor and emit the same basic blocks.
    _trip_bytes = 2 * len(h_starts) * tile_size_h * _dtype_bytes
    unroll = 1
    log2_unroll = 0
    while unroll * 2 <= _UNROLL and 2 * unroll * _trip_bytes <= _TRIP_SBUF_BUDGET:
        unroll *= 2
        log2_unroll += 1

    result = nl.ndarray((m_static, hidden), input_tensor.dtype, nl.shared_hbm)

    num_m_tiles_sbuf = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=num_m_tiles_sbuf, src=num_m_tiles[0])

    m_tile_start = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
    nisa.memset(m_tile_start, 0.0)

    # Slot 0 reads the carried cursor directly; slot k>0 needs its own scalar.
    row_start = [m_tile_start]
    for k in range(1, unroll):
        row_start.append(nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf, name=f"row_start_{k}"))

    # Trip-count split, on device because num_m_tiles is a runtime value: stage 0 runs
    # n>>log2(unroll) full-width trips, each later wide stage runs 0 or 1 trips (one bit
    # of the leftover), a final width-1 stage mops up, and every stage shares the carried
    # cursor so each resumes where the last stopped. Each stage is another fori_loop, and
    # loop entry costs a fence (~7.6us) even at zero trips, so a stage of width w pays
    # off only while (w-1)/2 > 1 -- hence stages stop at 4 instead of continuing to 2.
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
        _run_log_loop(
            count_reg,
            widths[stage],
            input_tensor,
            result,
            hidden,
            tile_size_m,
            h_starts,
            h_widths,
            m_tile_start,
            row_start,
        )

    return result
