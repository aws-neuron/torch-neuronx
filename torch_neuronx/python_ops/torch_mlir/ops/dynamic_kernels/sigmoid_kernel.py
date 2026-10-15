"""Sigmoid Activation Kernel: output = sigmoid(input) = 1 / (1 + exp(-input))

Dynamic m-dim with single block size = 128.
Uses nisa.activation with op=nl.sigmoid.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Max row tiles folded into one trip of the dynamic M loop.
_UNROLL = 8

# Per-partition SBUF ceiling for one trip's live tiles
_TRIP_SBUF_BUDGET = 128 * 1024


def _run_sigmoid_loop(
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
    """One nl.fori_loop that sigmoids `tiles_per_trip` consecutive 128-row tiles per trip.

    Slot 0 addresses rows straight off the carried `m_tile_start`, slot k>0 off its own
    `row_start[k]` = cursor + k*128, so the slots stay independent of each other.

    The body is batched by phase -- every load, then every activation, then every store.
    Ordering it per slot instead makes the slots' live ranges disjoint, so the SBUF
    allocator folds them all onto one buffer pair and each load waits on the previous
    activation: measured 2-deep pipelining, one DMA queue busy 15.6us of a 45.3us trip.
    """

    def body(_i):
        """Apply sigmoid to one unrolled group of row tiles.

        Args:
            _i (int): Dynamic loop iteration index, accepted but unused.
        """
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

        # Phase 2: sigmoid every tile.
        out_tiles = []
        for slot in range(tiles_per_trip):
            seg_tiles = []
            for seg in range(len(h_starts)):
                h_width = h_widths[seg]
                out_tile = nl.ndarray((tile_size_m, h_width), input_tensor.dtype, nl.sbuf)
                nisa.activation(
                    dst=out_tile[0:tile_size_m, 0:h_width],
                    data=in_tiles[slot][seg][0:tile_size_m, 0:h_width],
                    op=nl.sigmoid,
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
def sigmoid_kernel(input_tensor, num_m_tiles):
    """Apply sigmoid over dynamic row tiles.

    Args:
        input_tensor (nki.language.ndarray): Input with shape ``[m_static, hidden]``.
        num_m_tiles (nki.language.ndarray): Number of 128-row tiles to process.

    Returns:
        nki.language.ndarray: Static-shaped result in the input dtype.

    Note:
        Only the first ``128 * num_m_tiles`` rows contain sigmoid values; remaining
        rows are not initialized.
    """
    m_static, hidden = input_tensor.shape

    tile_size_m = 128

    # Give each SPMD program a contiguous hidden/num_programs slice
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    if hidden % num_progs == 0:
        h_span = hidden // num_progs
        h_base = prog_id * h_span
    else:
        h_span = hidden
        h_base = 0

    _dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(input_tensor.dtype), 2)
    tile_size_h = min(h_span, 8192 // _dtype_bytes)
    h_starts, h_widths = [], []
    _off = 0
    while _off < h_span:
        _w = min(tile_size_h, h_span - _off)
        h_starts.append(h_base + _off)
        h_widths.append(_w)
        _off += _w

    # Largest power-of-two unroll whose live tiles fit the budget.
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
        _run_sigmoid_loop(
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
