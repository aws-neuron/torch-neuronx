"""Dynamic multi-tier AllGather NKI kernel + StableHLO lowering for the C++ NeuronBackend.

The C++ dispatch reshapes the per-rank shard to a flat [numel, 1] and gathers it into
[world*numel, 1]. The kernel is shape-agnostic: the NEFF is traced at a fixed floor
trace_numel and a runtime int32 scalar drives the on-device tile counts.

"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from torch_neuronx.nki_hop import wrap_nki

_KiB, _MiB = 1024, 1024 * 1024
# Tiling tables keyed by (profile, group-size bucket); platforms map to a profile via
# _profile_for_platform (TRN2/TRN3 share "default"). Each table is a list of regimes
# (name, floor_bytes, (large, medium, small) byte targets).
# pick_dynamic_tiling picks the first regime with floor <= gathered
# bytes, then _tile_for converts the byte targets to pow2 element tiles.

# Default profile, small replica groups (size <= _GROUP_SMALL_MAX).
_DEFAULT_SMALL = [
    ("A", 32 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("B", 16 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("C", 8 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("D", 2 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("E", 512 * _KiB, (1 * _MiB, 512 * _KiB, 256 * _KiB)),
    ("F", 0, (256 * _KiB, 128 * _KiB, 64 * _KiB)),
]
# Default profile, large replica groups (size > _GROUP_SMALL_MAX).
_DEFAULT_LARGE = [
    ("A", 64 * _MiB, (64 * _MiB, 32 * _MiB, 16 * _MiB)),
    ("B", 32 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("C", 16 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("D", 4 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("E", 1 * _MiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("F", 0, (512 * _KiB, 256 * _KiB, 128 * _KiB)),
]
_TILING_TABLES = {
    ("default", "small"): _DEFAULT_SMALL,
    ("default", "large"): _DEFAULT_LARGE,
}
# Replica groups with size <= this use the "small" table; larger use "large".
_GROUP_SMALL_MAX = 8


def _profile_for_platform(platform):
    """Map a Neuron platform target (e.g. 'trn2'/'trn3') to a tiling profile. This is the ONLY
    place platform identity is interpreted; everything downstream keys on the profile. TRN2 and
    TRN3 share the 'default' profile today; any other/unknown platform also falls back to
    'default'. Add a branch here + _<PROFILE>_* tables when a platform needs its own tiling."""
    if platform in ("trn2", "trn3"):
        return "default"
    return "default"  # fallback: unknown platforms use the default profile


def _resolve_table(platform, group_size):
    """Pick the tiling table for a (platform, replica-group size), via (profile, group bucket)."""
    bucket = "small" if group_size <= _GROUP_SMALL_MAX else "large"
    return _TILING_TABLES[(_profile_for_platform(platform), bucket)]


def _pow2_floor(x):
    x = int(x)
    return 1 << (x.bit_length() - 1) if x >= 1 else 1


def _log2_const(x):
    """Trace-time log2 of a power-of-two tile size, via a plain integer loop
    so that NKI tracer folds it as a constant."""
    k = 0
    while x > 1:
        assert (x & 1) == 0, f"tile size must be a power of 2, got non-pow2 {x}"
        x >>= 1
        k += 1
    return k


def _tile_for(target_bytes, num_ranks, n_prgs, dtype_size=2):
    """Elements whose gathered size ~= target_bytes, as a power of two (>= n_prgs), so
    the kernel derives tile counts on-device with hardware-safe shift/AND."""
    bytes_per_elem = dtype_size * num_ranks  # gathered bytes per element (all ranks)
    r = max(1, round(target_bytes / bytes_per_elem))
    return max(int(n_prgs), _pow2_floor(r))


def regime_tiles(num_ranks, n_prgs, targets, dtype_size=2):
    """Convert a regime's three gathered-byte targets into three non-increasing pow2 element
    tiles (large >= medium >= small)"""
    large_t, medium_t, small_t = targets
    large = _tile_for(large_t, num_ranks, n_prgs, dtype_size)
    medium = min(large, _tile_for(medium_t, num_ranks, n_prgs, dtype_size))
    small = min(medium, _tile_for(small_t, num_ranks, n_prgs, dtype_size))
    return large, medium, small


def pick_dynamic_tiling(numel, num_ranks, n_prgs, dtype_size=2, platform="trn2"):
    """Select (large, medium, small, regime_name) element tiles for this shard by walking the
    (platform, group-size) tiling table: the first regime whose floor <= gathered bytes."""
    gathered = numel * dtype_size * num_ranks
    table = _resolve_table(platform, num_ranks)
    for name, floor, targets in table:
        if gathered >= floor:
            large, medium, small = regime_tiles(num_ranks, n_prgs, targets, dtype_size)
            return large, medium, small, name
    raise RuntimeError(f"no regime matched gathered={gathered}")  # unreachable: last floor == 0


def _pick_unroll(extent, trace):
    """Unroll factor in {1,2,4}: the deepest pow2 that fits the primary tile count"""
    primary = extent // trace if trace > 0 else 1
    if primary >= 4:
        return 4
    if primary >= 2:
        return 2
    return 1


def plan_dynamic_allgather(numel, world, n_prgs, dtype_size=2, platform="trn2"):
    """Pick the tile sizes and trace floor for gathering a per-rank shard of numel
    elements. Returns None (caller falls back to the native collective) when numel is
    too small to fill even one small tile, or world % n_prgs != 0``

    The tiling table is selected by (platform, replica-group size = world); see the module
    tables. Returns dict(large, medium, small, regime, trace_numel, unroll):
      * large, medium, small : the three non-increasing pow2 element tiles for this shard,
        from the selected table's matched regime. The kernel decomposes numel greedily
        large -> medium -> small, then a single small-tile tail.
      * regime       : the matched regime name (for logging only).
      * trace_numel  : the tile the NEFF is traced at, and the identity of this shard's
        dispatch bucket.
      * unroll       : tiles processed per dynamic-loop iteration.
    """
    if numel <= 0 or world % n_prgs != 0:
        return None
    large, medium, small, regime = pick_dynamic_tiling(numel, world, n_prgs, dtype_size, platform)
    if large <= numel:
        trace_numel = large
    elif medium <= numel:
        trace_numel = medium
    elif small <= numel:
        trace_numel = small
    else:
        return None
    return {
        "large": large,
        "medium": medium,
        "small": small,
        "regime": regime,
        "trace_numel": trace_numel,
        "unroll": _pick_unroll(numel, trace_numel),
    }


def _run_gather_loop(
    count_reg: nisa.VirtualRegister,
    tiles_per_trip: int,
    in_bufs: list[nl.NkiTensor],
    out_bufs: list[nl.NkiTensor],
    src_load: nl.NkiTensor,
    dst_off: list[nl.NkiTensor],
    off_in_t: list[nl.NkiTensor | None],
    off_out_t: list[nl.NkiTensor | None],
    inc: list[nl.NkiTensor | None],
    tile: int,
    half: int,
    in_start: int,
    rank_start: int,
    ranks_per_prg: int,
    num_ranks: int,
    flat_in: nl.NkiTensor,
    out_flat: nl.NkiTensor,
    replica_group: ReplicaGroup,
) -> None:
    """One nl.fori_loop that gathers `tiles_per_trip` tiles per trip on the given buffers.

    Shared by the bulk loop (tiles_per_trip = tier_unroll, independent hbm_in/out slots) and
    the unroll-remainder loop (tiles_per_trip = 1, dedicated single-slot buffers). Each trip:
    bounce this program's `half`-row slice of each tile into `in_bufs[j]`, ncc.all_gather it,
    then scatter every owned rank's tile into out_flat; finally advance the carried offsets.

    All per-tier state is passed in (mutable objects mutated in place: src_load, dst_off,
    off_in_t/off_out_t), so the body closure captures no enclosing loop variable.
    """

    def body(_i):
        for j in range(tiles_per_trip):
            if j == 0:
                off_in = src_load
            else:
                nisa.tensor_tensor(dst=off_in_t[j], data1=src_load, data2=inc[j], op=nl.add)
                off_in = off_in_t[j]
            nisa.dma_copy(
                dst=in_bufs[j][in_start : in_start + half, :],
                src=flat_in.ap(pattern=[[1, half], [1, 1]], scalar_offset=off_in, indirect_dim=0),
            )
            ncc.all_gather(
                dsts=[out_bufs[j]],
                srcs=[in_bufs[j]],
                replica_group=replica_group,
                collective_dim=0,
            )
            for r in range(rank_start, rank_start + ranks_per_prg):
                if j == 0:
                    off_out = dst_off[r]
                else:
                    nisa.tensor_tensor(dst=off_out_t[j], data1=dst_off[r], data2=inc[j], op=nl.add)
                    off_out = off_out_t[j]
                nisa.dma_copy(
                    dst=out_flat.ap(
                        pattern=[[1, tile], [1, 1]], scalar_offset=off_out, indirect_dim=0
                    ),
                    src=out_bufs[j][r * tile : (r + 1) * tile, :],
                )
        # Advance carried offsets by the tiles consumed this trip.
        step = inc[tiles_per_trip]
        nisa.tensor_tensor(dst=src_load, data1=src_load, data2=step, op=nl.add)
        for r in range(num_ranks):
            nisa.tensor_tensor(dst=dst_off[r], data1=dst_off[r], data2=step, op=nl.add)

    nl.fori_loop(0, count_reg, body)


@nki.jit
def dynamic_allgather_kernel(
    input_tensor: nl.NkiTensor,
    numel_tensor: nl.NkiTensor,
    replica_group: ReplicaGroup,
    num_ranks: int,
    large_tile: int,
    medium_tile: int,
    small_tile: int,
    unroll: int,
) -> nl.NkiTensor:
    """Dynamic all-gather kernel.

    The input_tensor is flattened to [numel, 1]. At runtime, numel_tensor drives the
    large, medium, small, and tail tile counts. The large/medium/small tiers process unroll
    tiles per iteration on independent buffers so the scheduler can overlap one tile's
    all-gather with the next tile's load.

    Dimensions:
      numel   per-rank element count (runtime); max_numel is the traced floor.
      tile    large_tile or small_tile elements, split half=tile/n_prgs per program.
      output  num_ranks * numel elements; rank r occupies [r*numel : (r+1)*numel).

    Args:
      input_tensor   Per-rank shard, bound flat as [max_numel, 1] (max_numel = trace floor).
      numel_tensor   Runtime [1, 1] int32 scalar = per-rank numel; drives on-device counts.
      replica_group  Ranks participating in the all-gather.
      num_ranks      World size for this group (compile-time constant).
      large_tile     Large-tier tile size in elements (power of two, compile-time constant).
      medium_tile    Medium-tier tile size in elements (power of two, <= large_tile).
      small_tile     Small/tail-tier tile size in elements (power of two, <= medium_tile).
      unroll         Tiles per large/medium/small dynamic-loop iteration; pow2 in {1, 2, 4}.

    Returns:
      out_flat  Gathered [num_ranks * max_numel, 1] shared-HBM tensor.

    Notes:
      - Multi-core (LNC>=2): ranks are split across programs (ranks_per_prg = num_ranks /
        n_prgs) and each tile's input rows are split by half; requires num_ranks % n_prgs == 0.
      - The tail tile starts at numel - small_tile, so when numel mod small_tile != 0 it
        re-gathers a few already-gathered elements. The double-write is idempotent (same
        source maps to the same output element), so the result is correct.
      - unroll splits a tier's tile count as bulk = count >> log2(unroll) plus an
        unroll-remainder loop of count & (unroll-1) single tiles on dedicated buffers. This
        unroll remainder is distinct from the tail tier. The tail tier is always a single
        tile, so it never unrolls.
      - The shard is reshaped outside the NEFF, so the kernel is shape-agnostic: one NEFF
        traced at [trace_numel, 1] serves any larger numel in the same bucket.

    Example:
      world=4, lnc=2 (n_prgs=2), large_tile=131072, medium_tile=65536, small_tile=32768,
      unroll=2, runtime numel=496520. On device: n_large=3, n_medium=1, n_small=1, tail=1.
      The large tier runs bulk=1 (2 tiles) then an unroll-remainder loop of 1 tile; the
      medium and small tiers each run one tile; the tail re-gathers the last 32768 elements.
      Output is [4*496520, 1] with rank r's shard at [r*496520 : (r+1)*496520).
    """
    max_numel, _one = input_tensor.shape  # bound flat as [numel, 1]
    dtype = input_tensor.dtype
    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)
    ranks_per_prg = num_ranks // n_prgs
    rank_start = prg_id * ranks_per_prg

    flat_in = input_tensor  # already [max_numel, 1]
    out_flat = nl.ndarray(
        (num_ranks * max_numel, 1), dtype=dtype, buffer=nl.shared_hbm, name="output"
    )

    mask_large = large_tile - 1
    mask_medium = medium_tile - 1
    mask_small = small_tile - 1
    log2_large = _log2_const(large_tile)
    log2_medium = _log2_const(medium_tile)
    log2_small = _log2_const(small_tile)
    log2_unroll = _log2_const(unroll)
    n_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sbuf, src=numel_tensor)

    # TODO: replace tensor ops below with sequencer register ALU ops to avoid the high
    # instruction overhead.
    r0_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r0_s, data=n_sbuf, op0=nl.bitwise_and, operand0=mask_large)
    n_large_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=n_large_s, data=n_sbuf, op0=nl.right_shift, operand0=log2_large)

    r1_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r1_s, data=r0_s, op0=nl.bitwise_and, operand0=mask_medium)
    n_medium_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=n_medium_s, data=r0_s, op0=nl.right_shift, operand0=log2_medium)

    r2_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r2_s, data=r1_s, op0=nl.bitwise_and, operand0=mask_small)
    n_small_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=n_small_s, data=r1_s, op0=nl.right_shift, operand0=log2_small)

    # n_tail_s = ceil(r2_s / small_tile). The "tail" tier gathers numel mod small_tile
    # (the element tail below one small tile); this is 0 or 1 tile. Distinct from the
    # "unroll remainder" (tile_count mod unroll) handled inside each tier's loop below.
    mask_small_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(mask_small_s, value=mask_small)
    n_tail_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=n_tail_s, data1=r2_s, data2=mask_small_s, op=nl.add)
    nisa.tensor_scalar(dst=n_tail_s, data=n_tail_s, op0=nl.right_shift, operand0=log2_small)

    # Tier start offsets use exact int32 tensor_tensor arithmetic: medium starts after the large
    # tiles (numel - r0); small starts after the medium tiles (numel - r1); the tail re-reads the
    # final small tile (numel - small_tile).
    start_medium_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_medium_s, data1=n_sbuf, data2=r0_s, op=nl.subtract)
    start_small_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_small_s, data1=n_sbuf, data2=r1_s, op=nl.subtract)
    small_tile_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(small_tile_s, value=small_tile)
    start_tail_back_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_tail_back_s, data1=n_sbuf, data2=small_tile_s, op=nl.subtract)

    _tiles = [large_tile, medium_tile, small_tile, small_tile]
    _counts = [n_large_s, n_medium_s, n_small_s, n_tail_s]
    _starts = [None, start_medium_s, start_small_s, start_tail_back_s]
    _tier_names = ["large", "medium", "small", "tail"]
    for i in range(len(_tiles)):
        tile = _tiles[i]
        count_s = _counts[i]
        start_s = _starts[i]
        tier = _tier_names[i]
        if max_numel < tile:
            continue
        assert tile % n_prgs == 0, f"{tier}: tile {tile} % n_prgs {n_prgs} != 0"
        half = tile // n_prgs
        in_start = prg_id * half

        # The tail tier is a single tile -> never unroll; large/small unroll by `unroll`.
        tier_unroll = 1 if tier == "tail" else unroll

        # `tier_unroll` independent buffers (enables overlap).
        hbm_in = []
        hbm_out = []
        for j in range(tier_unroll):
            hbm_in.append(
                nl.ndarray((tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_in_{tier}_{j}")
            )
            hbm_out.append(
                nl.ndarray(
                    (tile * num_ranks, 1),
                    dtype=dtype,
                    buffer=nl.shared_hbm,
                    name=f"hbm_out_{tier}_{j}",
                )
            )

        # Dedicated unroll-remainder buffers. The unroll-remainder loop is a separate
        # fori_loop region (its own collective subgraph function); reusing a bulk slot here
        # would leave that shared_hbm tensor live across the bulk->unroll-remainder boundary,
        # which the multi-program (LNC>=2) collective lowering rejects (NCC_IESL012:
        # "shared internal tensors live on input to function").
        hbm_in_rem = None
        hbm_out_rem = None
        if tier_unroll > 1:
            hbm_in_rem = nl.ndarray(
                (tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_in_{tier}_rem"
            )
            hbm_out_rem = nl.ndarray(
                (tile * num_ranks, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_out_{tier}_rem"
            )

        # Carried offsets (advance across outer iterations).
        src_load = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        dst_off = []
        if start_s is None:
            nisa.memset(src_load, value=in_start)
            for r in range(num_ranks):
                rank_off = nl.ndarray(
                    (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"dst_{tier}_r{r}"
                )
                nisa.tensor_scalar(dst=rank_off, data=n_sbuf, op0=nl.multiply, operand0=r)
                dst_off.append(rank_off)
        else:
            in_start_sb = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"in_start_{tier}"
            )
            nisa.memset(in_start_sb, value=in_start)
            nisa.tensor_copy(dst=src_load, src=start_s)
            nisa.tensor_tensor(dst=src_load, data1=src_load, data2=in_start_sb, op=nl.add)
            for r in range(num_ranks):
                rank_off = nl.ndarray(
                    (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"dst_{tier}_r{r}"
                )
                nisa.tensor_scalar(dst=rank_off, data=n_sbuf, op0=nl.multiply, operand0=r)
                nisa.tensor_tensor(dst=rank_off, data1=rank_off, data2=start_s, op=nl.add)
                dst_off.append(rank_off)

        # Per-slot (j>=1) offset temporaries so the unrolled stages are independent
        # (slot j reads/writes at base + j*tile; j*tile is a compile-time constant). Slot
        # 0 uses the carried scalars directly. off_in_t[j] is the read offset; off_out_t[j]
        # is a single write-offset scratch reused across this program's ranks.
        off_in_t = []
        off_out_t = []
        for j in range(tier_unroll):
            if j == 0:
                off_in_t.append(None)
                off_out_t.append(None)
            else:
                off_in_t.append(
                    nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"off_in_{tier}_{j}")
                )
                off_out_t.append(
                    nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"off_out_{tier}_{j}")
                )

        # Offset increments for writing tiles into a rank-major output buffer.
        # To avoid fp32 rounding on the tensor_scalar instruction, use tensor_tensor instruction.
        inc = [None]
        for k in range(1, tier_unroll + 1):
            ck = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"inc_{tier}_{k}")
            nisa.memset(ck, value=k * tile)
            inc.append(ck)

        # Tile-count split (pow2 unroll -> shift/AND). tier_unroll==1 => bulk=count, no remainder.
        if tier_unroll > 1:
            bulk_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=bulk_s, data=count_s, op0=nl.right_shift, operand0=log2_unroll)
            rem_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rem_s, data=count_s, op0=nl.bitwise_and, operand0=(unroll - 1))
        else:
            bulk_s = count_s
            rem_s = None

        # bulk loop: `tier_unroll` tiles per iteration on independent buffers.
        ub_bulk = nisa.register_alloc()
        nisa.register_load(ub_bulk, bulk_s)
        _run_gather_loop(
            ub_bulk,
            tier_unroll,
            hbm_in,
            hbm_out,
            src_load,
            dst_off,
            off_in_t,
            off_out_t,
            inc,
            tile=tile,
            half=half,
            in_start=in_start,
            rank_start=rank_start,
            ranks_per_prg=ranks_per_prg,
            num_ranks=num_ranks,
            flat_in=flat_in,
            out_flat=out_flat,
            replica_group=replica_group,
        )

        # unroll-remainder loop: count & (unroll-1) leftover tiles, one per trip on the dedicated
        # remainder buffers. Same gather loop with tiles_per_trip=1 (advance uses inc[1]).
        if tier_unroll > 1:
            ub_rem = nisa.register_alloc()
            nisa.register_load(ub_rem, rem_s)
            _run_gather_loop(
                ub_rem,
                1,
                [hbm_in_rem],
                [hbm_out_rem],
                src_load,
                dst_off,
                off_in_t,
                off_out_t,
                inc,
                tile=tile,
                half=half,
                in_start=in_start,
                rank_start=rank_start,
                ranks_per_prg=ranks_per_prg,
                num_ranks=num_ranks,
                flat_in=flat_in,
                out_flat=out_flat,
                replica_group=replica_group,
            )

    return out_flat


_wrapped_allgather = wrap_nki(dynamic_allgather_kernel)


def lower_dynamic_allgather(inputs, replica_groups):
    """StableHLO lowering for the dynamic allgather. Traces the kernel at the bucket
    floor (trace_numel elements) so one shape-agnostic NEFF serves any larger numel at
    runtime."""
    import torch

    from torch_neuronx.nki_hop import kernel_registry
    from torch_neuronx.python_ops.neuron_nki_call_lowering import _lower_custom_nki
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    input_tensor = inputs[0]
    world = len(replica_groups[0])
    lnc = int(get_logical_neuron_cores())
    # Use the ACTUAL dtype size (matches the C++ ComputePlan used for the bucketed key);
    # the plan defaults to bf16 (2 bytes) which would diverge for other dtypes.
    ds = input_tensor.element_size()
    numel = int(input_tensor.numel())
    from torch_neuronx import _C

    platform = get_platform_target()

    # Flat ELEMENT tiling (mirrors the C++ dispatch; see module docstring for the contract):
    # `trace_numel` is the bucket-FLOOR tile the NEFF is traced at, `unroll` is folded in.
    plan = plan_dynamic_allgather(numel, world, lnc, ds, platform)
    if plan is None:
        raise RuntimeError(
            f"dynamic allgather lowering reached below floor: numel={numel} "
            f"(world={world}, lnc={lnc}); C++ eligibility check disagrees."
        )

    # C++/Python parity: both are the flat (numel) element-tiling ComputePlan on the same
    # (platform, world) table, so the tiles + trace_numel floor + unroll are identical.
    _, _, _, cpp_trace_numel, _, cpp_unroll = _C._allgather_compute_plan(
        numel, world, lnc, ds, platform
    )
    # C++/Python parity: both apply the same flat element-tiling ComputePlan on the SAME
    # (platform, world) table, so trace_numel + unroll match exactly.
    assert plan["trace_numel"] == cpp_trace_numel and plan["unroll"] == cpp_unroll, (
        f"plan mismatch: python trace_numel={plan['trace_numel']}/unroll={plan['unroll']} vs "
        f"C++ trace_numel={cpp_trace_numel}/unroll={cpp_unroll} "
        f"(numel={numel}, world={world}); tiling diverged."
    )
    unroll = plan["unroll"]

    # Lower through the shared NKI-op path (_lower_custom_nki), like every other NKI kernel. The
    # kernel's compile-time (non-tensor) params are registered as constant args keyed for the NKI
    # cache; the names MUST match the kernel signature. ReplicaGroup is a @dataclass, so its repr
    # is content-based -> a stable constant_args_key.
    constant_args = {
        "replica_group": ReplicaGroup(replica_groups),
        "num_ranks": world,
        "large_tile": plan["large"],
        "medium_tile": plan["medium"],
        "small_tile": plan["small"],
        "unroll": unroll,
    }
    constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
    kernel_registry.add_constant_args(constant_args, constant_args_key)

    # Trace placeholders on the META device (shape-only stub; no on-device fill/NEFF). The
    # NEFF is shape-agnostic: trace the input as [trace_numel, 1] (C++ binds [numel, 1] at
    # runtime) + a [1,1] int32 numel scalar. grid = [lnc] (kernel num_programs).
    input_floor = torch.empty(plan["trace_numel"], 1, dtype=input_tensor.dtype, device="meta")
    scalar_ex = torch.zeros(1, 1, dtype=torch.int32, device="meta")
    return _lower_custom_nki(
        (input_floor, scalar_ex), _wrapped_allgather.kernel_idx, constant_args_key, [lnc]
    )
