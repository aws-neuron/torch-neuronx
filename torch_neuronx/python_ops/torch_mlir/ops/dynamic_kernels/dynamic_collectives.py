"""Dynamic multi-tier collective NKI kernels and StableHLO lowering for C++ NeuronBackend.

The C++ dispatch flattens each per-rank input to [numel, 1]. AllGather produces
[world*numel, 1], while AllReduce produces [numel, 1]. The kernels are shape-agnostic: each
NEFF is traced at a fixed floor trace_numel, and a runtime int32 scalar drives the on-device
tile counts.

"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from torch_neuronx.nki_hop import wrap_nki

_KiB, _MiB = 1024, 1024 * 1024
# Tiling tables keyed by (profile, collective, group-size bucket); platforms map to a profile
# via _profile_for_platform. Each table is a list of regimes
# (name, floor_bytes, (large, medium, small) byte targets).
# compute_tiling (the shared core) picks the first regime with floor <= message bytes, then
# _tile_for converts the byte targets to pow2 element tiles. All-gather counts gathered bytes
# (per_rank=False); all-reduce counts per-rank bytes (per_rank=True).

# Default AllGather profile, small replica groups (size <= _GROUP_SMALL_MAX).
_DEFAULT_AG_SMALL = [
    ("A", 32 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("B", 16 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("C", 8 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("D", 2 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("E", 512 * _KiB, (1 * _MiB, 512 * _KiB, 256 * _KiB)),
    ("F", 0, (256 * _KiB, 128 * _KiB, 64 * _KiB)),
]
# Default AllGather profile, large replica groups (size > _GROUP_SMALL_MAX).
_DEFAULT_AG_LARGE = [
    ("A", 64 * _MiB, (64 * _MiB, 32 * _MiB, 16 * _MiB)),
    ("B", 32 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("C", 16 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("D", 4 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("E", 1 * _MiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("F", 0, (512 * _KiB, 256 * _KiB, 128 * _KiB)),
]
# All-reduce tables. Targets are PER-RANK message bytes (all-reduce output == input size)
# Default profile, small replica groups (size <= _GROUP_SMALL_MAX).
_DEFAULT_AR_SMALL = [
    ("A", 8 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("B", 4 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("C", 2 * _MiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("D", 512 * _KiB, (1 * _MiB, 512 * _KiB, 256 * _KiB)),
    ("E", 128 * _KiB, (256 * _KiB, 128 * _KiB, 64 * _KiB)),
    ("F", 0, (64 * _KiB, 32 * _KiB, 16 * _KiB)),
]
# Default profile, large replica groups (size > _GROUP_SMALL_MAX).
_DEFAULT_AR_LARGE = [
    ("A", 16 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("B", 8 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("C", 4 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("D", 1 * _MiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("E", 256 * _KiB, (512 * _KiB, 256 * _KiB, 128 * _KiB)),
    ("F", 0, (128 * _KiB, 64 * _KiB, 32 * _KiB)),
]
# ── TRN2-tuned all-gather tables (profile "trn2") ──────────────────────────────
# 3-regime design: fewer NEFFs (3 per collective) while covering the full numel range.
# Small table (tp, world<=8): 3 regimes covering large (>=8MiB), medium (>=1MiB), small (<1MiB).
_TRN2_AG_SMALL = [
    ("A", 8 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("B", 1 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("C", 0, (512 * _KiB, 256 * _KiB, 128 * _KiB)),
]
# Large table (fsdp, world>8): 3 regimes.
_TRN2_AG_LARGE = [
    ("A", 64 * _MiB, (64 * _MiB, 32 * _MiB, 16 * _MiB)),
    ("B", 8 * _MiB, (16 * _MiB, 8 * _MiB, 8 * _MiB)),
    ("C", 0, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
]

# ── TRN3-tuned all-gather tables (profile "trn3") ──────────────────────────────
# 3-regime design: fewer NEFFs (3 per collective) while covering the full numel range.
# Small replica groups (size <= _GROUP_SMALL_MAX).
_TRN3_AG_SMALL = [
    ("A", 8 * _MiB, (32 * _MiB, 16 * _MiB, 8 * _MiB)),
    ("B", 1 * _MiB, (4 * _MiB, 2 * _MiB, 1 * _MiB)),
    ("C", 0, (512 * _KiB, 256 * _KiB, 128 * _KiB)),
]
# Large replica groups (size > _GROUP_SMALL_MAX): 3 regimes.
_TRN3_AG_LARGE = [
    ("A", 16 * _MiB, (64 * _MiB, 32 * _MiB, 16 * _MiB)),
    ("B", 4 * _MiB, (8 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("C", 0, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
]

# ── TRN3-tuned all-reduce tables (profile "trn3") ─────────────────────────────
# 3-regime design matching the all-gather pattern.
_TRN3_AR_SMALL = [
    ("A", 2 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("B", 256 * _KiB, (1 * _MiB, 512 * _KiB, 256 * _KiB)),
    ("C", 0, (128 * _KiB, 32 * _KiB, 16 * _KiB)),
]
# TRN3 all-reduce large table (3-regime).
_TRN3_AR_LARGE = [
    ("A", 4 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("B", 512 * _KiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("C", 0, (128 * _KiB, 64 * _KiB, 32 * _KiB)),
]

# ── TRN2 all-reduce tables (3-regime) ─────────────────────────────────────────
_TRN2_AR_SMALL = [
    ("A", 2 * _MiB, (8 * _MiB, 4 * _MiB, 2 * _MiB)),
    ("B", 256 * _KiB, (1 * _MiB, 512 * _KiB, 256 * _KiB)),
    ("C", 0, (128 * _KiB, 32 * _KiB, 16 * _KiB)),
]
_TRN2_AR_LARGE = [
    ("A", 4 * _MiB, (16 * _MiB, 8 * _MiB, 4 * _MiB)),
    ("B", 512 * _KiB, (2 * _MiB, 1 * _MiB, 512 * _KiB)),
    ("C", 0, (128 * _KiB, 64 * _KiB, 32 * _KiB)),
]

# Tables keyed by (profile, collective, group-size bucket). All-gather counts gathered bytes;
# all-reduce counts per-rank bytes (see _COLLECTIVE_PER_RANK).
_TILING_TABLES = {
    ("default", "allgather", "small"): _DEFAULT_AG_SMALL,
    ("default", "allgather", "large"): _DEFAULT_AG_LARGE,
    ("default", "allreduce", "small"): _DEFAULT_AR_SMALL,
    ("default", "allreduce", "large"): _DEFAULT_AR_LARGE,
    ("trn2", "allgather", "small"): _TRN2_AG_SMALL,
    ("trn2", "allgather", "large"): _TRN2_AG_LARGE,
    ("trn2", "allreduce", "small"): _TRN2_AR_SMALL,
    ("trn2", "allreduce", "large"): _TRN2_AR_LARGE,
    ("trn3", "allgather", "small"): _TRN3_AG_SMALL,
    ("trn3", "allgather", "large"): _TRN3_AG_LARGE,
    ("trn3", "allreduce", "small"): _TRN3_AR_SMALL,
    ("trn3", "allreduce", "large"): _TRN3_AR_LARGE,
}
# Per-element byte accounting per collective: all-gather = gathered (x world), all-reduce =
# per-rank. Consumed by _msg_bytes_per_elem via compute_tiling.
_COLLECTIVE_PER_RANK = {"allgather": False, "allreduce": True}
# Replica groups with size <= this use the "small" table; larger use "large".
_GROUP_SMALL_MAX = 8


def _profile_for_platform(platform):
    """Map a Neuron platform target (e.g. 'trn2'/'trn3') to a tiling profile. This is the ONLY
    place platform identity is interpreted; everything downstream keys on the profile.
    TRN2 uses 'trn2' (tuned large table); TRN3 uses 'trn3' (tuned small+large+AR tables).
    Unknown platforms fall back to 'default'."""
    if platform == "trn3":
        return "trn3"
    if platform == "trn2":
        return "trn2"
    return "default"


def _resolve_table(platform, group_size, collective="allgather"):
    """Pick the tiling table for (platform, collective, replica-group size), via
    (profile, collective, group bucket)."""
    bucket = "small" if group_size <= _GROUP_SMALL_MAX else "large"
    return _TILING_TABLES[(_profile_for_platform(platform), collective, bucket)]


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


def _msg_bytes_per_elem(num_ranks, dtype_size, per_rank):
    """Bytes of collective traffic per logical element -- the divisor that turns a regime's
    byte target into an element tile. All-gather counts GATHERED bytes across all ranks
    (dtype * world); all-reduce counts PER-RANK bytes (dtype), since its output is the same
    size as its input. This is the ONLY place the two collectives' byte accounting differs."""
    return dtype_size * (1 if per_rank else num_ranks)


def _tile_for(target_bytes, num_ranks, n_prgs, dtype_size=2, per_rank=False):
    """Elements whose message size ~= target_bytes, as a power of two (>= n_prgs), so the
    kernel derives tile counts on-device with hardware-safe shift/AND. `per_rank` selects the
    byte accounting (see _msg_bytes_per_elem). Mirrors C++ TileFor."""
    bytes_per_elem = _msg_bytes_per_elem(num_ranks, dtype_size, per_rank)
    r = max(1, round(target_bytes / bytes_per_elem))
    return max(int(n_prgs), _pow2_floor(r))


def regime_tiles(num_ranks, n_prgs, targets, dtype_size=2, per_rank=False):
    """Convert byte targets into non-increasing element tiles.

    Args:
        num_ranks (int): Number of ranks participating in the collective.
        n_prgs (int): Number of NKI programs used as the minimum tile size.
        targets (tuple): Large, medium, and small byte targets for the regime.
        dtype_size (int, optional): Number of bytes per tensor element. Defaults to ``2``.
        per_rank (bool, optional): Whether targets count per-rank rather than gathered bytes.
            Defaults to ``False``.

    Returns:
        tuple: Large, medium, and small element tile sizes.
    """
    large_t, medium_t, small_t = targets
    large = _tile_for(large_t, num_ranks, n_prgs, dtype_size, per_rank)
    medium = min(large, _tile_for(medium_t, num_ranks, n_prgs, dtype_size, per_rank))
    small = min(medium, _tile_for(small_t, num_ranks, n_prgs, dtype_size, per_rank))
    return large, medium, small


def compute_tiling(numel, num_ranks, n_prgs, table, dtype_size=2, per_rank=False):
    """Select a tiling regime and derive its element tile sizes.

    Args:
        numel (int): Number of logical elements in the per-rank shard.
        num_ranks (int): Number of ranks participating in the collective.
        n_prgs (int): Number of NKI programs used as the minimum tile size.
        table (list): Regimes ordered by descending byte floor with a final zero floor.
        dtype_size (int, optional): Number of bytes per tensor element. Defaults to ``2``.
        per_rank (bool, optional): Whether message bytes exclude the world-size multiplier.
            Defaults to ``False``.

    Returns:
        tuple: Large, medium, and small tile sizes followed by the matched regime name.

    Raises:
        RuntimeError: If no regime matches the computed message size.

    Note:
        All-gather counts gathered bytes, while all-reduce counts per-rank bytes.
    """
    message = numel * _msg_bytes_per_elem(num_ranks, dtype_size, per_rank)
    for name, floor, targets in table:
        if message >= floor:
            large, medium, small = regime_tiles(num_ranks, n_prgs, targets, dtype_size, per_rank)
            return large, medium, small, name
    raise RuntimeError(f"no regime matched message_bytes={message}")  # unreachable: last floor==0


def pick_dynamic_tiling(
    numel, num_ranks, n_prgs, dtype_size=2, platform="trn2", collective="allgather"
):
    """Select dynamic collective tiles from the platform-specific table.

    Args:
        numel (int): Number of logical elements in the per-rank shard.
        num_ranks (int): Number of ranks participating in the collective.
        n_prgs (int): Number of NKI programs used as the minimum tile size.
        dtype_size (int, optional): Number of bytes per tensor element. Defaults to ``2``.
        platform (str, optional): Neuron platform target. Defaults to ``"trn2"``.
        collective (str, optional): Collective name, ``"allgather"`` or ``"allreduce"``.
            Defaults to ``"allgather"``.

    Returns:
        tuple: Large, medium, and small tile sizes followed by the matched regime name.
    """
    table = _resolve_table(platform, num_ranks, collective)
    per_rank = _COLLECTIVE_PER_RANK[collective]
    return compute_tiling(numel, num_ranks, n_prgs, table, dtype_size, per_rank)


def _pick_unroll(extent, trace):
    """Unroll factor in {1,2,4}: the deepest pow2 that fits the primary tile count"""
    primary = extent // trace if trace > 0 else 1
    if primary >= 4:
        return 4
    if primary >= 2:
        return 2
    return 1


def _plan_dynamic(numel, world, n_prgs, dtype_size, platform, collective):
    """Shared plan builder for the dynamic collectives. Selects the (platform, collective,
    world) tiling table, converts it to (large, medium, small) element tiles, and picks the
    trace floor (largest tile <= numel) + unroll. Returns None (caller falls back to the native
    collective) when numel is too small to fill even one small tile, or world % n_prgs != 0.

    Returns dict(large, medium, small, regime, trace_numel, unroll):
      * large, medium, small : the three non-increasing pow2 element tiles for this shard, from
        the matched regime. The kernel decomposes numel greedily large -> medium -> small, then a
        single small-tile tail.
      * regime      : the matched regime name (for logging only).
      * trace_numel : the tile the NEFF is traced at, and the identity of this shard's dispatch
        bucket.
      * unroll      : tiles processed per dynamic-loop iteration.
    """
    if numel <= 0 or world % n_prgs != 0:
        return None
    large, medium, small, regime = pick_dynamic_tiling(
        numel, world, n_prgs, dtype_size, platform, collective
    )
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


def plan_dynamic_allgather(numel, world, n_prgs, dtype_size=2, platform="trn2"):
    """Build the dynamic all-gather tiling plan for a per-rank shard.

    Args:
        numel (int): Number of elements in the per-rank shard.
        world (int): Replica-group size.
        n_prgs (int): Number of NKI programs.
        dtype_size (int, optional): Number of bytes per tensor element. Defaults to ``2``.
        platform (str, optional): Neuron platform target. Defaults to ``"trn2"``.

    Returns:
        dict or None: Keys ``large``, ``medium``, ``small``, ``regime``,
            ``trace_numel``, and ``unroll``, or ``None`` when the shard is too small or
            ``world`` is not divisible by ``n_prgs``.

    Note:
        Regime selection is keyed on gathered bytes: ``numel * dtype_size * world``.
    """
    return _plan_dynamic(numel, world, n_prgs, dtype_size, platform, "allgather")


def plan_dynamic_allreduce(numel, world, n_prgs, dtype_size=2, platform="trn2"):
    """Build the dynamic all-reduce tiling plan for a per-rank tensor.

    Args:
        numel (int): Number of elements in the per-rank tensor.
        world (int): Replica-group size.
        n_prgs (int): Number of NKI programs.
        dtype_size (int, optional): Number of bytes per tensor element. Defaults to ``2``.
        platform (str, optional): Neuron platform target. Defaults to ``"trn2"``.

    Returns:
        dict or None: Keys ``large``, ``medium``, ``small``, ``regime``,
            ``trace_numel``, and ``unroll``, or ``None`` when the tensor is too small or
            ``world`` is not divisible by ``n_prgs``.

    Note:
        Within the world-selected table, regimes use per-rank bytes without a
        world-size multiplier.
    """
    return _plan_dynamic(numel, world, n_prgs, dtype_size, platform, "allreduce")


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
        """Gather one unrolled trip and advance its carried offsets.

        Args:
            _i (int): Loop iteration index supplied by ``nl.fori_loop``.
        """
        for j in range(tiles_per_trip):
            if j == 0:
                off_in = src_load
            else:
                nisa.tensor_tensor(dst=off_in_t[j], data1=src_load, data2=inc[j], op=nl.add)
                off_in = off_in_t[j]
            nisa.dma_copy(
                dst=in_bufs[j][in_start : in_start + half, :],
                src=flat_in.ap(pattern=[[1, half], [1, 1]], scalar_offset=off_in, indirect_dim=0),
                dge_mode=nisa.dge_mode.hwdge,
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
                    dge_mode=nisa.dge_mode.hwdge,
                )
        # Advance carried offsets by the tiles consumed this trip. Only update the
        # ranks this program actually scatters
        step = inc[tiles_per_trip]
        nisa.tensor_tensor(dst=src_load, data1=src_load, data2=step, op=nl.add)
        for r in range(rank_start, rank_start + ranks_per_prg):
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
    """Gather a runtime-sized flattened shard across a replica group.

    Args:
        input_tensor (nl.NkiTensor): Flat ``[max_numel, 1]`` bucket floor reused for larger inputs.
        numel_tensor (nl.NkiTensor): Runtime ``[1, 1]`` int32 count driving on-device tile counts.
        replica_group (ReplicaGroup): Ranks participating in the all-gather.
        num_ranks (int): Compile-time group size split across programs.
        large_tile (int): Largest tier in a greedy large-to-medium-to-small decomposition.
        medium_tile (int): Compile-time power-of-two tile no larger than ``large_tile``.
        small_tile (int): Compile-time tile also used for idempotent backward-aligned tails.
        unroll (int): Non-tail width in ``{1, 2, 4}``; independent buffers overlap
            bulk work and isolate remainders.

    Returns:
        nl.NkiTensor: Shared-HBM ``[num_ranks * max_numel, 1]``; rank ``r`` starts at ``r * numel``.

    Raises:
        AssertionError: If a tile or ``unroll`` is not a power of two or a tile is
            not divisible by the program count.
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
_precompiled_keys: set = set()


# ---------------------------------------------------------------------------
# Pre-compilation: warm the NEFF cache for all tiling buckets
# ---------------------------------------------------------------------------


def _enumerate_buckets(world, lnc, dtype_size, platform, collective):
    """Enumerate every (trace_numel, large, medium, small, unroll) bucket the ComputePlan can
    produce for this (world, lnc, dtype, platform, collective) across all numel, so precompiling
    the full set eliminates runtime recompiles."""
    per_rank = collective == "allreduce"
    bpe = _msg_bytes_per_elem(world, dtype_size, per_rank)
    table = _resolve_table(platform, world, collective)
    buckets = []
    seen = set()

    def _emit(trace, large, medium, small, unroll):
        key = (trace, large, medium, small, unroll)
        if key not in seen:
            seen.add(key)
            buckets.append(
                {
                    "trace_numel": trace,
                    "large": large,
                    "medium": medium,
                    "small": small,
                    "unroll": unroll,
                }
            )

    inf = float("inf")
    prev_floor = inf
    for _name, floor, targets in table:
        large, medium, small = regime_tiles(world, lnc, targets, dtype_size, per_rank)
        lo = -(-floor // bpe)  # ceil(floor / bpe)
        hi = inf if prev_floor == inf else -(-prev_floor // bpe)
        prev_floor = floor

        tiles_desc = [large, medium, small]  # non-increasing
        for idx, trace in enumerate(tiles_desc):
            if trace in tiles_desc[:idx]:
                continue

            # trace is selected while numel in [trace, next_larger_tile); larger tiles win above.
            larger = [t for t in tiles_desc if t > trace]
            next_tile = min(larger) if larger else inf
            t_lo = max(lo, trace)
            t_hi = min(hi, next_tile)
            if t_lo >= t_hi:
                continue  # this trace is unreachable within the regime's numel range
            # unroll = _pick_unroll(numel, trace): 1 in [t,2t), 2 in [2t,4t), 4 in [4t, inf).
            for unroll, u_lo, u_hi in (
                (1, trace, 2 * trace),
                (2, 2 * trace, 4 * trace),
                (4, 4 * trace, inf),
            ):
                if max(t_lo, u_lo) < min(t_hi, u_hi):
                    _emit(trace, large, medium, small, unroll)

    return buckets


def _maybe_precompile_remaining_buckets(replica_groups, dtype, current_plan, current_collective):
    """Precompile remaining dynamic-collective NEFF buckets for the triggering collective.

    Fires once per unique (world, collective) pair.
    """
    import os

    if os.environ.get("NEURON_DYN_NKI_COLLECTIVES") != "1":
        return
    if os.environ.get("NEURON_DYN_NKI_PRECOMPILE") != "1":
        return

    world = len(replica_groups[0])
    precompile_key = (world, current_collective)
    if precompile_key in _precompiled_keys:
        return
    _precompiled_keys.add(precompile_key)

    import logging

    import torch

    from torch_neuronx import _C
    from torch_neuronx.nki_hop import kernel_registry
    from torch_neuronx.python_ops.neuron_nki_call_lowering import _lower_custom_nki
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    logger = logging.getLogger(__name__)

    eager_args, eager_opt_level = _C._get_eager_compiler_config()
    dtype_size = torch._utils._element_size(dtype)
    lnc = int(get_logical_neuron_cores())
    platform = get_platform_target()

    current_key = (
        current_plan["trace_numel"],
        current_plan["large"],
        current_plan["medium"],
        current_plan["small"],
        current_plan["unroll"],
    )

    wrapped_kernel = _wrapped_allgather if current_collective == "allgather" else _wrapped_allreduce

    compiled = 0
    for bucket in _enumerate_buckets(world, lnc, dtype_size, platform, current_collective):
        bucket_key = (
            bucket["trace_numel"],
            bucket["large"],
            bucket["medium"],
            bucket["small"],
            bucket["unroll"],
        )
        if bucket_key == current_key:
            continue  # the current dispatch already compiled this NEFF

        constant_args = {
            "replica_group": ReplicaGroup(replica_groups),
            "num_ranks": world,
            "large_tile": bucket["large"],
            "medium_tile": bucket["medium"],
            "small_tile": bucket["small"],
            "unroll": bucket["unroll"],
        }
        if current_collective == "allreduce":
            constant_args["reduce_op"] = 0

        constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
        kernel_registry.add_constant_args(constant_args, constant_args_key)
        input_floor = torch.empty(bucket["trace_numel"], 1, dtype=dtype, device="meta")
        scalar_ex = torch.zeros(1, 1, dtype=torch.int32, device="meta")
        result = _lower_custom_nki(
            (input_floor, scalar_ex), wrapped_kernel.kernel_idx, constant_args_key, [lnc]
        )
        if result and result.stablehlo_bytes:
            _C.compile_graph_with_config(
                result.cache_key,
                result.stablehlo_bytes,
                True,  # has_collectives
                eager_args,
                eager_opt_level,
            )
            compiled += 1

    logger.debug(
        f"[precompile] submitted {compiled} NEFFs for background compilation "
        f"(world={world}, collective={current_collective}, platform={platform}, dtype={dtype})"
    )


def lower_dynamic_allgather(inputs, replica_groups):
    """Lower a dynamic all-gather to a bucketed shape-agnostic NEFF.

    Args:
        inputs (Sequence[torch.Tensor]): Input tensors, with the per-rank shard first.
        replica_groups (Sequence[Sequence[int]]): Participating replica groups.

    Returns:
        LoweringResult: StableHLO, cache key, return types, and collective metadata.

    Raises:
        RuntimeError: If the input is below the dynamic all-gather eligibility floor.
        AssertionError: If the Python and C++ trace-floor or unroll plans differ.

    Note:
        The kernel is traced at the bucket floor and reused for larger runtime element counts.
    """
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
    result = _lower_custom_nki(
        (input_floor, scalar_ex), _wrapped_allgather.kernel_idx, constant_args_key, [lnc]
    )

    # After the triggering op's own NEFF is lowered and submitted to the compilation pool,
    # precompile all other tiling-bucket NEFFs. Their neuronx-cc compilations run in parallel
    # on the C++ CompilationThreadPool.
    _maybe_precompile_remaining_buckets(replica_groups, input_tensor.dtype, plan, "allgather")

    return result


# ---------------------------------------------------------------------------
# Dynamic multi-tier AllReduce
# ---------------------------------------------------------------------------
def _reduce_op_from_code(code):
    if code == 0:
        return nl.add
    if code == 1:
        return nl.minimum
    return nl.maximum


def _reduce_identity_from_code(code):
    if code == 0:
        return 0.0
    if code == 1:
        return float("inf")
    return float("-inf")  # code == 2


def _run_reduce_loop(
    count_reg: nisa.VirtualRegister,
    tiles_per_trip: int,
    in_bufs: list[nl.NkiTensor],
    out_bufs: list[nl.NkiTensor],
    base_off: nl.NkiTensor,
    off_t: list[nl.NkiTensor | None],
    inc: list[nl.NkiTensor | None],
    tile: int,
    half: int,
    in_start: int,
    flat_in: nl.NkiTensor,
    out_flat: nl.NkiTensor,
    replica_group: ReplicaGroup,
    reduce_op,
    oob_skip: bool = False,
    identity_tile: nl.NkiTensor | None = None,
) -> None:
    """One nl.fori_loop that all-reduces `tiles_per_trip` tiles per trip.

    The regular tiers load and store fully in-bounds tiles. The forward tail passes
    ``oob_skip=True`` and an SBUF ``identity_tile``: each program first fills its shared-HBM
    input slice with the reduction identity, then overlays only in-bounds input elements. The
    collective still sees one fixed-size tile, while the OOB-skipped store writes only the true
    remainder. ``flat_in`` and ``out_flat`` are distinct NKI tensors (read-only input, fresh
    output). In-place is a caller concern: the collective backend binds the same caller tensor
    to both the input and output NEFF ports, so the write lands back in the input's storage.
    Each tile is staged into its collective buffer before its output DMA, so read-before-write
    holds under that binding.

    All per-tier state is passed in (base_off / off_t mutated in place), so the body closure
    captures no enclosing loop variable.
    """

    if identity_tile is not None:
        identity_elems = identity_tile.size
        assert (
            half % identity_elems == 0
        ), f"identity tile {identity_elems} must divide program slice {half}"
        identity_shape = identity_tile.shape
        identity_chunks = half // identity_elems

    def body(_i):
        """Reduce one unrolled trip and advance its carried offset.

        Args:
            _i (int): Loop iteration index supplied by ``nl.fori_loop``.
        """
        for j in range(tiles_per_trip):
            if j == 0:
                off = base_off
            else:
                nisa.tensor_tensor(dst=off_t[j], data1=base_off, data2=inc[j], op=nl.add)
                off = off_t[j]

            if identity_tile is not None:
                for k in range(identity_chunks):
                    chunk_start = in_start + k * identity_elems
                    chunk_end = chunk_start + identity_elems
                    nisa.dma_copy(
                        dst=in_bufs[j][chunk_start:chunk_end, :].reshape(identity_shape),
                        src=identity_tile,
                    )

            input_view = flat_in.ap(pattern=[[1, half], [1, 1]], scalar_offset=off, indirect_dim=0)
            if oob_skip:
                nisa.dma_copy(
                    dst=in_bufs[j][in_start : in_start + half, :],
                    src=input_view,
                    oob_mode=nisa.oob_mode.skip,
                    dge_mode=nisa.dge_mode.hwdge,
                )
            else:
                nisa.dma_copy(
                    dst=in_bufs[j][in_start : in_start + half, :],
                    src=input_view,
                    dge_mode=nisa.dge_mode.hwdge,
                )

            ncc.all_reduce(
                srcs=[in_bufs[j]], dsts=[out_bufs[j]], replica_group=replica_group, op=reduce_op
            )

            output_view = out_flat.ap(
                pattern=[[1, half], [1, 1]], scalar_offset=off, indirect_dim=0
            )
            if oob_skip:
                nisa.dma_copy(
                    dst=output_view,
                    src=out_bufs[j][in_start : in_start + half, :],
                    oob_mode=nisa.oob_mode.skip,
                    dge_mode=nisa.dge_mode.hwdge,
                )
            else:
                nisa.dma_copy(
                    dst=output_view,
                    src=out_bufs[j][in_start : in_start + half, :],
                    dge_mode=nisa.dge_mode.hwdge,
                )

        # Advance the carried offset by the tiles consumed this trip.
        step = inc[tiles_per_trip]
        nisa.tensor_tensor(dst=base_off, data1=base_off, data2=step, op=nl.add)

    nl.fori_loop(0, count_reg, body)


@nki.jit
def dynamic_allreduce_kernel(
    input_tensor: nl.NkiTensor,
    numel_tensor: nl.NkiTensor,
    replica_group: ReplicaGroup,
    num_ranks: int,
    large_tile: int,
    medium_tile: int,
    small_tile: int,
    unroll: int,
    reduce_op: int,
) -> nl.NkiTensor:
    """Reduce a runtime-sized flattened tensor across a replica group.

    Args:
        input_tensor (nl.NkiTensor): Flat ``[max_numel, 1]`` bucket floor reused for larger inputs.
        numel_tensor (nl.NkiTensor): Runtime ``[1, 1]`` int32 count driving greedy tier counts.
        replica_group (ReplicaGroup): Ranks participating in the all-reduce.
        num_ranks (int): Unused compile-time argument; ``replica_group`` determines participants.
        large_tile (int): Compile-time large power-of-two tile size.
        medium_tile (int): Compile-time power-of-two tile no larger than ``large_tile``.
        small_tile (int): Tile for the nonoverlapping, identity-filled OOB-skipping tail.
        unroll (int): Non-tail width in ``{1, 2, 4}``; remainders use separate buffers.
        reduce_op (int): Reduction code: ``0`` for sum, ``1`` for minimum, otherwise maximum.

    Returns:
        nl.NkiTensor: Dedicated functional output; the backend can bind its storage
            to the caller tensor without StableHLO alias metadata.

    Raises:
        AssertionError: If a tile or ``unroll`` is not a power of two or a tile is
            not divisible by the program count.
    """
    max_numel, _one = input_tensor.shape  # bound flat as [numel, 1]
    dtype = input_tensor.dtype
    n_prgs = nl.num_programs(axes=0)
    prg_id = nl.program_id(axis=0)

    flat_in = input_tensor
    out_flat = nl.ndarray((max_numel, 1), dtype=dtype, buffer=nl.shared_hbm, name="output")
    op = _reduce_op_from_code(reduce_op)
    identity = _reduce_identity_from_code(reduce_op)

    mask_large = large_tile - 1
    mask_medium = medium_tile - 1
    mask_small = small_tile - 1
    log2_large = _log2_const(large_tile)
    log2_medium = _log2_const(medium_tile)
    log2_small = _log2_const(small_tile)
    log2_unroll = _log2_const(unroll)
    n_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=n_sbuf, src=numel_tensor)

    # Runtime tile counts from numel via shift/AND (identical to the all-gather kernel).
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

    # n_tail_s = ceil(r2_s / small_tile): 0 or 1 tile for numel mod small_tile.
    mask_small_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(mask_small_s, value=mask_small)
    n_tail_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=n_tail_s, data1=r2_s, data2=mask_small_s, op=nl.add)
    nisa.tensor_scalar(dst=n_tail_s, data=n_tail_s, op0=nl.right_shift, operand0=log2_small)

    # Tier start offsets (exact int32 arithmetic): medium after the large tiles (numel - r0),
    # small after the medium tiles (numel - r1), and the forward tail at the first unprocessed
    # remainder element (numel - r2). The fixed tail tile extends past numel; OOB-skipped DMA
    # below suppresses those padded accesses.
    start_medium_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_medium_s, data1=n_sbuf, data2=r0_s, op=nl.subtract)
    start_small_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_small_s, data1=n_sbuf, data2=r1_s, op=nl.subtract)
    start_tail_forward_s = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=start_tail_forward_s, data1=n_sbuf, data2=r2_s, op=nl.subtract)

    _tiles = [large_tile, medium_tile, small_tile, small_tile]
    _counts = [n_large_s, n_medium_s, n_small_s, n_tail_s]
    _starts = [None, start_medium_s, start_small_s, start_tail_forward_s]
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

        # The tail tier is a single tile -> never unroll; large/medium/small unroll by `unroll`.
        tier_unroll = 1 if tier == "tail" else unroll

        # OOB-skipped loads preserve their destination for padded elements. Initialize the tail
        # staging slice to the reduction identity in bounded SBUF chunks so the collective never
        # consumes undefined HBM values. The configured tiles are powers of two, so this chunk
        # divides every program-owned half while keeping the SBUF partition dimension <= 128.
        identity_tile = None
        if tier == "tail":
            identity_elems = min(half, 64 * 1024)
            identity_partitions = min(identity_elems, 128)
            assert identity_elems % identity_partitions == 0
            identity_tile = nl.ndarray(
                (identity_partitions, identity_elems // identity_partitions),
                dtype=dtype,
                buffer=nl.sbuf,
                name="tail_identity",
            )
            nisa.memset(identity_tile, value=identity)

        # `tier_unroll` independent in/out buffers (enable overlap). All-reduce output is the same
        # size as its input, so both are [tile, 1] (half the footprint of the all-gather kernel).
        hbm_in = []
        hbm_out = []
        for j in range(tier_unroll):
            hbm_in.append(
                nl.ndarray((tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_in_{tier}_{j}")
            )
            hbm_out.append(
                nl.ndarray((tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_out_{tier}_{j}")
            )

        # Dedicated unroll-remainder buffers (separate collective subgraph region; reusing a bulk
        # slot would leave a shared_hbm tensor live across the bulk->remainder boundary, which the
        # multi-program collective lowering rejects -- see the all-gather kernel note).
        hbm_in_rem = None
        hbm_out_rem = None
        if tier_unroll > 1:
            hbm_in_rem = nl.ndarray(
                (tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_in_{tier}_rem"
            )
            hbm_out_rem = nl.ndarray(
                (tile, 1), dtype=dtype, buffer=nl.shared_hbm, name=f"hbm_out_{tier}_rem"
            )

        # Single carried offset (read == write). Large tier starts at this program's in_start;
        # the other tiers start at their tier-start plus in_start.
        base_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"off_{tier}")
        if start_s is None:
            nisa.memset(base_off, value=in_start)
        else:
            in_start_sb = nl.ndarray(
                (1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"in_start_{tier}"
            )
            nisa.memset(in_start_sb, value=in_start)
            nisa.tensor_copy(dst=base_off, src=start_s)
            nisa.tensor_tensor(dst=base_off, data1=base_off, data2=in_start_sb, op=nl.add)

        # Per-slot (j>=1) read/write offset temporaries so the unrolled stages are independent
        # (slot j is at base + j*tile; j*tile is a compile-time constant). Slot 0 uses base_off.
        off_t = []
        for j in range(tier_unroll):
            if j == 0:
                off_t.append(None)
            else:
                off_t.append(
                    nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"off_{tier}_{j}")
                )

        # Offset increments (k*tile) for the unrolled slots + the per-trip advance step.
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
        _run_reduce_loop(
            ub_bulk,
            tier_unroll,
            hbm_in,
            hbm_out,
            base_off,
            off_t,
            inc,
            tile=tile,
            half=half,
            in_start=in_start,
            flat_in=flat_in,
            out_flat=out_flat,
            replica_group=replica_group,
            reduce_op=op,
            oob_skip=(tier == "tail"),
            identity_tile=identity_tile,
        )

        # unroll-remainder loop: count & (unroll-1) leftover tiles, one per trip on the dedicated
        # remainder buffers. Same reduce loop with tiles_per_trip=1 (advance uses inc[1]).
        if tier_unroll > 1:
            ub_rem = nisa.register_alloc()
            nisa.register_load(ub_rem, rem_s)
            _run_reduce_loop(
                ub_rem,
                1,
                [hbm_in_rem],
                [hbm_out_rem],
                base_off,
                off_t,
                inc,
                tile=tile,
                half=half,
                in_start=in_start,
                flat_in=flat_in,
                out_flat=out_flat,
                replica_group=replica_group,
                reduce_op=op,
            )

    return out_flat


_wrapped_allreduce = wrap_nki(dynamic_allreduce_kernel)


# c10d ReduceOp int (SUM=0, MIN=3, MAX=4) -> kernel reduce-op code (see _reduce_op_from_code).
# Only these ops are supported by the NKI all_reduce; the C++ dispatch gates on the same set.
_C10D_REDUCE_TO_CODE = {0: 0, 3: 1, 4: 2}


def lower_dynamic_allreduce(inputs, replica_groups, reduce_op):
    """Lower a dynamic all-reduce to a bucketed shape-agnostic NEFF.

    Args:
        inputs (Sequence[torch.Tensor]): Input tensors, with the per-rank tensor first.
        replica_groups (Sequence[Sequence[int]]): Participating replica groups.
        reduce_op (int): c10d code ``0`` (sum), ``3`` (minimum), or ``4`` (maximum).

    Returns:
        LoweringResult: StableHLO, cache key, return types, and collective metadata.

    Raises:
        RuntimeError: If ``reduce_op`` is unsupported or the input is below the
            eligibility floor.
        AssertionError: If the Python and C++ trace-floor or unroll plans differ.

    Note:
        The reduction code is folded into the kernel cache key, and the bucket-floor NEFF
        serves larger runtime element counts. At execution, the C++ backend binds the caller
        tensor to both NEFF ports, so the reduction writes in place.
    """
    import torch

    from torch_neuronx import _C
    from torch_neuronx.nki_hop import kernel_registry
    from torch_neuronx.python_ops.neuron_nki_call_lowering import _lower_custom_nki
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    input_tensor = inputs[0]
    world = len(replica_groups[0])
    lnc = int(get_logical_neuron_cores())
    # ACTUAL dtype size (matches the C++ ComputePlan used for the bucketed key).
    ds = input_tensor.element_size()
    numel = int(input_tensor.numel())
    platform = get_platform_target()

    if reduce_op not in _C10D_REDUCE_TO_CODE:
        raise RuntimeError(
            f"dynamic allreduce lowering: unsupported reduce_op {reduce_op}; "
            f"C++ eligibility check disagrees."
        )
    reduce_code = _C10D_REDUCE_TO_CODE[reduce_op]

    # Per-rank element tiling (all-reduce output == input size); `trace_numel` is the bucket-FLOOR
    # tile the NEFF is traced at, `unroll` is folded in.
    plan = plan_dynamic_allreduce(numel, world, lnc, ds, platform)
    if plan is None:
        raise RuntimeError(
            f"dynamic allreduce lowering reached below floor: numel={numel} "
            f"(world={world}, lnc={lnc}); C++ eligibility check disagrees."
        )

    # C++/Python parity: both apply the same per-rank element-tiling ComputePlan on the SAME
    # (platform, world) all-reduce table, so trace_numel + unroll match exactly.
    _, _, _, cpp_trace_numel, _, cpp_unroll = _C._allreduce_compute_plan(
        numel, world, lnc, ds, platform
    )
    assert plan["trace_numel"] == cpp_trace_numel and plan["unroll"] == cpp_unroll, (
        f"plan mismatch: python trace_numel={plan['trace_numel']}/unroll={plan['unroll']} vs "
        f"C++ trace_numel={cpp_trace_numel}/unroll={cpp_unroll} "
        f"(numel={numel}, world={world}); tiling diverged."
    )
    unroll = plan["unroll"]

    # Compile-time (non-tensor) params registered as constant args for the NKI cache; names MUST
    # match the kernel signature. ReplicaGroup is a @dataclass -> content-based repr -> stable key.
    constant_args = {
        "replica_group": ReplicaGroup(replica_groups),
        "num_ranks": world,
        "large_tile": plan["large"],
        "medium_tile": plan["medium"],
        "small_tile": plan["small"],
        "unroll": unroll,
        "reduce_op": reduce_code,
    }
    constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
    kernel_registry.add_constant_args(constant_args, constant_args_key)

    # Trace placeholders on META: input as [trace_numel, 1] (C++ binds [numel, 1] at runtime) + a
    # [1,1] int32 numel scalar. The all-reduce output is the same [trace_numel, 1] shape.
    #
    # Functional lowering (no operand/output alias): the kernel reads `input_tensor` and writes a
    # fresh `out_flat`, so the NEFF has one ordinary output and no aliasing metadata. In-place is
    # achieved by the collective backend binding the SAME caller tensor to both the input and the
    # output port (SubmitInNeffCollective: inputs={io, scalar}, outputs={io}); read-before-write
    # holds because each tile is staged into its collective buffer before its output DMA. This is
    # the collective-kernel convention (see nki.collectives kernels) -- do NOT inject a
    # custom-call output_operand_alias, which neuronx-cc rejects when the kernel is functional
    # ("Operator level aliasing is not the same as kernel-level aliasing").
    input_floor = torch.empty(plan["trace_numel"], 1, dtype=input_tensor.dtype, device="meta")
    scalar_ex = torch.zeros(1, 1, dtype=torch.int32, device="meta")
    result = _lower_custom_nki(
        (input_floor, scalar_ex),
        _wrapped_allreduce.kernel_idx,
        constant_args_key,
        [lnc],
    )

    # After the triggering op's own NEFF is lowered and submitted to the compilation pool,
    # precompile all other tiling-bucket NEFFs. Their neuronx-cc compilations run in parallel
    # on the C++ CompilationThreadPool, overlapping with the model's forward pass.
    _maybe_precompile_remaining_buckets(replica_groups, input_tensor.dtype, plan, "allreduce")

    return result
