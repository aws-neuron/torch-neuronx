"""Parity test: the C++ tiling (collective_tiling::ComputePlan, exposed via
torch_neuronx._C._allgather_compute_plan / _allreduce_compute_plan) must match the Python
reference plan_dynamic_allgather / plan_dynamic_allreduce for every (numel, world, nprg,
platform), plus a structural check that every registered tiling table satisfies its invariants.
"""

import pytest

from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives import (
    _TILING_TABLES,
    _enumerate_buckets,
    plan_dynamic_allgather,
    plan_dynamic_allreduce,
    regime_tiles,
)


def validate_table(table, num_ranks=8, n_prgs=2, dtype_size=2, per_rank=False):
    """Assert a table's structural invariants for a representative (num_ranks, n_prgs, dtype).
    per_rank selects the byte accounting (all-gather=gathered, all-reduce=per-rank)."""
    prev_floor = None
    for name, floor, targets in table:
        large, medium, small = regime_tiles(num_ranks, n_prgs, targets, dtype_size, per_rank)
        for t in (large, medium, small):
            assert t & (t - 1) == 0, f"{name}: tile {t} not pow2"
        assert large >= medium >= small, f"{name}: not non-increasing {large}/{medium}/{small}"
        assert large % medium == 0 and medium % small == 0, f"{name}: tiles not nested"
        # No dead zone: any numel that SELECTS this regime (message >= floor) must be able to fill
        # at least the small tile, i.e. numel_at_floor >= small. Otherwise numels in
        # [numel_at_floor, small) select the regime but fit no tile -> silent native fallback (and
        # non-monotonic eligibility). The catch-all (floor 0) is exempt: its small tile IS the
        # global eligibility floor (numels below it legitimately fall back).
        if floor > 0:
            bpe = dtype_size * (1 if per_rank else num_ranks)
            numel_at_floor = -(-floor // bpe)  # ceil(floor / bytes-per-element)
            assert numel_at_floor >= small, (
                f"{name}: dead zone -- regime selects at numel>={numel_at_floor} but its small "
                f"tile is {small}; numels in [{numel_at_floor}, {small}) fall back to native"
            )
        if prev_floor is not None:
            assert floor < prev_floor, f"{name}: floors not strictly descending"
        prev_floor = floor
    assert table[-1][1] == 0, "last regime floor must be 0 (catch-all)"
    return True


_NUMELS = [
    1,
    100,
    1024,
    4096,
    8192,
    2**14,
    65536,  # gathered 512KB @ world=4
    131072,  # gathered 1MB
    200000,
    262144,  # gathered 2MB
    524288,  # gathered 4MB
    1048576,  # gathered 8MB
    2_000_000,
    2_097_152,  # gathered 16MB
    3_000_000,
    4_194_304,  # gathered 32MB @ world=4
    2**25,
    5_000_000,
    9_723_904,
    2**26,  # exercises the large-group table at world>=16
    50_000_000,
]
# world <= 8 exercises the "small" table; world > 8 exercises the "large" table.
_WORLDS = [2, 4, 8, 16, 32]
_NPRGS = [1, 2]
_PLATFORMS = ["trn2", "trn3"]  # exercise both platform-specific table profiles
_DTYPE_SIZE = 2  # bf16


def _cpp_allgather_plan(numel, world, nprg, platform):
    from torch_neuronx import _C

    return _C._allgather_compute_plan(
        numel, world, nprg, _DTYPE_SIZE, platform
    )  # (large, medium, small, trace_numel, eligible, unroll)


def _cpp_allreduce_plan(numel, world, nprg, platform):
    from torch_neuronx import _C

    return _C._allreduce_compute_plan(
        numel, world, nprg, _DTYPE_SIZE, platform
    )  # (large, medium, small, trace_numel, eligible, unroll)


def _assert_plan_parity(cpp, py, ctx):
    large, medium, small, trace_numel, eligible, unroll = cpp
    if not eligible:
        assert py is None, ctx
    else:
        assert py is not None, ctx
        assert (py["large"], py["medium"], py["small"], py["trace_numel"]) == (
            large,
            medium,
            small,
            trace_numel,
        ), ctx
        assert py["unroll"] == unroll, ctx  # unroll parity (folded into both plans)


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("nprg", _NPRGS)
def test_allgather_plan_parity(world, nprg, platform):
    for numel in _NUMELS:
        cpp = _cpp_allgather_plan(numel, world, nprg, platform)
        py = plan_dynamic_allgather(numel, world, nprg, _DTYPE_SIZE, platform)
        _assert_plan_parity(cpp, py, f"AG numel={numel} world={world} nprg={nprg} plat={platform}")


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("nprg", _NPRGS)
def test_allreduce_plan_parity(world, nprg, platform):
    for numel in _NUMELS:
        cpp = _cpp_allreduce_plan(numel, world, nprg, platform)
        py = plan_dynamic_allreduce(numel, world, nprg, _DTYPE_SIZE, platform)
        _assert_plan_parity(cpp, py, f"AR numel={numel} world={world} nprg={nprg} plat={platform}")


@pytest.mark.parametrize("key", sorted(_TILING_TABLES))
def test_tiling_table_invariants(key):
    """Every registered table is pow2 / nested / non-increasing with strictly descending floors."""
    _profile, collective, bucket = key
    group = 4 if bucket == "small" else 32
    per_rank = collective == "allreduce"
    validate_table(
        _TILING_TABLES[key], num_ranks=group, n_prgs=2, dtype_size=_DTYPE_SIZE, per_rank=per_rank
    )


# Dense numel sweep hitting every pow2 tile boundary and the 2x/4x unroll thresholds, plus the
# irregular production sizes -- enough to reach every (trace, unroll) the dispatch can produce.
_COVERAGE_NUMELS = sorted(
    set(range(1, 64))
    | {n for k in range(3, 29) for n in (2**k - 1, 2**k, 2**k + 1)}
    | {int(f * 2 ** (k - 1)) for k in range(3, 29) for f in (1.5, 2.5, 3.5, 5, 6, 7)}
    | set(_NUMELS)
)
_COLLECTIVES = ["allgather", "allreduce"]


@pytest.mark.parametrize("collective", _COLLECTIVES)
@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("nprg", _NPRGS)
def test_precompile_buckets_cover_all_dispatch(collective, platform, world, nprg):
    """The precompile enumeration (`_enumerate_buckets`) must be a SUPERSET of every
    (trace_numel, large, medium, small, unroll) bucket the runtime dispatch produces across the
    full numel range. If any dispatch bucket is missing, that shape recompiles at runtime instead
    of hitting the warmed NEFF cache -- the exact regression B4 fixes."""
    if world % nprg != 0:
        pytest.skip("world must be divisible by n_prgs")
    plan_fn = plan_dynamic_allgather if collective == "allgather" else plan_dynamic_allreduce
    enumerated = {
        (b["trace_numel"], b["large"], b["medium"], b["small"], b["unroll"])
        for b in _enumerate_buckets(world, nprg, _DTYPE_SIZE, platform, collective)
    }
    for numel in _COVERAGE_NUMELS:
        plan = plan_fn(numel, world, nprg, _DTYPE_SIZE, platform)
        if plan is None:
            continue  # below the eligibility floor -> native fallback, no NEFF
        key = (plan["large"], plan["medium"], plan["small"])
        bucket = (plan["trace_numel"], *key, plan["unroll"])
        assert bucket in enumerated, (
            f"dispatch bucket {bucket} (numel={numel}, {collective}, {platform}, world={world}, "
            f"nprg={nprg}) is not in the precompile set -> would recompile at runtime; "
            f"enumerated={sorted(enumerated)}"
        )


@pytest.mark.parametrize("collective", _COLLECTIVES)
@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("nprg", _NPRGS)
def test_enumerate_buckets_wellformed(collective, platform, world, nprg):
    """Each enumerated bucket is self-consistent: pow2 non-increasing tiles, trace is one of the
    three tiles, unroll in {1,2,4}, and the (dedup) set is non-empty."""
    if world % nprg != 0:
        pytest.skip("world must be divisible by n_prgs")
    buckets = _enumerate_buckets(world, nprg, _DTYPE_SIZE, platform, collective)
    assert buckets, f"no buckets enumerated for {collective}/{platform}/world={world}"
    seen = set()
    for b in buckets:
        large, medium, small = b["large"], b["medium"], b["small"]
        for t in (large, medium, small):
            assert t & (t - 1) == 0 and t > 0, f"tile {t} not a positive pow2"
        assert large >= medium >= small, f"tiles not non-increasing: {large}/{medium}/{small}"
        assert b["trace_numel"] in (large, medium, small), f"trace {b['trace_numel']} not a tile"
        assert b["unroll"] in (1, 2, 4), f"bad unroll {b['unroll']}"
        key = (b["trace_numel"], large, medium, small, b["unroll"])
        assert key not in seen, f"duplicate bucket {key}"
        seen.add(key)
