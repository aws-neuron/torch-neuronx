"""Parity test: the C++ tiling (ag_tiling::ComputePlan, exposed via
torch_neuronx._C._allgather_compute_plan) must match the Python reference
dynamic_allgather.plan_dynamic_allgather for every (numel, world, nprg, platform), plus a
structural check that every registered tiling table satisfies its invariants.
"""

import pytest

from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives import (
    _TILING_TABLES,
    plan_dynamic_allgather,
    regime_tiles,
)


def validate_table(table, num_ranks=8, n_prgs=2, dtype_size=2):
    """Assert a table's structural invariants for a representative (num_ranks, n_prgs, dtype)"""
    prev_floor = None
    for name, floor, targets in table:
        large, medium, small = regime_tiles(num_ranks, n_prgs, targets, dtype_size)
        for t in (large, medium, small):
            assert t & (t - 1) == 0, f"{name}: tile {t} not pow2"
        assert large >= medium >= small, f"{name}: not non-increasing {large}/{medium}/{small}"
        assert large % medium == 0 and medium % small == 0, f"{name}: tiles not nested"
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
    65536,  # ~ regime F/E boundary at world=4/small-table (gathered 512KB)
    131072,  # gathered 1MB
    200000,
    262144,  # gathered 2MB
    524288,  # gathered 4MB
    1048576,  # gathered 8MB
    2_000_000,
    2_097_152,  # gathered 16MB
    3_000_000,
    4_194_304,  # gathered 32MB (small-table B/A boundary)
    2**25,
    5_000_000,
    9_723_904,
    2**26,  # exercises the large-table upper regimes at world>=16
    50_000_000,
]
# world <= 8 exercises the "small" table; world > 8 exercises the "large" table.
_WORLDS = [2, 4, 8, 16, 32]
_NPRGS = [1, 2]
_PLATFORMS = ["trn2", "trn3"]  # both map to the trn2 tables (exercises the fallback/classifier)
_DTYPE_SIZE = 2  # bf16


def _cpp_plan(numel, world, nprg, platform):
    from torch_neuronx import _C

    return _C._allgather_compute_plan(
        numel, world, nprg, _DTYPE_SIZE, platform
    )  # (large, medium, small, trace_numel, eligible, unroll)


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("world", _WORLDS)
@pytest.mark.parametrize("nprg", _NPRGS)
def test_allgather_plan_parity(world, nprg, platform):
    for numel in _NUMELS:
        cpp = _cpp_plan(numel, world, nprg, platform)
        large, medium, small, trace_numel, eligible, unroll = cpp
        py = plan_dynamic_allgather(numel, world, nprg, _DTYPE_SIZE, platform)
        ctx = f"numel={numel} world={world} nprg={nprg} plat={platform}: cpp={cpp} py={py}"
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


@pytest.mark.parametrize("key", sorted(_TILING_TABLES))
def test_tiling_table_invariants(key):
    """Every registered table is pow2/nested/descending with disjoint trace rungs."""
    group = 4 if key[1] == "small" else 32
    validate_table(_TILING_TABLES[key], num_ranks=group, n_prgs=2, dtype_size=_DTYPE_SIZE)
