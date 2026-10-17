"""Perf-regression guard for prologue/epilogue fusion on in-NEFF collectives.

The feature (feat/collective-prologue-epilogue) absorbs a non-contiguous
(transposed) collective tensor's layout transform into the op's MLIR — a
"prologue" for inputs, an "epilogue" for outputs — instead of materializing a
contiguous copy via a CPU roundtrip. This test times each collective primitive
driven with a NON-contiguous (transposed) tensor and asserts the median
latency stays under a per-shape ceiling.

Why a ceiling instead of an ON-vs-OFF speedup:
  TORCH_NEURONX_ENABLE_PROLOGUE / _ENABLE_EPILOGUE are read once and cached in
  C++ static init (before this pytest worker imports torch_neuronx), so a
  fusion-OFF baseline can't be produced in-process. But fusion is ON by default,
  so we simply bound the ON latency. If fusion regresses it falls back to the
  CPU-roundtrip path (the measured fusion-OFF column below), which is 2.06x-97x
  slower and blows past every ceiling.

Ceiling = measured fusion-ON median x _MARGIN. The smallest measured speedup is
2.06x (all_to_all 256x256), so any _MARGIN < 2.06 keeps the ceiling strictly
between the ON and OFF latency for every shape: a healthy run passes, a full
fusion regression always trips it. _MARGIN=1.2 leaves headroom for run-to-run
jitter on the fast (sub-ms) small shapes while staying well under that bound;
raise it toward (but below) 2.0 if a tiny shape proves flaky on shared CI.

Measured medians (ms), world_size=4, from bench_collective_fusion:
    primitive       shape       OFF       ON     speedup
    all_gather      256x256     0.438    0.188     2.33x
    all_gather      1024x1024   3.593    0.307    11.70x
    all_gather      4096x4096  88.771    2.873    30.90x
    all_gather      8192x4096 184.362    6.629    27.81x
    all_reduce      256x256     0.646    0.199     3.25x
    all_reduce      1024x1024   5.520    0.275    20.07x
    all_reduce      4096x4096  94.190    1.423    66.21x
    all_reduce      8192x4096 191.602    2.687    71.32x
    all_to_all      256x256     1.258    0.610     2.06x
    all_to_all      1024x1024  11.426    0.912    12.53x
    all_to_all      4096x4096 330.749    6.982    47.37x
    all_to_all      8192x4096 671.938   13.500    49.77x
    reduce_scatter  256x256     0.432    0.191     2.27x
    reduce_scatter  1024x1024   3.511    0.227    15.45x
    reduce_scatter  4096x4096  86.886    0.974    89.25x
    reduce_scatter  8192x4096 177.862    1.840    96.68x

Non-contiguity is produced via `.t()` (a transposed view of the full buffer) —
a collective needs all of its data, so a transposed full tensor is the right
non-contiguous representative.
"""

import time

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest

# ─────────────────────────────────────────────────────────────────────────
# latency ceilings
# ─────────────────────────────────────────────────────────────────────────

# Ceiling multiplier over the measured fusion-ON median. Must stay below the
# smallest measured speedup (2.06x) so every ceiling sits under the fusion-OFF
# latency. See module docstring.
_MARGIN = 1.2

# Measured fusion-ON median latency (ms) per "{primitive}|{M}x{N}".
_FUSION_ON_MS = {
    "all_gather|256x256": 0.188,
    "all_gather|1024x1024": 0.307,
    "all_gather|4096x4096": 2.873,
    "all_gather|8192x4096": 6.629,
    "all_reduce|256x256": 0.199,
    "all_reduce|1024x1024": 0.275,
    "all_reduce|4096x4096": 1.423,
    "all_reduce|8192x4096": 2.687,
    "all_to_all|256x256": 0.610,
    "all_to_all|1024x1024": 0.912,
    "all_to_all|4096x4096": 6.982,
    "all_to_all|8192x4096": 13.500,
    "reduce_scatter|256x256": 0.191,
    "reduce_scatter|1024x1024": 0.227,
    "reduce_scatter|4096x4096": 0.974,
    "reduce_scatter|8192x4096": 1.840,
}

_UPPER_BOUND_MS = {key: ms * _MARGIN for key, ms in _FUSION_ON_MS.items()}

# Timing budget per (primitive, shape). Warmup must cover the one-time NEFF
# compilation on the first call; the median over `_ITERS` shrugs off jitter.
_WARMUP = 10
_ITERS = 30


# ─────────────────────────────────────────────────────────────────────────
# timing + assertion
# ─────────────────────────────────────────────────────────────────────────


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _time_median_ms(run):
    """Median per-iter latency in ms. Each loop is device-synchronized so the
    recorded time is that single collective's end-to-end latency."""
    for _ in range(_WARMUP):
        run()
    torch.neuron.synchronize()

    samples = []
    for _ in range(_ITERS):
        start = time.perf_counter()
        run()
        torch.neuron.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return _median(samples)


def _assert_under_ceiling(prim, rank, m, n, run):
    """Time `run` and assert its median latency is under the per-shape ceiling.

    Raised on any rank; DistributedTester surfaces the lowest-rank failure."""
    key = f"{prim}|{m}x{n}"
    ceiling = _UPPER_BOUND_MS[key]
    median_ms = _time_median_ms(run)
    assert median_ms <= ceiling, (
        f"Rank {rank}: {key} median latency {median_ms:.3f}ms exceeded ceiling "
        f"{ceiling:.3f}ms (= {_FUSION_ON_MS[key]:.3f}ms x {_MARGIN}). "
        f"Prologue/epilogue fusion may have regressed to the CPU-roundtrip path."
    )


# ─────────────────────────────────────────────────────────────────────────
# per-primitive workloads — each builds a strided (transposed) tensor and
# times exactly one collective, mirroring bench_collective_fusion builders.
# ─────────────────────────────────────────────────────────────────────────


def run_all_reduce_perf(rank, world_size, kwargs):
    m, n = kwargs["m"], kwargs["n"]
    tensor = torch.randn(m, n, dtype=torch.float32).to("neuron").t()
    assert not tensor.is_contiguous()

    def run():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    _assert_under_ceiling("all_reduce", rank, m, n, run)


def run_all_gather_perf(rank, world_size, kwargs):
    m, n = kwargs["m"], kwargs["n"]
    inp = torch.randn(m, n, dtype=torch.float32).to("neuron").t()  # (n, m), strided
    assert not inp.is_contiguous()
    rows, cols = inp.shape
    output = torch.empty(world_size * rows, cols, dtype=torch.float32).to("neuron")

    def run():
        dist.all_gather_into_tensor(output, inp)

    _assert_under_ceiling("all_gather", rank, m, n, run)


def run_reduce_scatter_perf(rank, world_size, kwargs):
    # reduce_scatter splits dim 0 of the (transposed) input across ranks, so
    # dim 0 of the view must be divisible by world_size.
    m, n = kwargs["m"], kwargs["n"]
    inp = torch.randn(m, n, dtype=torch.float32).to("neuron").t()  # (n, m), strided
    assert not inp.is_contiguous()
    rows, cols = inp.shape
    assert rows % world_size == 0, f"input dim0={rows} not divisible by world_size={world_size}"
    output = torch.empty(rows // world_size, cols, dtype=torch.float32).to("neuron")

    def run():
        dist.reduce_scatter_tensor(output, inp, op=dist.ReduceOp.SUM)

    _assert_under_ceiling("reduce_scatter", rank, m, n, run)


def run_all_to_all_perf(rank, world_size, kwargs):
    m, n = kwargs["m"], kwargs["n"]
    inputs = [torch.randn(m, n, dtype=torch.float32).to("neuron").t() for _ in range(world_size)]
    assert not inputs[0].is_contiguous()
    out_shape = inputs[0].shape
    outputs = [torch.empty(out_shape, dtype=torch.float32).to("neuron") for _ in range(world_size)]

    def run():
        dist.all_to_all(outputs, inputs)

    _assert_under_ceiling("all_to_all", rank, m, n, run)


# ─────────────────────────────────────────────────────────────────────────
# test class — world_size 4 (all_to_all requires 4/8/16/mult-of-32; the
# measured ceilings above are all from a single world_size=4 bench run).
# ─────────────────────────────────────────────────────────────────────────

_SHAPES = [(256, 256), (1024, 1024), (4096, 4096), (8192, 4096)]
_SHAPE_IDS = [f"{m}x{n}" for m, n in _SHAPES]


class TestStridedCollectivesPerf(BaseCollectiveOpTest):
    """Latency ceiling for fused non-contiguous collectives (world_size 4)."""

    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.parametrize("m,n", _SHAPES, ids=_SHAPE_IDS)
    def test_all_reduce_perf(self, m, n):
        self.distributed_tester.run_test(run_all_reduce_perf, m=m, n=n)

    @pytest.mark.parametrize("m,n", _SHAPES, ids=_SHAPE_IDS)
    def test_all_gather_perf(self, m, n):
        self.distributed_tester.run_test(run_all_gather_perf, m=m, n=n)

    @pytest.mark.parametrize("m,n", _SHAPES, ids=_SHAPE_IDS)
    def test_reduce_scatter_perf(self, m, n):
        self.distributed_tester.run_test(run_reduce_scatter_perf, m=m, n=n)

    @pytest.mark.parametrize("m,n", _SHAPES, ids=_SHAPE_IDS)
    def test_all_to_all_perf(self, m, n):
        self.distributed_tester.run_test(run_all_to_all_perf, m=m, n=n)
