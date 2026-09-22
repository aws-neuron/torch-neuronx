"""Tests for the dynamic NKI AllGather."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.utils.neuron_test_utils import assert_raises

_W = 4096  # fixed shard width; the flat tiling keys on numel = h*_W, so this pins each h's bucket


def _make_shard(global_rank, h, w, dtype):
    """Deterministic per-rank shard (same seed reproduces the reference)."""
    torch.manual_seed(7919 * (global_rank + 1) + h + w)
    if dtype.is_floating_point:
        return torch.randn(h, w, dtype=dtype)
    return torch.randint(-128, 127, (h, w), dtype=dtype)


def _gather_check(rank, ranks, h, w, dtype, *, group=None, input_3d=False, contiguous_out=True):
    """all_gather_into_tensor over ranks, assert every slice is bit-exact vs ref."""
    n = len(ranks)
    shard = _make_shard(rank, h, w, dtype)
    inp = (shard.reshape(1, h, w) if input_3d else shard).to("neuron")

    if contiguous_out:
        out = torch.empty(n * h, w, dtype=dtype, device="neuron")
    else:
        # Column-major view is non-contiguous, forces the static kernel fallback.
        out = torch.empty(w, n * h, dtype=dtype, device="neuron").t()

    dist.all_gather_into_tensor(out, inp, group=group)
    torch.neuron.synchronize()

    out_cpu = out.cpu().reshape(n * h, w)
    for j, gr in enumerate(ranks):
        expected = _make_shard(gr, h, w, dtype)
        assert torch.equal(
            out_cpu[j * h : (j + 1) * h], expected
        ), f"[rank {rank}] slice {j} (global {gr}) mismatch: h={h} w={w} dtype={dtype}"


def run_bucket_correctness(rank, world_size, kw):
    _gather_check(rank, list(range(world_size)), kw["h"], _W, kw["dtype"])


def run_misaligned_runtime_offsets(rank, world_size, kw):
    # At world=2, bf16 regime-A large_tile=2**23. This gives two large tiles plus a
    # one-element tail, and the second tile's rank-1 offset is above fp32's exact-integer range.
    _gather_check(rank, list(range(world_size)), 1, 2**24 + 1, kw["dtype"])


def run_dtype(rank, world_size, kw):
    _gather_check(rank, list(range(world_size)), 512, _W, kw["dtype"])


def run_3d_input(rank, world_size, kw):
    _gather_check(rank, list(range(world_size)), 256, _W, kw["dtype"], input_3d=True)


def run_below_floor_fallback(rank, world_size, kw):
    # numel = 2*8 = 16 is below the small-tile floor for any world, always static fallback.
    _gather_check(rank, list(range(world_size)), 2, 8, kw["dtype"])


def run_noncontiguous_fallback(rank, world_size, kw):
    _gather_check(rank, list(range(world_size)), 512, _W, kw["dtype"], contiguous_out=False)


def run_dynamic_reuse(rank, world_size, kw):
    # Single-NEFF assertion: lower_dynamic_allgather runs once per C++ dispatch-cache miss
    import torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives as _dc
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    dtype = kw["dtype"]
    ds = torch.empty((), dtype=dtype).element_size()
    lnc = int(get_logical_neuron_cores())
    platform = get_platform_target()
    # Regime A is single-rung (floor == large tile), so any huge numel traces at large_tile.
    probe = _dc.plan_dynamic_allgather(1 << 30, world_size, lnc, ds, platform)
    assert probe is not None, f"dynamic AG ineligible for world={world_size}, lnc={lnc}"
    large = probe["large"]
    # Smallest h with numel = h*_W >= 4*large_tile (unroll saturates at 4); +1/+2 stay in the same
    # (trace_numel, unroll==4) bucket but are different runtime numels. Gathered size is
    # ~4*large_tile_gathered, world-independent.
    base_h = (4 * large + _W - 1) // _W  # ceil(4*large / _W)
    seqs = [base_h, base_h + 1, base_h + 2]

    calls = []
    _orig_lower = _dc.lower_dynamic_allgather

    def _counting_lower(inputs, replica_groups):
        calls.append(1)
        return _orig_lower(inputs, replica_groups)

    _dc.lower_dynamic_allgather = _counting_lower
    try:
        for h in seqs:
            _gather_check(rank, list(range(world_size)), h, _W, dtype)
    finally:
        _dc.lower_dynamic_allgather = _orig_lower

    assert len(calls) == 1, (
        f"expected exactly 1 NEFF lowering across seqs={seqs} (single (trace_numel, unroll) "
        f"bucket at world={world_size}), got {len(calls)} — 0 means the dynamic path never ran "
        f"(silent fallback); >1 means the shapes did not share one NEFF (bucketing regression)"
    )


def run_async(rank, world_size, kw):
    h, dtype = 512, kw["dtype"]
    shard = _make_shard(rank, h, _W, dtype).to("neuron")
    out = torch.empty(world_size * h, _W, dtype=dtype, device="neuron")
    work = dist.all_gather_into_tensor(out, shard, async_op=True)
    assert work is not None
    work.wait()
    out_cpu = out.cpu()
    for j in range(world_size):
        assert torch.equal(out_cpu[j * h : (j + 1) * h], _make_shard(j, h, _W, dtype))


def run_subgroup(rank, world_size, kw):
    # Two groups of world_size/2: exercises the full-partition scoping the dynamic-allgather
    # kernel reconstructs for ncc.all_gather.
    half = world_size // 2
    g0 = dist.new_group(ranks=list(range(half)))
    g1 = dist.new_group(ranks=list(range(half, world_size)))
    my_ranks = list(range(half)) if rank < half else list(range(half, world_size))
    my_group = g0 if rank < half else g1
    _gather_check(rank, my_ranks, kw["h"], _W, kw["dtype"], group=my_group)


def run_wrong_output_size(rank, world_size, kw):
    h, dtype = 256, kw["dtype"]
    inp = _make_shard(rank, h, _W, dtype).to("neuron")
    bad_out = torch.empty(world_size * h - 1, _W, dtype=dtype, device="neuron")  # off by one
    dist.all_gather_into_tensor(bad_out, inp)
    torch.neuron.synchronize()


class TestDynamicAllGather(BaseCollectiveOpTest):
    """Dynamic NKI AllGather via the c10d backend dynamic-allgather path."""

    @property
    def world_size(self):
        # Default this suite to world=4 (still overridable via WORLD_SIZE); the base
        # class defaults to 2. world=4 also enables test_subgroup_partition (needs >=4).
        return int(os.environ.get("WORLD_SIZE", 4))

    @pytest.fixture(autouse=True)
    def _dynamic_env(self):
        """Enable the dynamic NKI AllGather + its compile/runtime flags for this
        test only
          * NEURON_DYN_NKI_COLLECTIVES=1                     — enable the C++ dynamic CC dispatch
          * NEURON_CC_FLAGS += grow flags                    — let the floor-traced NEFF grow
          * NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE=1  — enable CC control flow
        """
        _flags = ("--disable-internal-gather-scatter-bound-check",)
        keys = (
            "NEURON_DYN_NKI_COLLECTIVES",
            "NEURON_CC_FLAGS",
            "NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE",
        )
        saved = {k: os.environ.get(k) for k in keys}

        os.environ["NEURON_DYN_NKI_COLLECTIVES"] = "1"
        os.environ["NEURON_RT_EXPERIMENTAL_CC_CONTROL_FLOW_ENABLE"] = "1"
        cc = os.environ.get("NEURON_CC_FLAGS", "")
        for tok in _flags:
            if tok not in cc:
                cc = (cc + " " + tok).strip()
        os.environ["NEURON_CC_FLAGS"] = cc

        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @pytest.mark.parametrize("h", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    def test_bucket_correctness(self, h):
        self.distributed_tester.run_test(run_bucket_correctness, h=h, dtype=torch.bfloat16)

    def test_misaligned_runtime_offsets(self):
        self.distributed_tester.run_test(run_misaligned_runtime_offsets, dtype=torch.bfloat16)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
    def test_dtypes(self, dtype):
        self.distributed_tester.run_test(run_dtype, dtype=dtype)

    def test_3d_input(self):
        self.distributed_tester.run_test(run_3d_input, dtype=torch.bfloat16)

    def test_below_floor_fallback(self):
        self.distributed_tester.run_test(run_below_floor_fallback, dtype=torch.bfloat16)

    def test_noncontiguous_output_fallback(self):
        self.distributed_tester.run_test(run_noncontiguous_fallback, dtype=torch.bfloat16)

    def test_dynamic_reuse_same_bucket(self):
        # run_dynamic_reuse derives world-robust seqs sharing one (trace_numel, unroll) bucket.
        self.distributed_tester.run_test(run_dynamic_reuse, dtype=torch.bfloat16)

    def test_async(self):
        self.distributed_tester.run_test(run_async, dtype=torch.bfloat16)

    def test_subgroup_partition(self):
        if self.world_size < 4 or self.world_size % 2 != 0:
            pytest.skip("subgroup scoping test needs an even world_size >= 4")
        self.distributed_tester.run_test(run_subgroup, h=512, dtype=torch.bfloat16)

    @assert_raises(RuntimeError, match=r".*output shape mismatch.*")
    def test_wrong_output_size_errors(self):
        self.distributed_tester.run_test(run_wrong_output_size, dtype=torch.bfloat16)
