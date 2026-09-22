"""Tests for the dynamic NKI AllReduce."""

import os
import re

import pytest
import torch
import torch.distributed as dist

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest

_W = 4096  # fixed width; the flat tiling keys on numel = h*_W, so this pins each h's bucket

_DIST_OP = {"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}


def _ar_shard(global_rank, h, w, dtype):
    """Deterministic per-rank shard of small INTEGER values. Integers in [-4, 4] are exactly
    representable in bf16/fp16/fp32, and any SUM/MIN/MAX across up to 64 ranks stays exact
    (|sum| <= 256), so the reduction is bit-exact regardless of dtype. This tests the tiling /
    offset / tail plumbing, not CCE floating-point accumulation order."""
    torch.manual_seed(7919 * (global_rank + 1) + h + w)
    return torch.randint(-4, 5, (h, w)).to(dtype)


def _reduce_ref(stacked, op_name):
    """Reference reduction over the rank axis (dim 0)."""
    if op_name == "sum":
        return stacked.sum(0)
    if op_name == "min":
        return stacked.amin(0)
    return stacked.amax(0)


def _allreduce_check(rank, ranks, h, w, dtype, op_name="sum", *, group=None, contiguous=True):
    """all_reduce over `ranks`, assert the in-place result is bit-exact vs the reference."""
    shard = _ar_shard(rank, h, w, dtype)
    # contiguous -> dynamic path; non-contiguous [h, w] view (transpose of a contiguous [w, h])
    # forces the static fallback.
    t = shard.clone().to("neuron") if contiguous else shard.to("neuron").t().contiguous().t()

    dist.all_reduce(t, op=_DIST_OP[op_name], group=group)
    torch.neuron.synchronize()

    got = t.cpu().reshape(h, w)
    stacked = torch.stack([_ar_shard(gr, h, w, dtype).to(torch.float32) for gr in ranks])
    expected = _reduce_ref(stacked, op_name).to(dtype)
    assert torch.equal(
        got, expected
    ), f"[rank {rank}] allreduce {op_name} mismatch: h={h} w={w} dtype={dtype} n={len(ranks)}"


def run_bucket_correctness(rank, world_size, kw):
    _allreduce_check(rank, list(range(world_size)), kw["h"], _W, kw["dtype"])


def run_reduce_op(rank, world_size, kw):
    _allreduce_check(
        rank, list(range(world_size)), 512, _W, kw["dtype"], op_name=kw.get("op_name", "sum")
    )


def run_misaligned_runtime_offsets(rank, world_size, kw):
    # numel = 2**24 + 1 leaves a one-element remainder. The forward tail starts at that true
    # remainder and uses OOB-skipped DMA for the fixed-size padding. The large offset also exceeds
    # fp32's exact-integer range, exercising int32 address arithmetic.
    _allreduce_check(rank, list(range(world_size)), 1, 2**24 + 1, kw["dtype"])


def run_misaligned_tail(rank, world_size, kw):
    # For every supported dtype, 2**18+1 is eligible and misaligned to its selected small tile.
    # This directly exercises identity padding plus OOB-skipped load/store for SUM/MIN/MAX.
    _allreduce_check(
        rank,
        list(range(world_size)),
        1,
        2**18 + 1,
        kw["dtype"],
        op_name=kw.get("op_name", "sum"),
    )


def run_below_floor_fallback(rank, world_size, kw):
    # numel = 2*8 = 16 is below the smallest tile for any world -> static fallback.
    _allreduce_check(rank, list(range(world_size)), 2, 8, kw["dtype"])


def run_noncontiguous_fallback(rank, world_size, kw):
    _allreduce_check(rank, list(range(world_size)), 512, _W, kw["dtype"], contiguous=False)


def run_ar_reuse(rank, world_size, kw):
    # Single-NEFF assertion: lower_dynamic_allreduce runs once per C++ dispatch-cache miss. Derive
    # world-robust seqs sharing one (trace_numel, unroll) bucket (regime A is single-rung, so any
    # numel >= 4*large traces at large_tile with unroll saturated at 4).
    import torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives as _dc
    from torch_neuronx.utils import get_logical_neuron_cores, get_platform_target

    dtype = kw["dtype"]
    ds = torch.empty((), dtype=dtype).element_size()
    lnc = int(get_logical_neuron_cores())
    platform = get_platform_target()
    probe = _dc.plan_dynamic_allreduce(1 << 30, world_size, lnc, ds, platform)
    assert probe is not None, f"dynamic AR ineligible for world={world_size}, lnc={lnc}"
    large = probe["large"]
    base_h = (4 * large + _W - 1) // _W  # ceil(4*large / _W); numel >= 4*large => unroll==4
    seqs = [base_h, base_h + 1, base_h + 2]

    calls = []
    physical_aliases = []
    output_counts = []
    stablehlo_has_alias = []
    _orig_lower = _dc.lower_dynamic_allreduce

    def _counting_lower(inputs, replica_groups, reduce_op):
        calls.append(1)
        result = _orig_lower(inputs, replica_groups, reduce_op)
        physical_aliases.append(dict(result.nki_operand_output_aliases))
        output_counts.append(len(result.nki_return_types))
        stablehlo = result.stablehlo_bytes.decode()
        stablehlo_has_alias.append(
            bool(
                re.search(
                    r"output_operand_aliases = \[#stablehlo\.output_operand_alias<"
                    r"output_tuple_indices = \[\], operand_index = 0, "
                    r"operand_tuple_indices = \[\]>\]",
                    stablehlo,
                )
            )
        )
        return result

    _dc.lower_dynamic_allreduce = _counting_lower
    try:
        for h in seqs:
            _allreduce_check(rank, list(range(world_size)), h, _W, dtype)
    finally:
        _dc.lower_dynamic_allreduce = _orig_lower

    assert len(calls) == 1, (
        f"expected exactly 1 NEFF lowering across seqs={seqs} (single (trace_numel, unroll) "
        f"bucket at world={world_size}), got {len(calls)} — 0 means the dynamic path never ran "
        f"(silent fallback); >1 means the shapes did not share one NEFF (bucketing regression)"
    )
    assert output_counts == [
        1
    ], f"expected one ordinary NKI output, got output counts {output_counts}"
    assert physical_aliases == [
        {}
    ], f"expected no physical NKI must-alias outputs, got {physical_aliases}"
    assert stablehlo_has_alias == [False], (
        "expected no StableHLO custom-call output/operand alias for the functional NKI kernel, "
        f"got matches {stablehlo_has_alias}"
    )


def run_async(rank, world_size, kw):
    h, dtype = 512, kw["dtype"]
    shard = _ar_shard(rank, h, _W, dtype)
    t = shard.clone().to("neuron")
    work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    assert work is not None
    work.wait()
    stacked = torch.stack([_ar_shard(j, h, _W, dtype).to(torch.float32) for j in range(world_size)])
    expected = stacked.sum(0).to(dtype)
    assert torch.equal(t.cpu(), expected)


def run_subgroup(rank, world_size, kw):
    # Two groups of world_size/2: each rank reduces only within its group.
    half = world_size // 2
    g0 = dist.new_group(ranks=list(range(half)))
    g1 = dist.new_group(ranks=list(range(half, world_size)))
    my_ranks = list(range(half)) if rank < half else list(range(half, world_size))
    my_group = g0 if rank < half else g1
    _allreduce_check(rank, my_ranks, kw["h"], _W, kw["dtype"], group=my_group)


def run_coalesced_decompose(rank, world_size, kw):
    # Under NEURON_DYN_NKI_COLLECTIVES, all_reduce_coalesced decomposes into per-tensor
    # all-reduces (eligible -> dynamic single-tensor kernel, sub-floor -> native). Verify the
    # batch still reduces correctly element-for-element across a mix of eligible + sub-floor
    # tensors, guarding the per-tensor decomposition dispatch in the backend. The shapes are
    # distinct, so each tensor's seed (keyed on h+w) already gives it distinct data -- a wrong
    # or skipped tensor in the decomposition can't hide.
    dtype = kw["dtype"]
    ranks = list(range(world_size))
    shapes = [(1024, 2048), (512, 300), (7, 129)]  # 2097152 + 153600 (dynamic) + 903 (native)
    tensors = [_ar_shard(rank, h, w, dtype).clone().to("neuron") for h, w in shapes]
    dist.all_reduce_coalesced(tensors, op=_DIST_OP["sum"])
    torch.neuron.synchronize()
    for i, (h, w) in enumerate(shapes):
        got = tensors[i].cpu().reshape(h, w)
        stacked = torch.stack([_ar_shard(gr, h, w, dtype).to(torch.float32) for gr in ranks])
        expected = _reduce_ref(stacked, "sum").to(dtype)
        assert torch.equal(
            got, expected
        ), f"[rank {rank}] coalesced-decompose tensor {i} shape=({h},{w}) mismatch"


class TestDynamicAllReduce(BaseCollectiveOpTest):
    """Dynamic NKI AllReduce via the c10d backend dynamic-collective path."""

    @property
    def world_size(self):
        # Default to world=4 (overridable via WORLD_SIZE); also enables test_subgroup_partition.
        return int(os.environ.get("WORLD_SIZE", 4))

    @pytest.fixture(autouse=True)
    def _dynamic_env(self):
        """Enable the dynamic NKI collectives + compile/runtime flags for this test only:
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

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("op_name", ["sum", "min", "max"])
    def test_reduce_ops(self, op_name, dtype):
        # op x dtype cross at an eligible shape: every (reduce op, dtype) the kernel supports,
        # including fp32 MAX (a captured torchtitan reduce). Subsumes the old sum-only dtype sweep.
        self.distributed_tester.run_test(run_reduce_op, op_name=op_name, dtype=dtype)

    def test_misaligned_runtime_offsets(self):
        self.distributed_tester.run_test(run_misaligned_runtime_offsets, dtype=torch.bfloat16)

    @pytest.mark.parametrize("op_name", ["sum", "min", "max"])
    def test_misaligned_tail_reduce_ops(self, op_name):
        self.distributed_tester.run_test(run_misaligned_tail, dtype=torch.bfloat16, op_name=op_name)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
    def test_misaligned_tail_dtypes(self, dtype):
        self.distributed_tester.run_test(run_misaligned_tail, dtype=dtype)

    def test_below_floor_fallback(self):
        self.distributed_tester.run_test(run_below_floor_fallback, dtype=torch.bfloat16)

    def test_noncontiguous_fallback(self):
        self.distributed_tester.run_test(run_noncontiguous_fallback, dtype=torch.bfloat16)

    def test_dynamic_reuse_same_bucket(self):
        self.distributed_tester.run_test(run_ar_reuse, dtype=torch.bfloat16)

    def test_async(self):
        self.distributed_tester.run_test(run_async, dtype=torch.bfloat16)

    def test_subgroup_partition(self):
        if self.world_size < 4 or self.world_size % 2 != 0:
            pytest.skip("subgroup test needs an even world_size >= 4")
        self.distributed_tester.run_test(run_subgroup, h=512, dtype=torch.bfloat16)

    def test_coalesced_decompose(self):
        # all_reduce_coalesced under NEURON_DYN_NKI_COLLECTIVES decomposes into per-tensor
        # all-reduces (eligible -> dynamic, sub-floor -> native) and stays element-exact.
        self.distributed_tester.run_test(run_coalesced_decompose, dtype=torch.bfloat16)
