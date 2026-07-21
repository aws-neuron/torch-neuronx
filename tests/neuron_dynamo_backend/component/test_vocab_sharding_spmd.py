"""
Tests for VocabShardingSPMD and ShardIndexInjection FX passes.

These passes enable SPMD compilation for vocab-sharded models by rewriting
per-rank constants to expressions of a vocab_shard_index placeholder.

Background: DTensor's RowwiseParallel shards the embedding vocabulary across
TP ranks. Each rank gets a shard of the vocab (e.g., rank 0 gets tokens
0-4095, rank 1 gets 4096-8191). This produces three patterns in the FX graph
with rank-specific constants that prevent SPMD (single shared NEFF):

  Triad pattern (forward + loss graphs):
    lt.Scalar(tokens, LOWER)  — mask tokens below this rank's shard
    ge.Scalar(tokens, UPPER)  — mask tokens above this rank's shard
    bitwise_or(lt, ge)        — combined out-of-range mask
    sub.Tensor(tokens, LOWER) — shift token IDs to local shard indices
    where UPPER - LOWER = shard_size, LOWER = rank * shard_size

  Flat-offset add pattern (backward graph):
    add.Tensor(flat_indices, OFFSET) → scatter_add
    where OFFSET = rank * shard_size * hidden_dim
    This computes global indices for the embedding gradient scatter.

  Backward slice pattern (backward graph):
    slice.Tensor(grad, 0, START, END)
    where START = rank * shard_size, END = START + shard_size
    DTensor slices the full vocab gradient to extract this rank's shard.

ShardIndexInjection infers the vocab shard index from these constants
(LOWER // shard_size) and injects a vocab_shard_index placeholder.
VocabShardingSPMD rewrites the constants to expressions of that placeholder.
"""

from unittest.mock import MagicMock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule, Node

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState
from torch_neuronx.neuron_dynamo_backend.fx.passes.vocab_sharding_spmd import (
    BackwardSlice,
    FlatOffsetAdd,
    ShardIndexInjection,
    VocabShardingSPMD,
    VocabTriad,
    find_backward_slices,
    find_flat_offset_adds,
    find_vocab_triads,
)

# ============================================================
# Helpers
# ============================================================


def _make_state():
    """Create a NeuronFXPassState for testing mutation recording."""
    return NeuronFXPassState(job=MagicMock(), model_name="test", dumping_enabled=False)


# ============================================================
# Graph builders
# ============================================================


def _meta(shape, dtype):
    with FakeTensorMode():
        return torch.empty(shape, dtype=dtype)


def _build_single_triad(rank: int, tp_size: int = 4, vocab_size: int = 16384):
    """Standard RowwiseParallel embedding graph for one vocab shard index."""
    shard_size = vocab_size // tp_size
    offset = rank * shard_size
    g = Graph()
    tokens = g.placeholder("tokens")
    tokens.meta["val"] = _meta((1, 4096), torch.int64)
    emb = g.placeholder("emb_table")
    emb.meta["val"] = _meta((shard_size, 128), torch.bfloat16)

    lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, offset))
    lt.meta["val"] = _meta((1, 4096), torch.bool)
    ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, offset + shard_size))
    ge.meta["val"] = _meta((1, 4096), torch.bool)
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    bor.meta["val"] = _meta((1, 4096), torch.bool)
    clone = g.call_function(torch.ops.aten.clone.default, (tokens,))
    clone.meta["val"] = _meta((1, 4096), torch.int64)
    sub = g.call_function(torch.ops.aten.sub.Tensor, (clone, offset))
    sub.meta["val"] = _meta((1, 4096), torch.int64)
    zero = g.call_function(torch.ops.aten.full.default, ([], 0), {"dtype": torch.int64})
    zero.meta["val"] = _meta((), torch.int64)
    where = g.call_function(torch.ops.aten.where.self, (bor, zero, sub))
    where.meta["val"] = _meta((1, 4096), torch.int64)
    idx = g.call_function(torch.ops.aten.index.Tensor, (emb, [where]))
    idx.meta["val"] = _meta((1, 4096, 128), torch.bfloat16)
    g.output(idx)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_no_vocab_graph():
    """Pure matmul graph — no vocab sharding pattern."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((4, 4), torch.float32)
    w = g.placeholder("w")
    w.meta["val"] = _meta((4, 4), torch.float32)
    mm = g.call_function(torch.ops.aten.mm.default, (x, w))
    mm.meta["val"] = _meta((4, 4), torch.float32)
    g.output(mm)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_negative_index_wrapping():
    """Graph with lt(x, 0) for negative index wrapping — NOT vocab sharding.

    Pattern from loss backward: arange → lt(arange, 0) → where(lt, add, arange).
    No matching ge.Scalar feeds into the same bitwise_or.
    """
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((4096,), torch.int64)
    arange = g.call_function(torch.ops.aten.arange.default, (4096,))
    arange.meta["val"] = _meta((4096,), torch.int64)
    lt = g.call_function(torch.ops.aten.lt.Scalar, (arange, 0))
    lt.meta["val"] = _meta((4096,), torch.bool)
    add = g.call_function(torch.ops.aten.add.Tensor, (arange, 4096))
    add.meta["val"] = _meta((4096,), torch.int64)
    where = g.call_function(torch.ops.aten.where.self, (lt, add, arange))
    where.meta["val"] = _meta((4096,), torch.int64)
    g.output(where)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_two_triads(rank: int, tp_size: int = 4, vocab_size: int = 16384):
    """Graph with two vocab triads (like loss backward)."""
    shard_size = vocab_size // tp_size
    offset = rank * shard_size
    g = Graph()
    tokens = g.placeholder("tokens")
    tokens.meta["val"] = _meta((4096,), torch.int64)

    # Triad 1
    lt1 = g.call_function(torch.ops.aten.lt.Scalar, (tokens, offset))
    lt1.meta["val"] = _meta((4096,), torch.bool)
    ge1 = g.call_function(torch.ops.aten.ge.Scalar, (tokens, offset + shard_size))
    ge1.meta["val"] = _meta((4096,), torch.bool)
    bor1 = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt1, ge1))
    bor1.meta["val"] = _meta((4096,), torch.bool)
    clone1 = g.call_function(torch.ops.aten.clone.default, (tokens,))
    clone1.meta["val"] = _meta((4096,), torch.int64)
    sub1 = g.call_function(torch.ops.aten.sub.Tensor, (clone1, offset))
    sub1.meta["val"] = _meta((4096,), torch.int64)

    # Some intervening ops (like in loss backward)
    dummy = g.call_function(torch.ops.aten.neg.default, (tokens,))
    dummy.meta["val"] = _meta((4096,), torch.int64)

    # Triad 2 (same input, same bounds — recomputed in backward)
    lt2 = g.call_function(torch.ops.aten.lt.Scalar, (tokens, offset))
    lt2.meta["val"] = _meta((4096,), torch.bool)
    ge2 = g.call_function(torch.ops.aten.ge.Scalar, (tokens, offset + shard_size))
    ge2.meta["val"] = _meta((4096,), torch.bool)
    bor2 = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt2, ge2))
    bor2.meta["val"] = _meta((4096,), torch.bool)
    clone2 = g.call_function(torch.ops.aten.clone.default, (tokens,))
    clone2.meta["val"] = _meta((4096,), torch.int64)
    sub2 = g.call_function(torch.ops.aten.sub.Tensor, (clone2, offset))
    sub2.meta["val"] = _meta((4096,), torch.int64)

    g.output((sub1, sub2))
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_partial_pattern():
    """Graph with lt.Scalar but no matching ge.Scalar in bitwise_or."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((4096,), torch.int64)
    lt = g.call_function(torch.ops.aten.lt.Scalar, (x, 100))
    lt.meta["val"] = _meta((4096,), torch.bool)
    # ge on a DIFFERENT input
    y = g.placeholder("y")
    y.meta["val"] = _meta((4096,), torch.int64)
    ge = g.call_function(torch.ops.aten.ge.Scalar, (y, 200))
    ge.meta["val"] = _meta((4096,), torch.bool)
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    bor.meta["val"] = _meta((4096,), torch.bool)
    g.output(bor)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_wrong_shard_bounds():
    """Graph where upper - lower doesn't divide evenly (lower % shard_size != 0)."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((4096,), torch.int64)
    # lower=100, upper=200 → shard_size=100, 100 % 100 == 0 → valid
    # lower=50, upper=200 → shard_size=150, 50 % 150 != 0 → invalid
    lt = g.call_function(torch.ops.aten.lt.Scalar, (x, 50))
    lt.meta["val"] = _meta((4096,), torch.bool)
    ge = g.call_function(torch.ops.aten.ge.Scalar, (x, 200))
    ge.meta["val"] = _meta((4096,), torch.bool)
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    bor.meta["val"] = _meta((4096,), torch.bool)
    g.output(bor)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_two_lt_same_bitwise_or():
    """bitwise_or fed by two lt nodes (not lt+ge)."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((4096,), torch.int64)
    lt1 = g.call_function(torch.ops.aten.lt.Scalar, (x, 0))
    lt1.meta["val"] = _meta((4096,), torch.bool)
    lt2 = g.call_function(torch.ops.aten.lt.Scalar, (x, 4096))
    lt2.meta["val"] = _meta((4096,), torch.bool)
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt1, lt2))
    bor.meta["val"] = _meta((4096,), torch.bool)
    g.output(bor)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_triad_with_extra_lt():
    """Vocab triad + extra non-vocab lt.Scalar (like loss backward lt(arange, 0))."""
    g = Graph()
    tokens = g.placeholder("tokens")
    tokens.meta["val"] = _meta((4096,), torch.int64)

    # Valid triad
    lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, 0))
    lt.meta["val"] = _meta((4096,), torch.bool)
    ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, 4096))
    ge.meta["val"] = _meta((4096,), torch.bool)
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    bor.meta["val"] = _meta((4096,), torch.bool)
    clone = g.call_function(torch.ops.aten.clone.default, (tokens,))
    clone.meta["val"] = _meta((4096,), torch.int64)
    sub = g.call_function(torch.ops.aten.sub.Tensor, (clone, 0))
    sub.meta["val"] = _meta((4096,), torch.int64)

    # Extra non-vocab lt (negative index wrapping)
    arange = g.call_function(torch.ops.aten.arange.default, (4096,))
    arange.meta["val"] = _meta((4096,), torch.int64)
    lt_extra = g.call_function(torch.ops.aten.lt.Scalar, (arange, 0))
    lt_extra.meta["val"] = _meta((4096,), torch.bool)

    g.output((sub, lt_extra))
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _apply_both_passes(gm):
    """Apply ShardIndexInjection then VocabShardingSPMD."""
    gm = ShardIndexInjection()(gm).graph_module
    gm = VocabShardingSPMD()(gm).graph_module
    return gm


def _get_scalar_constants(gm):
    """Extract scalar constants from lt/ge/sub nodes."""
    scalars = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target in (torch.ops.aten.lt.Scalar, torch.ops.aten.ge.Scalar):
            scalars.append(("lt/ge", n.args[1]))
        elif n.target == torch.ops.aten.sub.Tensor and isinstance(n.args[1], int | float):
            scalars.append(("sub", n.args[1]))
    return scalars


def _count_placeholders(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "placeholder")


# Graph builders: flat-offset backward pattern
# ============================================================


def _build_flat_offset_backward(rank: int, shard_size: int = 4096, hidden_dim: int = 4096):
    """Build an FX graph mimicking the backward flat-offset pattern.

    Pattern: add.Tensor(flat_indices, vocab_shard_index * shard_size * hidden_dim) -> scatter_add
    This is the backward of index_select for RowwiseParallel embedding.
    """
    flat_shard_size = shard_size * hidden_dim
    offset = rank * flat_shard_size

    g = Graph()
    flat_indices = g.placeholder("flat_indices")
    flat_indices.meta["val"] = _meta((flat_shard_size,), torch.int64)
    src = g.placeholder("src")
    src.meta["val"] = _meta((flat_shard_size,), torch.bfloat16)
    grad_weight = g.placeholder("grad_weight")
    grad_weight.meta["val"] = _meta((shard_size * hidden_dim * 4,), torch.bfloat16)

    add_node = g.call_function(torch.ops.aten.add.Tensor, (flat_indices, offset))
    add_node.meta["val"] = _meta((flat_shard_size,), torch.int64)

    scatter = g.call_function(torch.ops.aten.scatter_add.default, (grad_weight, 0, add_node, src))
    scatter.meta["val"] = _meta((shard_size * hidden_dim * 4,), torch.bfloat16)

    g.output(scatter)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_add_no_scatter():
    """add.Tensor with constant but no scatter_add user — should NOT match."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((1000,), torch.int64)
    add_node = g.call_function(torch.ops.aten.add.Tensor, (x, 1000))
    add_node.meta["val"] = _meta((1000,), torch.int64)
    g.output(add_node)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_add_non_divisible():
    """add.Tensor where constant is not a multiple of input shape — should NOT match."""
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((1000,), torch.int64)
    grad = g.placeholder("grad")
    grad.meta["val"] = _meta((5000,), torch.bfloat16)
    add_node = g.call_function(torch.ops.aten.add.Tensor, (x, 1500))
    add_node.meta["val"] = _meta((1000,), torch.int64)
    scatter = g.call_function(torch.ops.aten.scatter_add.default, (grad, 0, add_node, x))
    scatter.meta["val"] = _meta((5000,), torch.bfloat16)
    g.output(scatter)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


# ============================================================
# Graph builders: mixed patterns
# ============================================================


def _build_mixed_triad_and_flat(
    rank: int, tp_size: int = 4, vocab_size: int = 16384, hidden_dim: int = 4096
):
    """Graph with both a triad and a flat-offset add (same shard index)."""
    shard_size = vocab_size // tp_size
    lower = rank * shard_size
    upper = (rank + 1) * shard_size
    flat_shard_size = shard_size * hidden_dim
    flat_offset = rank * flat_shard_size

    g = Graph()
    tokens = g.placeholder("tokens")
    flat_indices = g.placeholder("flat_indices")
    src = g.placeholder("src")
    grad_weight = g.placeholder("grad_weight")
    with FakeTensorMode() as m:
        tokens.meta["val"] = m.from_tensor(torch.randint(0, vocab_size, (1, 4096)))
        flat_indices.meta["val"] = m.from_tensor(
            torch.randint(0, flat_shard_size, (flat_shard_size,), dtype=torch.int64)
        )
        src.meta["val"] = m.from_tensor(torch.randn(flat_shard_size, dtype=torch.bfloat16))
        grad_weight.meta["val"] = m.from_tensor(
            torch.zeros(vocab_size * hidden_dim, dtype=torch.bfloat16)
        )

    # Triad
    lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, lower))
    ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, upper))
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    clone = g.call_function(torch.ops.aten.clone.default, (tokens,))
    sub = g.call_function(torch.ops.aten.sub.Tensor, (clone, lower))
    for n in [lt, ge]:
        n.meta["val"] = m.from_tensor(torch.zeros(1, 4096, dtype=torch.bool))
    bor.meta["val"] = m.from_tensor(torch.zeros(1, 4096, dtype=torch.bool))
    clone.meta["val"] = tokens.meta["val"]
    sub.meta["val"] = m.from_tensor(torch.randint(0, shard_size, (1, 4096)))

    # Flat-offset add → scatter_add
    add_node = g.call_function(torch.ops.aten.add.Tensor, (flat_indices, flat_offset))
    add_node.meta["val"] = flat_indices.meta["val"]
    scatter = g.call_function(torch.ops.aten.scatter_add.default, (grad_weight, 0, add_node, src))
    scatter.meta["val"] = grad_weight.meta["val"]

    g.output((bor, sub, scatter))
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_conflicting_shard_indices():
    """Graph with a triad (shard_index=1) and flat-offset (shard_index=2).

    This should NOT happen in practice but tests the safety check.
    """
    vocab_size, tp_size, hidden_dim = 16384, 4, 4096
    shard_size = vocab_size // tp_size
    flat_shard_size = shard_size * hidden_dim

    g = Graph()
    tokens = g.placeholder("tokens")
    flat_indices = g.placeholder("flat_indices")
    src = g.placeholder("src")
    grad_weight = g.placeholder("grad_weight")
    with FakeTensorMode() as m:
        tokens.meta["val"] = m.from_tensor(torch.randint(0, vocab_size, (1, 4096)))
        flat_indices.meta["val"] = m.from_tensor(
            torch.randint(0, flat_shard_size, (flat_shard_size,), dtype=torch.int64)
        )
        src.meta["val"] = m.from_tensor(torch.randn(flat_shard_size, dtype=torch.bfloat16))
        grad_weight.meta["val"] = m.from_tensor(
            torch.zeros(vocab_size * hidden_dim, dtype=torch.bfloat16)
        )

    # Triad for rank 1 (shard_index=1): lower=4096, upper=8192
    rank1_lower, rank1_upper = shard_size, 2 * shard_size
    lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, rank1_lower))
    ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, rank1_upper))
    bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
    clone = g.call_function(torch.ops.aten.clone.default, (tokens,))
    sub = g.call_function(torch.ops.aten.sub.Tensor, (clone, rank1_lower))
    for n in [lt, ge]:
        n.meta["val"] = m.from_tensor(torch.zeros(1, 4096, dtype=torch.bool))
    bor.meta["val"] = m.from_tensor(torch.zeros(1, 4096, dtype=torch.bool))
    clone.meta["val"] = tokens.meta["val"]
    sub.meta["val"] = m.from_tensor(torch.randint(0, shard_size, (1, 4096)))

    # Flat-offset for rank 2 (shard_index=2): offset = 2 * flat_shard_size
    rank2_offset = 2 * flat_shard_size
    add_node = g.call_function(torch.ops.aten.add.Tensor, (flat_indices, rank2_offset))
    add_node.meta["val"] = flat_indices.meta["val"]
    scatter = g.call_function(torch.ops.aten.scatter_add.default, (grad_weight, 0, add_node, src))
    scatter.meta["val"] = grad_weight.meta["val"]

    g.output((bor, sub, scatter))
    g.lint()
    return GraphModule(torch.nn.Module(), g)


# ============================================================
# Tests: find_vocab_triads detection
# ============================================================


class TestFindVocabTriads:
    """Tests for the triad detection function."""

    def test_single_triad_detected(self):
        gm = _build_single_triad(rank=0)
        triads = find_vocab_triads(gm.graph)
        assert len(triads) == 1
        assert triads[0].lower_bound == 0
        assert triads[0].upper_bound == 4096
        assert triads[0].shard_size == 4096
        assert triads[0].sub_node is not None

    def test_single_triad_rank1(self):
        gm = _build_single_triad(rank=1)
        triads = find_vocab_triads(gm.graph)
        assert len(triads) == 1
        assert triads[0].lower_bound == 4096
        assert triads[0].upper_bound == 8192

    def test_two_triads_detected(self):
        gm = _build_two_triads(rank=0)
        triads = find_vocab_triads(gm.graph)
        assert len(triads) == 2
        for t in triads:
            assert t.sub_node is not None

    def test_no_vocab_graph(self):
        gm = _build_no_vocab_graph()
        assert find_vocab_triads(gm.graph) == []

    def test_negative_index_wrapping_not_detected(self):
        """lt(arange, 0) without matching ge in bitwise_or should not match."""
        gm = _build_negative_index_wrapping()
        assert find_vocab_triads(gm.graph) == []

    def test_partial_pattern_different_inputs(self):
        """lt and ge on different inputs feeding same bitwise_or → rejected."""
        gm = _build_partial_pattern()
        assert find_vocab_triads(gm.graph) == []

    def test_wrong_shard_bounds(self):
        """lower % shard_size != 0 → rejected."""
        gm = _build_wrong_shard_bounds()
        assert find_vocab_triads(gm.graph) == []

    def test_two_lt_same_bitwise_or(self):
        """bitwise_or(lt, lt) instead of bitwise_or(lt, ge) → rejected."""
        gm = _build_two_lt_same_bitwise_or()
        assert find_vocab_triads(gm.graph) == []

    def test_triad_with_extra_lt(self):
        """Valid triad + extra non-vocab lt → only 1 triad detected."""
        gm = _build_triad_with_extra_lt()
        triads = find_vocab_triads(gm.graph)
        assert len(triads) == 1


# ============================================================
# Tests: ShardIndexInjection
# ============================================================


class TestShardIndexInjection:
    """Tests for the vocab_shard_index placeholder injection pass."""

    def test_injects_rank_for_vocab_graph(self):
        gm = _build_single_triad(rank=0)
        assert _count_placeholders(gm) == 2
        state = _make_state()
        rpass = ShardIndexInjection(state)
        result = rpass(gm)
        assert result.modified
        assert _count_placeholders(result.graph_module) == 3
        assert len(state.pass_mutations[0].input_mutations) == 1
        assert state.pass_mutations[0].input_mutations[0].indices[0] == 2  # original_input_count
        assert state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == 0

    def test_rank_is_last_placeholder(self):
        gm = _build_single_triad(rank=0)
        gm = ShardIndexInjection()(gm).graph_module
        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        assert phs[-1].name == "vocab_shard_index"
        assert phs[0].name == "tokens"
        assert phs[1].name == "emb_table"

    def test_no_injection_for_no_vocab_graph(self):
        """Graphs without vocab patterns are left untouched."""
        gm = _build_no_vocab_graph()
        state = _make_state()
        rpass = ShardIndexInjection(state)
        result = rpass(gm)
        assert not result.modified
        assert _count_placeholders(result.graph_module) == 2  # unchanged
        assert len(state.pass_mutations) == 0

    def test_no_injection_for_negative_index_wrapping(self):
        """Negative index wrapping graph has no vocab patterns — no injection."""
        gm = _build_negative_index_wrapping()
        result = ShardIndexInjection()(gm)
        assert not result.modified

    def test_rank_meta_is_fake_tensor(self):
        """vocab_shard_index placeholder must have FakeTensor meta, not real tensor."""
        gm = _build_single_triad(rank=0)
        gm = ShardIndexInjection()(gm).graph_module
        shard_node = next(n for n in gm.graph.nodes if n.name == "vocab_shard_index")
        val = shard_node.meta["val"]
        assert val.shape == ()
        assert val.dtype == torch.int32

    def test_idempotent_double_application(self):
        """Running ShardIndexInjection twice doesn't add a second placeholder."""
        gm = _build_single_triad(rank=0)
        gm = ShardIndexInjection()(gm).graph_module
        count_before = _count_placeholders(gm)
        result = ShardIndexInjection()(gm)
        assert not result.modified
        assert _count_placeholders(result.graph_module) == count_before

    def test_empty_graph_no_crash(self):
        """Graph with no placeholders doesn't crash."""
        g = Graph()
        g.output(None)
        g.lint()
        gm = GraphModule(torch.nn.Module(), g)
        result = ShardIndexInjection()(gm)
        assert not result.modified

    def test_escape_hatch(self):
        """TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE=1 disables injection."""
        import os

        gm = _build_single_triad(rank=0)
        os.environ["TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE"] = "1"
        try:
            result = ShardIndexInjection()(gm)
            assert not result.modified
            ph_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
            assert "vocab_shard_index" not in ph_names
        finally:
            del os.environ["TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE"]

    def test_shard_size_1_not_detected(self):
        """Triad with shard_size=1 is not detected (too permissive)."""
        g = Graph()
        tokens = g.placeholder("tokens")
        with FakeTensorMode() as m:
            tokens.meta["val"] = m.from_tensor(torch.randint(0, 4, (1, 16)))
        # shard_size=1: lower=0, upper=1
        lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, 0))
        ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, 1))
        bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
        for n in [lt, ge]:
            n.meta["val"] = m.from_tensor(torch.zeros(1, 16, dtype=torch.bool))
        bor.meta["val"] = lt.meta["val"]
        g.output(bor)
        g.lint()
        gm = GraphModule(torch.nn.Module(), g)
        triads = find_vocab_triads(gm.graph)
        assert len(triads) == 0


# ============================================================
# Tests: VocabShardingSPMD
# ============================================================


class TestVocabShardingSPMD:
    """Tests for the constant rewriting pass."""

    def test_no_op_without_rank_placeholder(self):
        """VocabShardingSPMD should be a no-op if vocab_shard_index placeholder is missing."""
        gm = _build_single_triad(rank=0)
        result = VocabShardingSPMD()(gm)
        assert not result.modified

    def test_rewrites_single_triad(self):
        gm = _build_single_triad(rank=0)
        gm = _apply_both_passes(gm)
        # No scalar constants should remain in lt/ge/sub
        scalars = _get_scalar_constants(gm)
        assert scalars == [], f"Scalar constants remain: {scalars}"

    def test_all_ranks_identical(self):
        """After passes, all 4 ranks produce structurally identical graphs."""
        graphs = {}
        for rank in range(4):
            gm = _build_single_triad(rank=rank)
            gm = _apply_both_passes(gm)
            graphs[rank] = gm.print_readable(print_output=False)
        for rank in range(1, 4):
            assert graphs[0] == graphs[rank], f"Rank 0 and rank {rank} differ after passes"

    def test_two_triads_all_ranks_identical(self):
        graphs = {}
        for rank in range(4):
            gm = _build_two_triads(rank=rank)
            gm = _apply_both_passes(gm)
            graphs[rank] = gm.print_readable(print_output=False)
        for rank in range(1, 4):
            assert graphs[0] == graphs[rank]

    def test_no_op_for_no_vocab_graph(self):
        gm = _build_no_vocab_graph()
        gm = ShardIndexInjection()(gm).graph_module
        result = VocabShardingSPMD()(gm)
        assert not result.modified

    def test_numerical_correctness(self):
        """Rewritten graph produces same output as original for all ranks."""
        vocab_size, tp_size, dim = 16384, 4, 128
        shard_size = vocab_size // tp_size
        tokens = torch.randint(0, vocab_size, (1, 4096))
        emb = torch.randn(shard_size, dim, dtype=torch.bfloat16)

        for rank in range(tp_size):
            gm_orig = _build_single_triad(rank, tp_size, vocab_size)
            gm_new = _build_single_triad(rank, tp_size, vocab_size)
            gm_new = _apply_both_passes(gm_new)
            rank_tensor = torch.tensor(rank, dtype=torch.int32)
            out_orig = gm_orig(tokens, emb)
            out_new = gm_new(tokens, emb, rank_tensor)
            assert torch.equal(out_orig, out_new), f"Rank {rank}: outputs differ"

    def test_lt_becomes_lt_tensor(self):
        """After rewrite, lt.Scalar should become lt.Tensor."""
        gm = _build_single_triad(rank=0)
        gm = _apply_both_passes(gm)
        lt_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.lt.Tensor
        ]
        assert len(lt_nodes) == 1

    def test_ge_becomes_ge_tensor(self):
        gm = _build_single_triad(rank=0)
        gm = _apply_both_passes(gm)
        ge_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.ge.Tensor
        ]
        assert len(ge_nodes) == 1

    def test_graph_lints_after_rewrite(self):
        """Graph should pass lint() after rewrite (no topo violations)."""
        for rank in range(4):
            gm = _build_single_triad(rank=rank)
            gm = _apply_both_passes(gm)
            gm.graph.lint()  # Raises if topo order violated

    def test_two_triads_graph_lints(self):
        for rank in range(4):
            gm = _build_two_triads(rank=rank)
            gm = _apply_both_passes(gm)
            gm.graph.lint()

    def test_triad_with_extra_lt_only_rewrites_triad(self):
        """Extra non-vocab lt.Scalar should be left untouched."""
        gm = _build_triad_with_extra_lt()
        gm = _apply_both_passes(gm)
        # The extra lt.Scalar(arange, 0) should still be lt.Scalar
        lt_scalar_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.lt.Scalar
        ]
        assert (
            len(lt_scalar_nodes) == 1
        ), f"Expected 1 remaining lt.Scalar, got {len(lt_scalar_nodes)}"


# ============================================================
# Tests: mixed patterns and shard index validation
# ============================================================


class TestTPPlusDPShardIndex:
    """Verify shard_index correctness under TP+DP (e.g., TP4+FSDP2).

    In TP4+FSDP2 with 8 global ranks, the mapping is:
        global 0,4 → TP shard 0 (same triad: lower=0, upper=4096)
        global 1,5 → TP shard 1 (same triad: lower=4096, upper=8192)
        global 2,6 → TP shard 2 (same triad: lower=8192, upper=12288)
        global 3,7 → TP shard 3 (same triad: lower=12288, upper=16384)

    DTensor produces the same FX graph for ranks in the same TP shard,
    so we test that identical graphs yield the same shard_index.
    """

    def test_same_tp_shard_same_index(self):
        """Ranks 0 and 4 (both TP shard 0) get shard_index=0."""
        for _ in range(2):  # simulate global rank 0 and 4
            gm = _build_single_triad(rank=0)
            state = _make_state()
            inj = ShardIndexInjection(state)
            inj(gm)
            assert state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == 0

    def test_all_tp_shards_correct(self):
        """Each TP shard index is correctly inferred from the triad constants."""
        for tp_shard in range(4):
            gm = _build_single_triad(rank=tp_shard)
            state = _make_state()
            inj = ShardIndexInjection(state)
            inj(gm)
            assert (
                state.pass_mutations[0].input_mutations[0].transform_params["shard_index"]
                == tp_shard
            )

    def test_tp_plus_dp_graphs_identical_within_shard(self):
        """Ranks in the same TP shard produce identical rewritten graphs."""
        for tp_shard in range(4):
            # Two "global ranks" with the same TP shard (e.g., rank 0 & 4)
            gm_a = _build_single_triad(rank=tp_shard)
            gm_b = _build_single_triad(rank=tp_shard)
            gm_a = _apply_both_passes(gm_a)
            gm_b = _apply_both_passes(gm_b)
            assert gm_a.print_readable(print_output=False) == gm_b.print_readable(
                print_output=False
            )

    def test_different_tp_shards_different_index(self):
        """Different TP shards get different shard_index values."""
        indices = set()
        for tp_shard in range(4):
            gm = _build_single_triad(rank=tp_shard)
            state = _make_state()
            inj = ShardIndexInjection(state)
            inj(gm)
            indices.add(state.pass_mutations[0].input_mutations[0].transform_params["shard_index"])
        assert indices == {0, 1, 2, 3}


class TestMixedPatterns:
    """Tests for graphs with both triads and flat-offset adds."""

    def test_mixed_all_ranks_identical(self):
        """Triad + flat-offset in same graph, all ranks produce identical output."""
        readables = []
        for rank in range(4):
            gm = _build_mixed_triad_and_flat(rank=rank)
            gm = _apply_both_passes(gm)
            readables.append(gm.print_readable(print_output=False))
        assert len(set(readables)) == 1

    def test_mixed_shard_index_consistent(self):
        """IOMutation shard_index matches the vocab shard index for mixed graphs."""
        for rank in range(4):
            gm = _build_mixed_triad_and_flat(rank=rank)
            state = _make_state()
            injection = ShardIndexInjection(state)
            injection(gm)
            assert (
                state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == rank
            )

    def test_mixed_numerical_correctness(self):
        """Rewritten mixed graph produces same output as original."""
        for rank in range(4):
            gm_orig = _build_mixed_triad_and_flat(rank=rank)
            gm_new = _build_mixed_triad_and_flat(rank=rank)
            gm_new = _apply_both_passes(gm_new)

            shard_size, hidden_dim = 4096, 4096
            flat_shard_size = shard_size * hidden_dim
            tokens = torch.randint(0, 16384, (1, 4096))
            flat_idx = torch.randint(0, flat_shard_size, (flat_shard_size,), dtype=torch.int64)
            src = torch.randn(flat_shard_size, dtype=torch.bfloat16)
            grad_w = torch.zeros(16384 * 4096, dtype=torch.bfloat16)

            orig_out = gm_orig(tokens, flat_idx, src, grad_w)
            new_out = gm_new(tokens, flat_idx, src, grad_w, torch.tensor(rank, dtype=torch.int32))
            for o, n in zip(orig_out, new_out, strict=False):
                torch.testing.assert_close(o, n)

    def test_conflicting_shard_indices_no_injection(self):
        """Conflicting shard indices → no injection, pass is no-op."""
        gm = _build_conflicting_shard_indices()
        injection = ShardIndexInjection()
        result = injection(gm)
        assert not result.modified
        # No vocab_shard_index placeholder added
        ph_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        assert "vocab_shard_index" not in ph_names

    def test_conflicting_indices_vocab_pass_noop(self):
        """With conflicting indices, VocabShardingSPMD is a no-op (no shard node)."""
        gm = _build_conflicting_shard_indices()
        injection = ShardIndexInjection()
        inject_result = injection(gm)
        # No vocab_shard_index → VocabShardingSPMD can't find it → no-op
        result = VocabShardingSPMD()(inject_result.graph_module)
        assert not result.modified


# ============================================================
# Tests: find_flat_offset_adds detection
# ============================================================


class TestFindFlatOffsetAdds:
    """Tests for the flat-offset add detection function."""

    def test_rank0_detected(self):
        """Rank 0 has offset=0, which is 0 % N == 0 → detected."""
        gm = _build_flat_offset_backward(rank=0)
        adds = find_flat_offset_adds(gm.graph)
        assert len(adds) == 1
        assert adds[0].flat_shard_size == 4096 * 4096

    def test_rank1_detected(self):
        gm = _build_flat_offset_backward(rank=1)
        adds = find_flat_offset_adds(gm.graph)
        assert len(adds) == 1
        assert adds[0].flat_shard_size == 4096 * 4096

    def test_no_scatter_not_detected(self):
        gm = _build_add_no_scatter()
        assert find_flat_offset_adds(gm.graph) == []

    def test_non_divisible_not_detected(self):
        gm = _build_add_non_divisible()
        assert find_flat_offset_adds(gm.graph) == []

    def test_no_vocab_graph_not_detected(self):
        gm = _build_no_vocab_graph()
        assert find_flat_offset_adds(gm.graph) == []


# ============================================================
# Tests: ShardIndexInjection for flat-offset pattern
# ============================================================


class TestShardIndexInjectionFlatOffset:
    """Tests for vocab_shard_index injection triggered by flat-offset pattern (no triads)."""

    def test_injects_rank_for_flat_offset(self):
        gm = _build_flat_offset_backward(rank=1)
        assert _count_placeholders(gm) == 3  # flat_indices, src, grad_weight
        state = _make_state()
        rpass = ShardIndexInjection(state)
        result = rpass(gm)
        assert result.modified
        assert _count_placeholders(result.graph_module) == 4
        assert state.pass_mutations[0].input_mutations[0].indices[0] == 3  # original_input_count

    def test_rank_is_last_placeholder(self):
        gm = _build_flat_offset_backward(rank=1)
        gm = ShardIndexInjection()(gm).graph_module
        phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
        assert phs[-1].name == "vocab_shard_index"


# ============================================================
# Tests: VocabShardingSPMD for flat-offset pattern
# ============================================================


class TestVocabShardingSPMDFlatOffset:
    """Tests for flat-offset add rewriting."""

    def test_all_ranks_identical(self):
        """After passes, all 4 ranks produce structurally identical graphs."""
        graphs = {}
        for rank in range(4):
            gm = _build_flat_offset_backward(rank=rank)
            gm = _apply_both_passes(gm)
            graphs[rank] = gm.print_readable(print_output=False)
        for rank in range(1, 4):
            assert graphs[0] == graphs[rank], f"Rank 0 and rank {rank} differ after passes"

    def test_no_constant_remains(self):
        """After rewrite, the add.Tensor should have a graph node, not a constant."""
        for rank in range(4):
            gm = _build_flat_offset_backward(rank=rank)
            gm = _apply_both_passes(gm)
            for n in gm.graph.nodes:
                if n.op == "call_function" and n.target == torch.ops.aten.add.Tensor:
                    assert isinstance(
                        n.args[1], Node
                    ), f"Rank {rank}: add.Tensor still has constant {n.args[1]}"

    def test_numerical_correctness(self):
        """Rewritten graph produces same output as original for all ranks."""
        shard_size, hidden_dim = 4096, 4096
        flat_shard_size = shard_size * hidden_dim

        for rank in range(4):
            gm_orig = _build_flat_offset_backward(
                rank=rank, shard_size=shard_size, hidden_dim=hidden_dim
            )
            gm_new = _build_flat_offset_backward(
                rank=rank, shard_size=shard_size, hidden_dim=hidden_dim
            )
            gm_new = _apply_both_passes(gm_new)

            flat_indices = torch.randint(0, flat_shard_size, (flat_shard_size,))
            src = torch.randn(flat_shard_size, dtype=torch.bfloat16)
            grad_weight = torch.zeros(flat_shard_size * 4, dtype=torch.bfloat16)
            rank_tensor = torch.tensor(rank, dtype=torch.int32)

            out_orig = gm_orig(flat_indices, src, grad_weight)
            out_new = gm_new(flat_indices, src, grad_weight, rank_tensor)
            assert torch.equal(out_orig, out_new), f"Rank {rank}: outputs differ"

    def test_graph_lints_after_rewrite(self):
        for rank in range(4):
            gm = _build_flat_offset_backward(rank=rank)
            gm = _apply_both_passes(gm)
            gm.graph.lint()


# ============================================================
# Injection robustness and fallback path tests
# ============================================================


# ============================================================
# Backward slice pattern builders and tests
# ============================================================


def _build_backward_slice(
    rank: int, tp_size: int = 4, vocab_size: int = 16384, hidden_dim: int = 4096
):
    """Backward embedding gradient slice: slice.Tensor(grad, 0, rank*shard, (rank+1)*shard)."""
    shard_size = vocab_size // tp_size
    start = rank * shard_size
    end = start + shard_size

    g = Graph()
    grad = g.placeholder("grad")
    grad.meta["val"] = _meta((vocab_size, hidden_dim), torch.bfloat16)
    sliced = g.call_function(torch.ops.aten.slice.Tensor, (grad, 0, start, end))
    sliced.meta["val"] = _meta((shard_size, hidden_dim), torch.bfloat16)
    clone = g.call_function(torch.ops.aten.clone.default, (sliced,))
    clone.meta["val"] = _meta((shard_size, hidden_dim), torch.bfloat16)
    g.output(clone)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_backward_slice_wrong_size(vocab_size: int = 16384, hidden_dim: int = 4096):
    """Slice that doesn't evenly divide vocab — should NOT be detected."""
    g = Graph()
    grad = g.placeholder("grad")
    grad.meta["val"] = _meta((vocab_size, hidden_dim), torch.bfloat16)
    sliced = g.call_function(torch.ops.aten.slice.Tensor, (grad, 0, 0, 3000))
    sliced.meta["val"] = _meta((3000, hidden_dim), torch.bfloat16)
    g.output(sliced)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


def _build_backward_slice_dim1(
    rank: int = 0, tp_size: int = 4, seq_len: int = 4096, hidden_dim: int = 16384
):
    """Slice on dim=1 — should NOT be detected."""
    shard_size = hidden_dim // tp_size
    start = rank * shard_size
    end = start + shard_size
    g = Graph()
    x = g.placeholder("x")
    x.meta["val"] = _meta((seq_len, hidden_dim), torch.bfloat16)
    sliced = g.call_function(torch.ops.aten.slice.Tensor, (x, 1, start, end))
    sliced.meta["val"] = _meta((seq_len, shard_size), torch.bfloat16)
    g.output(sliced)
    g.lint()
    return GraphModule(torch.nn.Module(), g)


class TestFindBackwardSlices:
    """Tests for find_backward_slices() pattern detection."""

    def test_rank0_detected(self):
        gm = _build_backward_slice(rank=0)
        slices = find_backward_slices(gm.graph)
        assert len(slices) == 1
        assert slices[0].start == 0
        assert slices[0].shard_size == 4096

    def test_rank2_detected(self):
        gm = _build_backward_slice(rank=2)
        slices = find_backward_slices(gm.graph)
        assert len(slices) == 1
        assert slices[0].start == 8192

    def test_all_ranks_detected(self):
        for rank in range(4):
            gm = _build_backward_slice(rank=rank)
            slices = find_backward_slices(gm.graph)
            assert len(slices) == 1
            assert slices[0].start == rank * 4096

    def test_shard_index_inferred(self):
        for rank in range(4):
            gm = _build_backward_slice(rank=rank)
            slices = find_backward_slices(gm.graph)
            assert slices[0].start // slices[0].shard_size == rank

    def test_wrong_shard_size_not_detected(self):
        gm = _build_backward_slice_wrong_size()
        assert len(find_backward_slices(gm.graph)) == 0

    def test_dim1_not_detected(self):
        gm = _build_backward_slice_dim1()
        assert len(find_backward_slices(gm.graph)) == 0

    def test_no_vocab_graph_not_detected(self):
        gm = _build_no_vocab_graph()
        assert len(find_backward_slices(gm.graph)) == 0

    def test_shard_size_1_not_detected(self):
        """shard_size=1 is trivial (start % 1 == 0 always) — must be rejected."""
        g = Graph()
        x = g.placeholder("x")
        x.meta["val"] = _meta((4, 64), torch.bfloat16)
        sliced = g.call_function(torch.ops.aten.slice.Tensor, (x, 0, 0, 1))
        sliced.meta["val"] = _meta((1, 64), torch.bfloat16)
        g.output(sliced)
        g.lint()
        gm = GraphModule(torch.nn.Module(), g)
        assert len(find_backward_slices(gm.graph)) == 0

    def test_single_shard_not_detected(self):
        """tp_size=1 means shard_size == vocab_size — no-op slice, must be rejected."""
        gm = _build_backward_slice(rank=0, tp_size=1)
        assert len(find_backward_slices(gm.graph)) == 0


class TestShardIndexInjectionBackwardSlice:
    """ShardIndexInjection should detect backward slices and inject placeholder."""

    def test_injects_for_backward_slice(self):
        gm = _build_backward_slice(rank=0)
        state = _make_state()
        rpass = ShardIndexInjection(state)
        result = rpass(gm)
        assert result.modified
        assert state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == 0

    def test_shard_index_from_backward_slice(self):
        for rank in range(4):
            gm = _build_backward_slice(rank=rank)
            state = _make_state()
            rpass = ShardIndexInjection(state)
            rpass(gm)
            assert (
                state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == rank
            )


class TestVocabShardingSPMDBackwardSlice:
    """VocabShardingSPMD should rewrite backward slice to use vocab_shard_index."""

    def test_all_ranks_identical(self):
        hashes = set()
        for rank in range(4):
            gm = _build_backward_slice(rank=rank)
            gm = _apply_both_passes(gm)
            hashes.add(gm.graph.python_code("self").src)
        assert len(hashes) == 1

    def test_no_constant_start_end_remains(self):
        gm = _build_backward_slice(rank=1)
        gm = _apply_both_passes(gm)
        for node in gm.graph.nodes:
            if node.target == torch.ops.aten.slice.Tensor:
                assert isinstance(
                    node.args[2], Node
                ), f"start should be Node, got {type(node.args[2])}"
                assert isinstance(
                    node.args[3], Node
                ), f"end should be Node, got {type(node.args[3])}"

    def test_graph_lints_after_rewrite(self):
        for rank in range(4):
            gm = _build_backward_slice(rank=rank)
            gm = _apply_both_passes(gm)
            gm.graph.lint()

    def test_numerical_correctness(self):
        """Rewritten graph produces same output as original for all ranks."""
        vocab_size, tp_size, hidden_dim = 16384, 4, 4096
        grad = torch.randn(vocab_size, hidden_dim, dtype=torch.bfloat16)
        for rank in range(tp_size):
            gm_orig = _build_backward_slice(rank, tp_size, vocab_size, hidden_dim)
            gm_new = _build_backward_slice(rank, tp_size, vocab_size, hidden_dim)
            gm_new = _apply_both_passes(gm_new)
            out_orig = gm_orig(grad)
            out_new = gm_new(grad, torch.tensor(rank, dtype=torch.int32))
            assert torch.equal(out_orig, out_new), f"Rank {rank}: outputs differ"


class TestBackwardSliceMixedPatterns:
    """Test backward slice interacting with other patterns in the same graph."""

    def test_backward_slice_with_triad_same_rank(self):
        """Backward slice + triad in same graph, same rank → both rewritten, SPMD."""
        for rank in range(4):
            g = Graph()
            tokens = g.placeholder("tokens")
            tokens.meta["val"] = _meta((4096,), torch.int64)
            grad = g.placeholder("grad")
            grad.meta["val"] = _meta((16384, 4096), torch.bfloat16)

            # Triad pattern
            shard_size = 4096
            lower = rank * shard_size
            upper = lower + shard_size
            lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, lower))
            lt.meta["val"] = _meta((4096,), torch.bool)
            ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, upper))
            ge.meta["val"] = _meta((4096,), torch.bool)
            bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
            bor.meta["val"] = _meta((4096,), torch.bool)
            sub = g.call_function(torch.ops.aten.sub.Tensor, (tokens, lower))
            sub.meta["val"] = _meta((4096,), torch.int64)

            # Backward slice pattern
            sliced = g.call_function(torch.ops.aten.slice.Tensor, (grad, 0, lower, upper))
            sliced.meta["val"] = _meta((shard_size, 4096), torch.bfloat16)

            g.output((bor, sub, sliced))
            g.lint()
            gm = GraphModule(torch.nn.Module(), g)

            state = _make_state()
            rpass = ShardIndexInjection(state)
            rpass(gm)
            assert (
                state.pass_mutations[0].input_mutations[0].transform_params["shard_index"] == rank
            )

        # All ranks identical after rewrite
        hashes = set()
        for rank in range(4):
            g = Graph()
            tokens = g.placeholder("tokens")
            tokens.meta["val"] = _meta((4096,), torch.int64)
            grad = g.placeholder("grad")
            grad.meta["val"] = _meta((16384, 4096), torch.bfloat16)
            shard_size = 4096
            lower = rank * shard_size
            upper = lower + shard_size
            lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, lower))
            lt.meta["val"] = _meta((4096,), torch.bool)
            ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, upper))
            ge.meta["val"] = _meta((4096,), torch.bool)
            bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
            bor.meta["val"] = _meta((4096,), torch.bool)
            sub = g.call_function(torch.ops.aten.sub.Tensor, (tokens, lower))
            sub.meta["val"] = _meta((4096,), torch.int64)
            sliced = g.call_function(torch.ops.aten.slice.Tensor, (grad, 0, lower, upper))
            sliced.meta["val"] = _meta((shard_size, 4096), torch.bfloat16)
            g.output((bor, sub, sliced))
            g.lint()
            gm = GraphModule(torch.nn.Module(), g)
            gm = _apply_both_passes(gm)
            hashes.add(gm.graph.python_code("self").src)
        assert len(hashes) == 1

    def test_conflicting_backward_slice_and_triad(self):
        """Backward slice rank != triad rank → no injection."""
        g = Graph()
        tokens = g.placeholder("tokens")
        tokens.meta["val"] = _meta((4096,), torch.int64)
        grad = g.placeholder("grad")
        grad.meta["val"] = _meta((16384, 4096), torch.bfloat16)

        # Triad for rank 0
        lt = g.call_function(torch.ops.aten.lt.Scalar, (tokens, 0))
        lt.meta["val"] = _meta((4096,), torch.bool)
        ge = g.call_function(torch.ops.aten.ge.Scalar, (tokens, 4096))
        ge.meta["val"] = _meta((4096,), torch.bool)
        bor = g.call_function(torch.ops.aten.bitwise_or.Tensor, (lt, ge))
        bor.meta["val"] = _meta((4096,), torch.bool)

        # Backward slice for rank 2
        sliced = g.call_function(torch.ops.aten.slice.Tensor, (grad, 0, 8192, 12288))
        sliced.meta["val"] = _meta((4096, 4096), torch.bfloat16)

        g.output((bor, sliced))
        g.lint()
        gm = GraphModule(torch.nn.Module(), g)

        rpass = ShardIndexInjection()
        result = rpass(gm)
        assert not result.modified  # conflicting indices → no injection


class TestInjectionRobustness:
    """Verify that graphs with an injected vocab_shard_index placeholder
    execute correctly when the shard tensor is provided, even if the graph
    has no vocab sharding patterns (e.g., plain matmul, backward graphs,
    tensor subclass inputs).
    """

    def _forward_with_shard(self, gm, state, *inputs):
        """Call gm.forward with shard tensor appended if injection occurred."""
        args = list(inputs)
        if state.pass_mutations and len(state.pass_mutations[0].input_mutations) > 0:
            shard_index = state.pass_mutations[0].input_mutations[0].transform_params["shard_index"]
            args.append(torch.tensor(shard_index, dtype=torch.int32))
        return gm.forward(*args)

    def test_no_vocab_graph_with_shard_input(self):
        """No-vocab graph works when shard tensor is appended."""
        gm = _build_no_vocab_graph()
        state = _make_state()
        rpass = ShardIndexInjection(state)
        rpass(gm)
        result = self._forward_with_shard(gm, state, torch.randn(4, 4), torch.randn(4, 4))
        assert result is not None

    def test_output_shape_preserved(self):
        """Output shape is unchanged by the extra shard placeholder."""
        gm = _build_no_vocab_graph()
        state = _make_state()
        rpass = ShardIndexInjection(state)
        rpass(gm)
        result = self._forward_with_shard(gm, state, torch.randn(4, 4), torch.randn(4, 4))
        assert result.shape == (4, 4)

    def test_subclass_input(self):
        """Tensor subclass inputs work with injected shard placeholder."""
        gm = _build_no_vocab_graph()
        state = _make_state()
        rpass = ShardIndexInjection(state)
        rpass(gm)

        class MySubclass(torch.Tensor):
            pass

        x = torch.randn(4, 4).as_subclass(MySubclass)
        result = self._forward_with_shard(gm, state, x, torch.randn(4, 4))
        assert result is not None

    def test_backward_graph(self):
        """Backward-like graph works with injected shard placeholder."""
        g = Graph()
        grad_out = g.placeholder("grad_output")
        grad_out.meta["val"] = _meta((4, 4), torch.float32)
        weight = g.placeholder("weight")
        weight.meta["val"] = _meta((4, 4), torch.float32)
        grad_input = g.call_function(torch.ops.aten.mm.default, (grad_out, weight))
        grad_input.meta["val"] = _meta((4, 4), torch.float32)
        g.output(grad_input)
        g.lint()
        gm = GraphModule(torch.nn.Module(), g)

        state = _make_state()
        rpass = ShardIndexInjection(state)
        rpass(gm)
        result = self._forward_with_shard(gm, state, torch.randn(4, 4), torch.randn(4, 4))
        assert result.shape == (4, 4)


class TestFallbackWithInjectedInputs:
    """Verify that the execution fallback logic correctly appends injected
    inputs (shard tensor) so that gm.forward() signature matches.
    """

    def test_vocab_graph_fallback(self):
        """Vocab graph forward works when shard tensor is appended."""
        gm = _build_single_triad(rank=0)
        state = _make_state()
        rpass = ShardIndexInjection(state)
        result = rpass(gm)
        assert result.modified

        original_inputs = [torch.arange(16384, dtype=torch.int64), torch.randn(16384, 64)]
        fallback_inputs = list(original_inputs)
        if len(state.pass_mutations[0].input_mutations) > 0:
            shard_index = state.pass_mutations[0].input_mutations[0].transform_params["shard_index"]
            fallback_inputs.append(torch.tensor(shard_index, dtype=torch.int32))

        result = gm.forward(*fallback_inputs)
        assert result is not None

    def test_no_vocab_graph_no_extra_inputs(self):
        """No-vocab graph needs no extra inputs when injection is skipped."""
        gm = _build_no_vocab_graph()
        state = _make_state()
        rpass = ShardIndexInjection(state)
        rpass(gm)

        original_inputs = [torch.randn(4, 4), torch.randn(4, 4)]
        fallback_inputs = list(original_inputs)
        if state.pass_mutations and len(state.pass_mutations[0].input_mutations) > 0:
            shard_index = state.pass_mutations[0].input_mutations[0].transform_params["shard_index"]
            fallback_inputs.append(torch.tensor(shard_index, dtype=torch.int32))

        assert len(fallback_inputs) == len(original_inputs)
        result = gm.forward(*fallback_inputs)
        assert result is not None
