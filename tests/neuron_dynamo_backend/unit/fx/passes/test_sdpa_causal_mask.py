"""
Unit tests for the SdpaCausalMaskRewrite fx pass.
"""

import copy
import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.fx.passes.infra.pass_manager import PassManager

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.decompositions import _can_use_nki_flash_attention
from torch_neuronx.neuron_dynamo_backend.fx.passes import sdpa_causal_mask
from torch_neuronx.neuron_dynamo_backend.fx.passes.sdpa_causal_mask import (
    SdpaCausalMaskRewrite,
)
from torch_neuronx.neuron_dynamo_backend.fx.pipelines.preprocessing_pipeline import (
    preprocess_graph,
)

B, H, L, E = 1, 2, 8, 4
_SDPA = torch._C._nn.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _fresh_dynamo_cache():
    """Dynamo marks a dimension dynamic once it changes between two compiles of
    one code object, and the fold needs a static extent.
    """
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _qkv(length=L, width=L, batch=B, heads=H, kv_heads=None, dtype=torch.float32):
    torch.manual_seed(0)
    kv_heads = heads if kv_heads is None else kv_heads
    return (
        torch.randn(batch, heads, length, E, dtype=dtype),
        torch.randn(batch, kv_heads, width, E, dtype=dtype),
        torch.randn(batch, kv_heads, width, E, dtype=dtype),
    )


def _pre_aot(model, *inputs):
    gm = get_aot_graphs(model, *inputs).pre_aot_graph
    assert gm is not None, "pre-AOT graph was not captured"
    return gm


def _sdpa_nodes(gm, target=_SDPA):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


def _sdpa_node(gm, target=_SDPA):
    nodes = _sdpa_nodes(gm, target)
    if not nodes:
        pytest.fail("no scaled_dot_product_attention node in the graph")
    return nodes[0]


def _run_pass(model, *inputs, target=_SDPA):
    """Run the pass on the pre-AOT graph. Return the SDPA node and the graph module."""
    result = PassManager(passes=[SdpaCausalMaskRewrite()])(_pre_aot(model, *inputs))
    return _sdpa_node(result.graph_module, target), result.graph_module


def _args_of(node):
    """Return the SDPA arguments by name, whether passed positionally."""
    names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal")
    merged = dict(node.kwargs)
    merged.update(dict(zip(names, node.args, strict=False)))
    return merged


def _fired(node) -> bool:
    """Return True if the pass traded the mask for the flag."""
    args = _args_of(node)
    return args.get("attn_mask") is None and args.get("is_causal") is True


def _kept(node) -> bool:
    """Return True if the pass left the mask on the call."""
    args = _args_of(node)
    return isinstance(args.get("attn_mask"), torch.fx.Node) and args.get("is_causal") is not True


class CausalMask(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CausalFloatMask(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        keep = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        mask = torch.where(keep, 0.0, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class SentinelMask(nn.Module):
    """A float mask that drops a position with ``dropped``. The default is the
    sentinel HuggingFace writes, the minimum of the query dtype.
    """

    def __init__(self, dropped=None):
        super().__init__()
        self.dropped = dropped

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        keep = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        dropped = torch.finfo(q.dtype).min if self.dropped is None else self.dropped
        mask = torch.where(keep, 0.0, dropped).to(q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class BidirectionalMask(nn.Module):
    """An all-true mask over the full (L, W) extent."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(1, 1, length, 1, dtype=torch.bool, device=q.device).expand(
            B, 1, length, width
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class SlidingWindowMask(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        rows = torch.arange(length, device=q.device).unsqueeze(-1)
        cols = torch.arange(width, device=q.device).unsqueeze(0)
        mask = (cols <= rows) & (rows - cols < 4)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class RuntimeMask(nn.Module):
    """The mask arrives as a graph input, so its value is not fixed at compile time."""

    def forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class PositionalDropout(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, mask, 0.5)


class FiniteBias(nn.Module):
    """An additive bias, not a mask. Rewriting it would discard the bias."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        keep = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        bias = torch.where(keep, 0.0, -10000.0)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class WideningMask(nn.Module):
    """The mask's batch extent exceeds the query's, so SDPA broadcasts the output
    up to the mask's batch. CPU eager rejects this program, but the compiled path
    accepts it and returns the wider shape.
    """

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.expand(2, 1, length, width))


class QueryBatchMask(nn.Module):
    """The benign form: the mask is expanded to the query's own batch."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.expand(q.shape[0], 1, length, width)
        )


class RankedMask(nn.Module):
    """A causal mask broadcast to rank 2, 3 or 4."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        if self.rank == 3:
            mask = mask.expand(q.shape[0], length, width)
        elif self.rank == 4:
            mask = mask.expand(q.shape[0], 1, length, width)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class ExpandedMask(nn.Module):
    """A causal mask whose batch axis is an expand over a real head stride.

    The strides are then (0, L * S, S, 1), which no single view can flatten, so
    reshape has to copy every plane.
    """

    def forward(self, q, k, v):
        batch, heads = q.shape[0], q.shape[1]
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(1, heads, length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.expand(batch, heads, length, width)
        )


class PerHeadMask(nn.Module):
    """Causal for head 0 and dense for head 1."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        rows = torch.arange(length, device=q.device).unsqueeze(-1)
        cols = torch.arange(width, device=q.device).unsqueeze(0)
        heads = torch.arange(q.shape[1], device=q.device).reshape(-1, 1, 1)
        mask = (cols <= rows).expand(q.shape[1], length, width) | (heads == 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0))


class ComparisonMask(nn.Module):
    """``dead_row=True`` drops every key for query row 0, which is_causal cannot
    express, because is_causal lets row 0 attend to itself.
    """

    def __init__(self, dead_row):
        super().__init__()
        self.dead_row = dead_row

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        rows = torch.arange(length, device=q.device).unsqueeze(-1)
        cols = torch.arange(width, device=q.device).unsqueeze(0)
        alive = rows != 0 if self.dead_row else rows >= 0
        mask = (cols <= rows) & alive
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class ScaledMask(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.125)


class GqaMask(nn.Module):
    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


class ReturnsTheMask(nn.Module):
    """The mask is also a graph output, so erasure must spare its producers."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), mask


class TwoCallSites(nn.Module):
    """Two calls share one cone, and both accept it."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return a + b


class WritesTheCone(nn.Module):
    """``write=True`` puts an operator.setitem on a cone node. The write makes the
    mask non-causal, and the fold replays the producers only, so it cannot see it.
    """

    def __init__(self, write=True):
        super().__init__()
        self.write = write

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        if self.write:
            base[0, width - 1] = True
        return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())


class InplaceOverload(nn.Module):
    """``write=True`` puts an in-place aten overload on a cone node."""

    def __init__(self, write=True):
        super().__init__()
        self.write = write

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        base = torch.ones(length, width, device=q.device).tril()
        if self.write:
            torch.ops.aten.add_.Tensor(base, 1.0)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone().bool())


class ViewWithoutAWrite(nn.Module):
    """A live view of a cone node, with nothing writing through it."""

    def forward(self, q, k, v):
        length, width = q.shape[-2], k.shape[-2]
        base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
        alias = base.view(length, width)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone()), alias.sum()


class TestRewrites:
    """Masks the pass must recognize as causal."""

    def test_causal_mask_becomes_is_causal(self):
        node, _ = _run_pass(CausalMask(), *_qkv())
        assert _fired(node)

    def test_mask_producers_are_erased(self):
        _, gm = _run_pass(CausalMask(), *_qkv())
        targets = {n.target for n in gm.graph.nodes}
        assert torch.ones not in targets
        assert "tril" not in targets


class TestRewrittenGraph:
    """The rewritten module must return what the original returned."""

    def test_a_bool_mask_rewrite_keeps_the_values(self):
        self._compare(CausalMask(), *_qkv())

    @staticmethod
    def _compare(model, *inputs):
        before = _pre_aot(model, *inputs)
        result = PassManager(passes=[SdpaCausalMaskRewrite()])(copy.deepcopy(before))
        assert result.modified is True
        torch.testing.assert_close(result.graph_module(*inputs), before(*inputs))


class TestBails:
    """Masks the pass must leave alone."""

    @pytest.mark.parametrize(
        "model_cls", [BidirectionalMask, SlidingWindowMask, FiniteBias, PositionalDropout]
    )
    def test_non_causal_mask_is_kept(self, model_cls):
        node, _ = _run_pass(model_cls(), *_qkv())
        assert _kept(node)

    def test_runtime_mask_is_kept(self):
        q, k, v = _qkv()
        mask = torch.ones(L, L, dtype=torch.bool).tril()
        node, _ = _run_pass(RuntimeMask(), q, k, v, mask)
        assert _kept(node)

    def test_positional_dropout_survives(self):
        """args[:3] truncation would silently drop a positional dropout_p."""
        node, _ = _run_pass(PositionalDropout(), *_qkv())
        assert _args_of(node).get("dropout_p") == 0.5

    def test_mask_narrower_than_key_is_kept(self):
        """tril() of an (L, 1) mask is all-true, so only the extent check rejects this."""

        class NarrowMask(nn.Module):
            def forward(self, q, k, v):
                length = q.shape[-2]
                mask = torch.ones(length, 1, dtype=torch.bool, device=q.device).tril()
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        node, _ = _run_pass(NarrowMask(), *_qkv())
        assert _kept(node)


class TestBroadcastExtents:
    """SDPA broadcasts the mask against (B, H), and is_causal cannot widen."""

    def test_a_widening_mask_is_kept(self):
        node, _ = _run_pass(WideningMask(), *_qkv())
        assert _kept(node)

    def test_a_mask_on_the_query_batch_is_rewritten(self):
        node, _ = _run_pass(QueryBatchMask(), *_qkv(batch=3))
        assert _fired(node)

    def test_an_empty_batch_keeps_the_mask(self):
        """A zero-length batch leaves the mask without a row to compare."""
        node, _ = _run_pass(QueryBatchMask(), *_qkv(batch=0))
        assert _kept(node)


class TestMaskRanks:
    """SDPA broadcasts the mask's leading axes, so rank 2 to rank 4 are all legal."""

    @pytest.mark.parametrize("rank", [2, 3, 4])
    def test_every_broadcastable_rank_is_rewritten(self, rank):
        node, _ = _run_pass(RankedMask(rank), *_qkv())
        assert _fired(node)

    def test_a_per_head_mask_is_kept(self):
        """A mask that differs between two heads cannot become a flag."""
        node, _ = _run_pass(PerHeadMask(), *_qkv())
        assert _kept(node)


class TestUnequalExtents:
    """The nkilib backward kernel asserts that the query and the key length match
    under a causal mask, so unequal extents must keep the mask.
    """

    @pytest.mark.parametrize(("length", "width"), [(4, 8), (8, 4)])
    def test_unequal_extents_keep_the_mask(self, length, width):
        node, _ = _run_pass(CausalMask(), *_qkv(length, width))
        assert _kept(node)

    def test_equal_extents_still_rewrite(self):
        node, _ = _run_pass(CausalMask(), *_qkv(8, 8))
        assert _fired(node)


class TestMaskDtypes:
    """Only a bool mask is rewritten. HuggingFace's ``eager_mask`` builds the
    additive float form, and the SDPA path never calls it.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_a_float_sentinel_mask_is_kept(self, dtype):
        node, _ = _run_pass(SentinelMask(), *_qkv(dtype=dtype))
        assert _kept(node)

    def test_a_zero_and_negative_infinity_mask_is_kept(self):
        node, _ = _run_pass(CausalFloatMask(), *_qkv())
        assert _kept(node)

    def test_a_finite_bias_is_kept(self):
        node, _ = _run_pass(SentinelMask(torch.finfo(torch.float32).min / 4), *_qkv())
        assert _kept(node)


class TestFullyMaskedRow:
    """is_causal lets a query row attend to itself, so a row the mask drops
    entirely is not causal.
    """

    def test_a_dead_row_keeps_the_mask(self):
        node, _ = _run_pass(ComparisonMask(dead_row=True), *_qkv())
        assert _kept(node)

    def test_the_same_comparison_without_the_dead_row_is_rewritten(self):
        node, _ = _run_pass(ComparisonMask(dead_row=False), *_qkv())
        assert _fired(node)


class TestAtenTargets:
    """The pass must match both aten spellings of the call."""

    @pytest.mark.parametrize(
        "target",
        [
            torch.ops.aten.scaled_dot_product_attention,
            torch.ops.aten.scaled_dot_product_attention.default,
        ],
    )
    def test_an_aten_call_is_rewritten(self, target):
        class AtenSdpa(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                return target(q, k, v, mask)

        node, _ = _run_pass(AtenSdpa(), *_qkv(), target=target)
        assert _fired(node)


class TestRebuiltCall:
    """The rewrite discards node.args, so every argument it keeps is rebuilt."""

    def test_scale_survives(self):
        node, _ = _run_pass(ScaledMask(), *_qkv())
        assert _fired(node)
        assert _args_of(node).get("scale") == 0.125

    def test_enable_gqa_survives(self):
        node, _ = _run_pass(GqaMask(), *_qkv(heads=4, kv_heads=2))
        assert _fired(node)
        assert _args_of(node).get("enable_gqa") is True


class TestOperandExtents:
    def test_a_non_tensor_traced_value_is_rejected(self):
        """A traced value that is not a tensor must not raise out of the pass."""
        graph = torch.fx.Graph()
        query, key = graph.placeholder("q"), graph.placeholder("k")
        query.meta["example_value"] = [L, E]
        key.meta["example_value"] = torch.zeros(B, H, L, E)
        assert SdpaCausalMaskRewrite._operand_extents({"query": query, "key": key}) is None


class TestDeviceLiterals:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (torch.device("meta"), torch.device("cpu")),
            ("privateuseone:0", torch.device("cpu")),
            (torch.device("cpu"), torch.device("cpu")),
            (3, 3),
        ],
    )
    def test_only_a_non_cpu_device_becomes_cpu(self, value, expected):
        assert sdpa_causal_mask._cpu_if_device(value) == expected


class TestTwoCones:
    def test_both_masks_are_rewritten(self):
        """Two SDPA calls with separate cones. The memo must not let one mask's
        verdict decide the other's.
        """

        class TwoCones(nn.Module):
            def forward(self, q1, q2, k1, k2, v1, v2):
                size = q1.shape[-2]
                m1 = torch.ones(size, size, dtype=torch.bool, device=q1.device).tril()
                rows = torch.arange(size, device=q2.device).unsqueeze(-1)
                m2 = torch.arange(size, device=q2.device).unsqueeze(0) <= rows
                a = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=m1)
                b = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=m2)
                return a.sum() + b.sum()

        torch.manual_seed(0)
        qkv = [torch.randn(1, 1, L, E) for _ in range(6)]
        gm = _pre_aot(TwoCones(), *qkv)
        PassManager(passes=[SdpaCausalMaskRewrite()])(gm)

        assert sum(_fired(n) for n in _sdpa_nodes(gm)) == 2


class TestErasureScope:
    """Erasure must touch the folded mask producers and nothing else."""

    def test_input_mutation_survives(self):
        class MutatesAnInput(nn.Module):
            def forward(self, q, k, v, scratch):
                scratch += 1
                length, width = q.shape[-2], k.shape[-2]
                mask = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        q, k, v = _qkv()
        scratch = torch.zeros(4)
        node, gm = _run_pass(MutatesAnInput(), q, k, v, scratch)
        assert _fired(node)
        assert any(n.target is operator.iadd for n in gm.graph.nodes)

    def test_a_mask_that_is_also_an_output_survives(self):
        """A cone node that still has a user must keep its producers."""
        q, k, v = _qkv()
        node, gm = _run_pass(ReturnsTheMask(), q, k, v)
        assert _fired(node)
        assert "tril" in {n.target for n in gm.graph.nodes}
        _, mask = gm(q, k, v)
        torch.testing.assert_close(mask, torch.ones(L, L, dtype=torch.bool).tril())

    def test_a_cone_that_both_call_sites_drop_is_erased(self):
        _, gm = _run_pass(TwoCallSites(), *_qkv())
        nodes = _sdpa_nodes(gm)
        assert len(nodes) == 2
        assert all(_fired(n) for n in nodes)
        targets = {n.target for n in gm.graph.nodes}
        assert torch.ones not in targets
        assert "tril" not in targets


class TestSharedMask:
    def test_verdict_is_per_call_not_per_mask(self):
        """One mask node, two call sites, only one of which the pass accepts."""

        class SharedMask(nn.Module):
            def forward(self, q, k, v):
                width = k.shape[-2]
                mask = torch.ones(width, width, dtype=torch.bool, device=k.device).tril()
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                b = F.scaled_dot_product_attention(q[0], k[0], v[0], attn_mask=mask)
                return a.sum() + b.sum()

        result = PassManager(passes=[SdpaCausalMaskRewrite()])(_pre_aot(SharedMask(), *_qkv()))
        gm = result.graph_module
        assert [_fired(n) for n in _sdpa_nodes(gm)] == [True, False]
        assert "tril" in {n.target for n in gm.graph.nodes}


class TestAliasMutation:
    def test_write_through_a_view_is_detected(self):
        """A view hop is not a mutation, so the walk must follow it."""

        class MutatesThroughView(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                alias = base.view(length, width)
                alias[0, width - 1] = True
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())

        node, _ = _run_pass(MutatesThroughView(), *_qkv())
        assert _kept(node)

    def test_write_through_a_cast_is_detected(self):
        """A cast returns its input when the dtype already matches, so the walk
        must follow it like a view.
        """

        class MutatesThroughCast(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                alias = base.to(torch.bool)
                alias[0, width - 1] = True
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())

        node, _ = _run_pass(MutatesThroughCast(), *_qkv())
        assert _kept(node)

    def test_write_through_a_function_view_is_detected(self):
        """Dynamo spells torch.transpose as a call_function, not a call_method."""

        class MutatesThroughTranspose(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                alias = torch.transpose(base, 0, 1)
                alias[0, width - 1] = True
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())

        node, _ = _run_pass(MutatesThroughTranspose(), *_qkv())
        assert _kept(node)

    def test_an_inplace_torch_function_is_detected(self):
        """torch.clamp_ declares no schema, so only its name reports the write."""

        class MutatesInPlace(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, device=q.device).tril()
                torch.clamp_(base, min=1.0)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.bool())

        node, _ = _run_pass(MutatesInPlace(), *_qkv())
        assert _kept(node)

    def test_a_write_through_out_is_detected(self):
        """A torch builtin declares no schema for the write through out=."""

        class MutatesThroughOut(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                torch.logical_not(base, out=base)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())

        node, _ = _run_pass(MutatesThroughOut(), *_qkv())
        assert _kept(node)

    def test_a_live_view_with_no_write_is_rewritten(self):
        """The control for the cases above: a view alone must not block the fold."""
        node, _ = _run_pass(ViewWithoutAWrite(), *_qkv())
        assert _fired(node)

    @pytest.mark.parametrize(
        "view",
        [
            pytest.param(lambda m: m.detach(), id="detach"),
            pytest.param(lambda m: m.diagonal(), id="diagonal"),
            pytest.param(lambda m: m.ravel(), id="ravel"),
            pytest.param(lambda m: m.type_as(m), id="type_as"),
            pytest.param(lambda m: m.T, id="attribute"),
        ],
    )
    def test_write_through_a_storage_sharing_op_is_detected(self, view):
        """Each of these returns a tensor whose data_ptr equals its input's."""

        class MutatesThroughShared(nn.Module):
            def __init__(self, view):
                super().__init__()
                self.view = view

            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, dtype=torch.bool, device=q.device).tril()
                shared = self.view(base)
                shared[0] = True
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.clone())

        node, _ = _run_pass(MutatesThroughShared(view), *_qkv())
        assert _kept(node)

    @pytest.mark.parametrize("positional", [False, True])
    def test_a_write_through_an_inplace_flag_is_detected(self, positional):
        """F.hardtanh(inplace=True) zeroes the mask. It carries no schema and no
        trailing underscore, so the flag is the only evidence of the write.
        """

        class InplaceFunctional(nn.Module):
            def __init__(self, positional):
                super().__init__()
                self.positional = positional

            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, device=q.device).tril()
                if self.positional:
                    F.hardtanh(base, 0.0, 0.0, True)
                else:
                    F.hardtanh(base, 0.0, 0.0, inplace=True)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.bool())

        node, _ = _run_pass(InplaceFunctional(positional), *_qkv())
        assert _kept(node)

    def test_the_same_functional_without_the_flag_is_rewritten(self):
        class PureFunctional(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                base = torch.ones(length, width, device=q.device).tril()
                F.hardtanh(base, 0.0, 1.0)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=base.bool())

        node, _ = _run_pass(PureFunctional(), *_qkv())
        assert _fired(node)

    def test_pure_dunder_user_does_not_block(self):
        """__eq__ ends in an underscore but does not mutate."""

        class PureDunderSibling(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                rows = torch.arange(length, device=q.device).unsqueeze(-1)
                cols = torch.arange(width, device=q.device).unsqueeze(0)
                mask = cols <= rows
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask), cols.__eq__(rows)

        node, _ = _run_pass(PureDunderSibling(), *_qkv())
        assert _fired(node)


class TestMutatedCone:
    """A write to a cone node makes the folded value stale. Each case pairs with
    the same cone without the write.
    """

    def test_setitem_on_a_cone_node_is_detected(self):
        node, _ = _run_pass(WritesTheCone(), *_qkv())
        assert _kept(node)

    def test_the_same_cone_without_the_setitem_is_rewritten(self):
        node, _ = _run_pass(WritesTheCone(write=False), *_qkv())
        assert _fired(node)

    def test_an_inplace_aten_overload_is_detected(self):
        """aten.add_.Tensor spells the overload after its name, so only the
        mutable schema reports the write.
        """
        node, _ = _run_pass(InplaceOverload(), *_qkv())
        assert _kept(node)

    def test_the_same_cone_without_the_overload_is_rewritten(self):
        node, _ = _run_pass(InplaceOverload(write=False), *_qkv())
        assert _fired(node)


class TestIdempotence:
    def test_a_second_run_changes_nothing(self):
        """The pass runs inside a PassManager that can run it again."""
        first = PassManager(passes=[SdpaCausalMaskRewrite()])(_pre_aot(CausalMask(), *_qkv()))
        assert first.modified is True
        printed = first.graph_module.print_readable(print_output=False)

        second = PassManager(passes=[SdpaCausalMaskRewrite()])(first.graph_module)
        assert second.modified is False
        assert second.graph_module.print_readable(print_output=False) == printed
        assert _fired(_sdpa_node(second.graph_module))


class TestDynamicExtents:
    def test_a_dynamic_length_keeps_the_mask(self):
        """Dynamo marks the length dynamic on the second compile of one module and
        passes it as a placeholder, which the cone refuses.
        """
        model = CausalMask()
        first = PassManager(passes=[SdpaCausalMaskRewrite()])(_pre_aot(model, *_qkv(8, 8)))
        assert first.modified is True

        second = PassManager(passes=[SdpaCausalMaskRewrite()])(_pre_aot(model, *_qkv(16, 16)))
        assert second.modified is False
        assert _kept(_sdpa_node(second.graph_module))


class TestPipeline:
    def test_the_preprocessing_pipeline_rewrites_the_mask(self):
        """The pass must stay registered, and the other pipeline members must accept
        the graph it leaves behind. The pipeline manager re-raises a pass exception.
        """
        result = preprocess_graph(_pre_aot(CausalMask(), *_qkv()))
        assert _fired(_sdpa_node(result.graph_module))


class TestNkiGate:
    def test_the_rewrite_unlocks_the_nki_forward_gate(self, monkeypatch):
        """The mask operand is the only reason the gate rejects this call."""
        monkeypatch.setenv("TORCH_NEURONX_ENABLE_NKI_SDPA", "1")
        q, k, v = _qkv(512, 512)
        mask = torch.ones(512, 512, dtype=torch.bool).tril()
        assert _can_use_nki_flash_attention(q, k, v, mask) is False
        assert _can_use_nki_flash_attention(q, k, v, None) is True

        node, _ = _run_pass(CausalMask(), q, k, v)
        assert _fired(node)


class TestDefaultDevice:
    def test_a_non_cpu_default_device_still_folds(self):
        """A producer without a device argument takes the ambient default device.
        The fold must reach a real value anyway.
        """
        gm = _pre_aot(CausalMask(), *_qkv())
        with torch.device("meta"):
            result = PassManager(passes=[SdpaCausalMaskRewrite()])(gm)
        assert _fired(_sdpa_node(result.graph_module))


class TestHostMemory:
    def test_an_expanded_mask_does_not_copy_every_plane(self, monkeypatch):
        """The comparison must read the mask in place. A reshape of an expanded mask
        copies the batch extent times the mask's storage.
        """
        seen = []
        real_equal = torch.equal

        def spy(a, b):
            seen.append(max(a.untyped_storage().nbytes(), b.untyped_storage().nbytes()))
            return real_equal(a, b)

        monkeypatch.setattr(torch, "equal", spy)
        node, _ = _run_pass(ExpandedMask(), *_qkv(batch=4))
        assert _fired(node)
        assert max(seen) == H * L * L, f"an allocation exceeded the mask's storage: {seen}"


class TestFailClosed:
    """The caller re-raises, so the pass must never propagate an exception."""

    @pytest.mark.parametrize("method", ["_is_causal", "_operand_extents", "_fold"])
    def test_a_raise_inside_the_rewrite_keeps_the_mask(self, monkeypatch, method):
        def boom(*_args, **_kwargs):
            raise RuntimeError("induced")

        monkeypatch.setattr(SdpaCausalMaskRewrite, method, boom)
        node, gm = _run_pass(CausalMask(), *_qkv())
        assert _kept(node)
        gm.graph.lint()

    def test_a_string_argument_is_not_read_as_a_device(self):
        """_is_non_cpu_device reads any string without "cpu" as a device. Replacing
        rounding_mode="floor" with a device makes the op raise and the fold fail.
        """

        class StringArgCone(nn.Module):
            def forward(self, q, k, v):
                length, width = q.shape[-2], k.shape[-2]
                rows = torch.arange(length, device=q.device).unsqueeze(-1)
                cols = torch.div(
                    torch.arange(width, device=q.device), 1, rounding_mode="floor"
                ).unsqueeze(0)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=cols <= rows)

        node, _ = _run_pass(StringArgCone(), *_qkv())
        assert _fired(node)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("floor", "floor"),
            ("constant", "constant"),
            ("torch.BoolTensor", "torch.BoolTensor"),
            ("privateuseone:0", torch.device("cpu")),
            ("meta", torch.device("cpu")),
            (torch.device("meta"), torch.device("cpu")),
        ],
    )
    def test_only_a_parseable_device_becomes_cpu(self, value, expected):
        assert sdpa_causal_mask._cpu_if_device(value) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("__and__", False), ("__eq__", False), ("__iadd__", True), ("__setitem__", True)],
    )
    def test_a_mutating_dunder_counts_as_a_write(self, name, expected):
        graph = torch.fx.Graph()
        node = graph.create_node("call_method", name, args=(graph.placeholder("x"),))
        assert sdpa_causal_mask._mutates(node) is expected
