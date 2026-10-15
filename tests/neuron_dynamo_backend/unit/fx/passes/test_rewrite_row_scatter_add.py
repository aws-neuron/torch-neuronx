"""
Unit tests for the rewrite_row_scatter_add FX pass.

The pass rewrites ``aten.scatter_add.default(x, 0, expand(idx_1d), src)`` to
``aten.index_put.default(x, [idx_1d, None, ...], src, accumulate=True)`` so the
downstream StableHLO scatter lowers with a contiguous window over the
non-scatter dims. See ``rewrite_row_scatter_add.py`` for the motivation
(NCF-1926: 128x1 MoE combine scatter).
"""

from __future__ import annotations

import pytest
import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend.fx.passes.rewrite_row_scatter_add import (
    RewriteRowScatterAdd,
)
from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import _add_metadata_to_node

# ============================================================================
# Helpers
# ============================================================================


def _count_ops(gm: fx.GraphModule, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target == target)


def _find_op(gm: fx.GraphModule, target) -> fx.Node | None:
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target == target:
            return n
    return None


def _build_row_scatter_add_graph(
    self_shape: tuple[int, ...],
    idx_1d_len: int,
    *,
    dim: int = 0,
    idx_pre_expand_shape: tuple[int, ...] | None = None,
    expand_op=torch.ops.aten.expand.default,
    dtype: torch.dtype = torch.bfloat16,
    idx_dtype: torch.dtype = torch.int64,
    expand_sizes: list | None = None,
) -> fx.GraphModule:
    """Build an FX graph shaped like the MoE row-broadcast scatter_add.

    idx_pre_expand_shape defaults to ``[N, 1, ..., 1]`` matching the operand
    rank; pass an explicit tuple to build non-conforming shapes for the
    negative tests.
    """
    # scatter_add requires index.shape == src.shape. In the row-broadcast case
    # every non-``dim`` axis of the destination matches src too.
    src_shape = (idx_1d_len,) + self_shape[1:]

    if idx_pre_expand_shape is None:
        idx_pre_expand_shape = (idx_1d_len,) + (1,) * (len(self_shape) - 1)

    graph = fx.Graph()
    x = graph.placeholder("x")
    _add_metadata_to_node(x, shape=self_shape, dtype=dtype)

    idx_1d = graph.placeholder("idx_1d")
    _add_metadata_to_node(idx_1d, shape=idx_pre_expand_shape, dtype=idx_dtype)

    src = graph.placeholder("src")
    _add_metadata_to_node(src, shape=src_shape, dtype=dtype)

    # expand_sizes lets a test pass a -1 (keep-dim) sentinel like the real
    # graph does (expand(view[N, 1], [-1, H])); the output meta stays src_shape.
    idx_expanded = graph.call_function(
        expand_op, args=(idx_1d, expand_sizes if expand_sizes is not None else list(src_shape))
    )
    _add_metadata_to_node(idx_expanded, shape=src_shape, dtype=idx_dtype)

    node = graph.call_function(torch.ops.aten.scatter_add.default, args=(x, dim, idx_expanded, src))
    _add_metadata_to_node(node, shape=self_shape, dtype=dtype)
    graph.output(node)
    return fx.GraphModule(torch.nn.Module(), graph)


# ============================================================================
# Positive tests: rewrite fires
# ============================================================================


class TestRewriteFires:
    """The pass rewrites row-broadcast scatter_add into index_put."""

    def test_rewrites_2d_expand(self):
        """The MoE unpermute+combine case: [2048, 4096] += [4096, 4096] via
        expand(idx_1d[4096], [4096, 4096])."""
        gm = _build_row_scatter_add_graph(self_shape=(2048, 4096), idx_1d_len=4096)
        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is True
        assert _count_ops(gm, torch.ops.aten.scatter_add.default) == 0
        put = _find_op(gm, torch.ops.aten.index_put.default)
        assert put is not None
        # index_put.default(self, indices, values), kwargs={"accumulate": True}
        assert put.args[0].name == "x"
        indices = put.args[1]
        # A single 1-D index over dim 0; no trailing None. torch-mlir turns any
        # extra list entry (including None) into an inserted window dim, which
        # would reproduce the scalar 128x1 scatter.
        assert len(indices) == 1
        assert put.args[2].name == "src"
        assert put.kwargs == {"accumulate": True}
        # The dim-0 index must be 1-D [N]; a [N, 1] index would raise at runtime.
        idx_meta = indices[0].meta.get("val")
        assert idx_meta is not None and idx_meta.dim() == 1, "index must be 1-D"
        # The dead expand that built the full [N, H] index must be gone.
        assert _count_ops(gm, torch.ops.aten.expand.default) == 0

    def test_rewrites_3d_expand(self):
        """Row-broadcast into a 3-D operand keeps the trailing dims as None."""
        gm = _build_row_scatter_add_graph(self_shape=(128, 16, 64), idx_1d_len=256)
        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is True
        put = _find_op(gm, torch.ops.aten.index_put.default)
        assert put is not None
        indices = put.args[1]
        # Single 1-D index, no trailing None even for a 3-D operand.
        assert len(indices) == 1
        idx_meta = indices[0].meta.get("val")
        assert idx_meta is not None and idx_meta.dim() == 1
        assert _count_ops(gm, torch.ops.aten.expand.default) == 0

    def test_rewrites_expand_keepdim_sentinel(self):
        """Real graphs emit expand(view[N, 1], [-1, H]); the -1 sentinel must
        not defeat the shape match (regression for NCF-1926 235B run)."""
        gm = _build_row_scatter_add_graph(
            self_shape=(2048, 4096), idx_1d_len=4096, expand_sizes=[-1, 4096]
        )
        result = RewriteRowScatterAdd().call(gm)
        assert result.modified is True
        assert _count_ops(gm, torch.ops.aten.scatter_add.default) == 0
        assert _find_op(gm, torch.ops.aten.index_put.default) is not None

    def test_rewrites_broadcast_to(self):
        """The pass accepts aten.broadcast_to as well as aten.expand."""
        gm = _build_row_scatter_add_graph(
            self_shape=(2048, 4096),
            idx_1d_len=4096,
            expand_op=torch.ops.aten.broadcast_to.default,
        )
        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is True
        assert _find_op(gm, torch.ops.aten.index_put.default) is not None


# ============================================================================
# Negative tests: rewrite does NOT fire (semantics-preserving)
# ============================================================================


class TestRewriteDoesNotFire:
    """The pass leaves the graph unchanged when the row-broadcast pattern is
    not proven."""

    def test_skips_non_zero_dim(self):
        """dim != 0 leaves the scatter_add alone."""
        gm = _build_row_scatter_add_graph(self_shape=(2048, 4096), idx_1d_len=4096, dim=1)
        before = _count_ops(gm, torch.ops.aten.scatter_add.default)
        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is False
        assert _count_ops(gm, torch.ops.aten.scatter_add.default) == before
        assert _find_op(gm, torch.ops.aten.index_put.default) is None

    def test_skips_1d_operand(self):
        """A rank-1 scatter_add has no window dim to promote; skip."""
        # Build a 1-D scatter_add graph by hand: no expand at all.
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(2048,), dtype=torch.bfloat16)
        idx = graph.placeholder("idx")
        _add_metadata_to_node(idx, shape=(4096,), dtype=torch.int64)
        src = graph.placeholder("src")
        _add_metadata_to_node(src, shape=(4096,), dtype=torch.bfloat16)
        node = graph.call_function(torch.ops.aten.scatter_add.default, args=(x, 0, idx, src))
        _add_metadata_to_node(node, shape=(2048,), dtype=torch.bfloat16)
        graph.output(node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is False
        assert _find_op(gm, torch.ops.aten.scatter_add.default) is not None
        assert _find_op(gm, torch.ops.aten.index_put.default) is None

    def test_skips_non_row_broadcast_index(self):
        """An index whose pre-expand shape is not [N, 1, ..., 1] is not a
        proven row-broadcast. Genuine per-element indices must not be
        rewritten."""
        # Pre-expand shape is [4096, 4096] — a materialized 2-D index (no
        # broadcast structure). The pass must skip.
        gm = _build_row_scatter_add_graph(
            self_shape=(2048, 4096),
            idx_1d_len=4096,
            idx_pre_expand_shape=(4096, 4096),
        )
        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is False
        assert _find_op(gm, torch.ops.aten.scatter_add.default) is not None
        assert _find_op(gm, torch.ops.aten.index_put.default) is None

    def test_skips_no_expand(self):
        """A scatter_add whose index is not an expand/broadcast_to node is
        left alone (structure not provable from the FX graph)."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(2048, 4096), dtype=torch.bfloat16)
        idx = graph.placeholder("idx")
        # A 2-D index that did not come from an expand.
        _add_metadata_to_node(idx, shape=(4096, 4096), dtype=torch.int64)
        src = graph.placeholder("src")
        _add_metadata_to_node(src, shape=(4096, 4096), dtype=torch.bfloat16)
        node = graph.call_function(torch.ops.aten.scatter_add.default, args=(x, 0, idx, src))
        _add_metadata_to_node(node, shape=(2048, 4096), dtype=torch.bfloat16)
        graph.output(node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = RewriteRowScatterAdd().call(gm)

        assert result.modified is False


# ============================================================================
# Correctness: run the pass on a real FX graph and execute the result
# ============================================================================


def _build_row_gather_graph(
    self_shape: tuple[int, ...],
    idx_1d_len: int,
    *,
    dim: int = 0,
    expand_sizes: list | None = None,
    dtype: torch.dtype = torch.float32,
) -> fx.GraphModule:
    """gather(x, 0, expand(idx[N, 1, ...], out_shape)) — the backward twin of
    the row-broadcast scatter_add."""
    trailing = self_shape[1:]
    out_shape = (idx_1d_len, *trailing)
    graph = fx.Graph()
    x = graph.placeholder("x")
    _add_metadata_to_node(x, shape=self_shape, dtype=dtype)
    idx_1d = graph.placeholder("idx_1d")
    _add_metadata_to_node(idx_1d, shape=(idx_1d_len,) + (1,) * len(trailing), dtype=torch.int64)
    idx_expanded = graph.call_function(
        torch.ops.aten.expand.default,
        args=(idx_1d, expand_sizes if expand_sizes is not None else list(out_shape)),
    )
    _add_metadata_to_node(idx_expanded, shape=out_shape, dtype=torch.int64)
    node = graph.call_function(torch.ops.aten.gather.default, args=(x, dim, idx_expanded))
    _add_metadata_to_node(node, shape=out_shape, dtype=dtype)
    graph.output(node)
    return fx.GraphModule(torch.nn.Module(), graph)


class TestRewriteGather:
    """The gather twin: gather(x, 0, expand(idx)) -> index(x, [idx_1d])."""

    def test_rewrites_row_gather(self):
        gm = _build_row_gather_graph((2048, 4096), 4096, expand_sizes=[-1, 4096])
        result = RewriteRowScatterAdd().call(gm)
        assert result.modified is True
        assert _count_ops(gm, torch.ops.aten.gather.default) == 0
        idx_node = _find_op(gm, torch.ops.aten.index.Tensor)
        assert idx_node is not None
        indices = idx_node.args[1]
        assert len(indices) == 1
        assert indices[0].meta.get("val").dim() == 1
        assert _count_ops(gm, torch.ops.aten.expand.default) == 0

    def test_skips_topk_gather(self):
        """A genuine gather along the last dim (router top-k) is untouched."""
        graph = fx.Graph()
        a = graph.placeholder("a")
        _add_metadata_to_node(a, shape=(1, 512, 8), dtype=torch.float32)
        i = graph.placeholder("i")
        _add_metadata_to_node(i, shape=(1, 512, 8), dtype=torch.int64)
        g = graph.call_function(torch.ops.aten.gather.default, args=(a, -1, i))
        _add_metadata_to_node(g, shape=(1, 512, 8), dtype=torch.float32)
        graph.output(g)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        assert RewriteRowScatterAdd().call(gm).modified is False

    def test_gather_execution_parity(self):
        """Run the pass and execute; index(x,[idx]) must equal the gather."""
        t, h, n = 8, 4, 16
        gm = _build_row_gather_graph((t, h), n)
        x = torch.randn(t, h)
        idx_pre = torch.randint(0, t, (n, 1))
        ref = gm(x, idx_pre)
        assert RewriteRowScatterAdd().call(gm).modified is True
        assert _count_ops(gm, torch.ops.aten.gather.default) == 0
        torch.testing.assert_close(gm(x, idx_pre), ref)


class TestPassExecutionParity:
    """Apply the pass to an executable FX graph and run it. The rewritten
    graph must produce the same tensor as the original scatter_add, and must
    not raise (a non-1-D index would raise a shape-mismatch at runtime).

    Duplicate-target cases (top-k MoE combine) confirm accumulate semantics
    survive the rewrite; that is the case the NKI scatter_add adapter rejects.
    """

    def _run_parity(self, self_shape, idx_values, dtype=torch.float32):
        """Build ``scatter_add(x, 0, expand(idx), src)``, capture its output,
        rewrite it, and assert the rewritten graph runs and matches."""
        n = int(idx_values.shape[0])
        trailing = self_shape[1:]
        src_shape = (n, *trailing)
        gm = _build_row_scatter_add_graph(self_shape, n, dtype=dtype)

        torch.manual_seed(0)
        x = torch.randn(self_shape, dtype=dtype)
        # The graph's idx placeholder is [N, 1, ..., 1] (pre-expand shape).
        idx_pre = idx_values.reshape((n,) + (1,) * len(trailing))
        src = torch.randn(src_shape, dtype=dtype)

        # Reference: the original graph (a genuine scatter_add) before rewrite.
        reference = gm(x, idx_pre, src)

        result = RewriteRowScatterAdd().call(gm)
        assert result.modified is True
        assert _count_ops(gm, torch.ops.aten.scatter_add.default) == 0

        # Executes the rewritten index_put graph; a [N, 1] index would raise.
        rewritten = gm(x, idx_pre, src)
        # scatter_add and index_put(accumulate) can sum duplicate-target rows
        # in a different order, so a low-precision dtype differs by rounding.
        # Use a dtype-appropriate tolerance rather than expecting bit-equality.
        if dtype == torch.bfloat16:
            torch.testing.assert_close(rewritten, reference, rtol=0.05, atol=0.03)
        else:
            torch.testing.assert_close(rewritten, reference)

    def test_unique_targets(self):
        idx = torch.randperm(8).to(torch.int64)
        self._run_parity((8, 4), idx)

    def test_top_k_duplicate_targets(self):
        # N = T * top_k (top_k = 2): every row is targeted twice, so accumulate
        # must add both updates.
        idx = torch.arange(8, dtype=torch.int64).repeat(2)
        self._run_parity((8, 4), idx)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtypes(self, dtype):
        idx = torch.randint(0, 16, (32,), dtype=torch.int64)
        self._run_parity((16, 8), idx, dtype=dtype)

    def test_3d_operand(self):
        idx = torch.randint(0, 128, (256,), dtype=torch.int64)
        self._run_parity((128, 16, 64), idx)
