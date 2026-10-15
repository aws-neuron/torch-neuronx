"""
Rewrite a row-indexed scatter_add / gather into windowed indexing.

The forward MoE combine (``t[idx] += src``) traces to
``aten.scatter_add(t, 0, expand(idx_1d), src)``; its backward traces to the
mirror gather ``aten.gather(grad, 0, expand(idx_1d))``. Both are rewritten to
the windowed single-1-D-index forms (``index_put(t, [idx_1d], ..., accumulate)``
and ``index(t, [idx_1d])``). See the scatter_add motivation below; the gather
case is identical with "gather"/"index" in place of "scatter_add"/"index_put".

Motivation:
    ``t[idx] += src`` on a 2-D or higher tensor (for example the MoE
    unpermute/combine step in torchtitan) traces to
    ``aten.scatter_add.default(t, 0, index, src)`` where ``index`` is a 1-D
    tensor of row ids that PyTorch broadcasts to ``src.shape``. The generic
    torch-mlir lowering of ``scatter_add`` emits a StableHLO ``scatter`` with
    ``update_window_dims={}`` and every operand dim in
    ``inserted_window_dims``. Every element of ``src`` becomes an individual
    indexed atomic add. The Neuron tensorizer then tiles the scatter to
    ``128x1`` (partition = 128, free = 1) which produces very many
    single-element DMAs and stores.

    The equivalent ``aten.index_put.default(t, [idx_1d], src,
    accumulate=True)`` (a single 1-D index over dim 0, the trailing dims left
    unindexed) lowers to a StableHLO ``scatter`` with the trailing dims in
    ``update_window_dims``. The Neuron tensorizer then tiles the hidden dim as
    a contiguous free axis (``128x512``) and eliminates the 128x1 pathology.

    Note: the index list must NOT be padded with ``None`` for the trailing
    dims. The torch-mlir AtenIndexPutHackedTwin converter counts every list
    entry (``None`` included) as an inserted window dim, which reproduces the
    scalar 128x1 scatter.

    See the NCF-1926 investigation for the full IR trace.

Scope:
    Fires only when ALL of these are true (semantics-preserving rewrite):
      - target op is ``aten.scatter_add.default``
      - the scatter dim is 0
      - the operand rank is at least 2
      - the ``index`` FX node is the direct result of an
        ``aten.expand.default`` (or ``aten.broadcast_to.default``) whose
        input has shape ``[N, 1, ..., 1]`` (a 1-D row-id tensor with unit
        dims added along every non-scatter axis)
      - the expand target shape matches ``src.shape``

    This makes the row-broadcast structure of the index provable from the
    FX graph. If any condition does not hold, the pass is a no-op.

    The pass does NOT try to prove row-broadcast structure by value equality
    or by inspecting arbitrary index construction chains; that would need a
    host synchronization or brittle pattern matching that could miscompile.

Placement:
    Runs in ``post_aot_processing_pipeline`` BEFORE
    ``LegalizeIndexIndices``. LegalizeIndexIndices normalizes the negative
    indices on any ``aten.index_put.default`` the pipeline emits, so this
    pass must run first to give it the rewritten form.
"""

from __future__ import annotations

import logging

import torch
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass
from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import _add_metadata_to_node

logger = logging.getLogger(__name__)

# The row-broadcast form the pass looks for.
_EXPAND_OPS = {
    torch.ops.aten.expand.default,
    torch.ops.aten.broadcast_to.default,
}


def _is_scatter_add_default(node: Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.scatter_add.default


def _is_gather_default(node: Node) -> bool:
    return node.op == "call_function" and node.target is torch.ops.aten.gather.default


def _get_val_meta(node: Node) -> torch.Tensor | None:
    """Return the ``val`` FakeTensor stored on the FX node's meta, or None."""
    if not isinstance(node, Node):
        return None
    return node.meta.get("val", None)


def _expand_source_1d_index(index_node: Node, expected_shape: tuple[int, ...]) -> Node | None:
    """Return the 1-D row-id source of a row-broadcast index, or None.

    A row-broadcast index is one that comes from
    ``idx_1d.reshape([N, 1, ..., 1]).expand(expected_shape)``: a 1-D tensor
    of row ids with unit dims added on every non-scatter axis, then expanded
    to ``expected_shape``. The ``[N, 1, ..., 1]`` source node is returned;
    every other shape is rejected. The caller reshapes it to ``[N]`` before
    use, because advanced indexing needs a 1-D index.
    """
    if not (index_node.op == "call_function" and index_node.target in _EXPAND_OPS):
        return None

    # aten.expand.default signature: expand(self, sizes) -> Tensor
    # aten.broadcast_to.default signature: broadcast_to(self, size) -> Tensor
    if len(index_node.args) < 2:
        return None
    src_node = index_node.args[0]
    if not isinstance(src_node, Node):
        return None

    src_meta = _get_val_meta(src_node)
    if src_meta is None:
        return None
    # The expand OUTPUT shape must match src.shape (aten.scatter_add requires
    # index.shape == src.shape). Compare the expand's output meta shape, not
    # its size argument: the argument can carry -1 (keep-dim) sentinels -- for
    # example expand(view[N, 1], [-1, H]) -- which never equal the resolved
    # shape.
    out_meta = _get_val_meta(index_node)
    if out_meta is None or tuple(out_meta.shape) != expected_shape:
        return None

    # The pre-expand shape must be [N, 1, 1, ..., 1] of the same rank as the
    # expand target. This proves the source is 1-D content broadcast along
    # every non-scatter axis.
    src_shape = tuple(src_meta.shape)
    if len(src_shape) != len(expected_shape):
        return None
    if src_shape[0] != expected_shape[0]:
        return None
    if not all(s == 1 for s in src_shape[1:]):
        return None

    return src_node


class RewriteRowScatterAdd(NeuronFXPass):
    """Rewrite a row-broadcast scatter_add / gather to windowed indexing.

    Two mirror-image rewrites, both gated on a provable row-broadcast index
    (dim 0, operand rank >= 2, index = expand/broadcast_to of a
    ``[N, 1, ..., 1]`` source):

    - ``scatter_add(x, 0, expand(idx_1d), src)``
        -> ``index_put(x, [idx_1d], src, accumulate=True)``
    - ``gather(x, 0, expand(idx_1d))``
        -> ``index(x, [idx_1d])``  (the ``aten.Tensor`` overload)

    The scatter_add rewrite preserves accumulate semantics (top-k MoE combine
    sends several updates to the same row). The gather is the backward twin of
    that combine. Both windowed forms lower to a StableHLO scatter/gather with
    the trailing dims as window dims, so the tensorizer tiles the hidden dim as
    a contiguous free axis (128x512) instead of the scalar 128x1 form.
    """

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        graph = gm.graph
        modified = False

        for node in [n for n in graph.nodes if _is_scatter_add_default(n)]:
            modified |= self._rewrite_scatter_add(graph, node)
        for node in [n for n in graph.nodes if _is_gather_default(n)]:
            modified |= self._rewrite_gather(graph, node)

        if modified:
            # Drop the now-dead expand/broadcast_to that materialized the
            # full [N, ..., H] index; leaving it re-introduces the large
            # broadcast this pass removes.
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()

        return PassResult(gm, modified=modified)

    def _row_broadcast_1d_index(self, graph, node, index_node, expected_shape):
        """Return a 1-D [N] index node for a row-broadcast index, or None.

        ``expected_shape`` is the shape the expanded index must equal (the src
        shape for scatter_add, the output shape for gather). Reshapes the
        pre-expand [N, 1, ..., 1] source to [N] and returns the new node.
        """
        idx_src = _expand_source_1d_index(index_node, expected_shape)
        if idx_src is None:
            return None
        idx_src_meta = _get_val_meta(idx_src)
        n = idx_src_meta.shape[0]
        with graph.inserting_before(node):
            idx_1d = graph.call_function(torch.ops.aten.reshape.default, args=(idx_src, [n]))
            _add_metadata_to_node(idx_1d, shape=(n,), dtype=idx_src_meta.dtype)
        return idx_1d

    def _preflight(self, node, index_node):
        """Shared gate: dim 0 (normalized), operand rank >= 2. Returns rank or
        None. A rank-1 op has no window dim to promote."""
        # signature: op(self, dim, index, ...) -> Tensor
        self_node, dim = node.args[0], node.args[1]
        if not isinstance(index_node, Node):
            return None
        self_meta = _get_val_meta(self_node)
        if self_meta is None:
            return None
        rank = self_meta.dim()
        if rank < 2 or dim % rank != 0:
            return None
        return rank

    def _rewrite_scatter_add(self, graph, node) -> bool:
        # aten.scatter_add.default(self, dim, index, src) -> Tensor
        if len(node.args) != 4:
            return False
        self_node, _dim, index_node, src_node = node.args
        if not isinstance(src_node, Node) or self._preflight(node, index_node) is None:
            return False
        src_meta = _get_val_meta(src_node)
        if src_meta is None:
            return False
        idx_1d = self._row_broadcast_1d_index(graph, node, index_node, tuple(src_meta.shape))
        if idx_1d is None:
            return False
        with graph.inserting_before(node):
            # A single 1-D index over dim 0; trailing dims unindexed so
            # torch-mlir windows them. Never pad with None (each list entry
            # becomes an inserted window dim -> scalar 128x1 scatter).
            new_node = graph.call_function(
                torch.ops.aten.index_put.default,
                args=(self_node, [idx_1d], src_node),
                kwargs={"accumulate": True},
            )
        self._replace(graph, node, new_node)
        logger.debug(f"RewriteRowScatterAdd: {node.name} scatter_add -> index_put")
        return True

    def _rewrite_gather(self, graph, node) -> bool:
        # aten.gather.default(self, dim, index, *, sparse_grad=False) -> Tensor
        if len(node.args) < 3:
            return False
        self_node, _dim, index_node = node.args[0], node.args[1], node.args[2]
        if self._preflight(node, index_node) is None:
            return False
        # gather output shape == index shape == expand output shape.
        out_meta = _get_val_meta(node)
        if out_meta is None:
            return False
        idx_1d = self._row_broadcast_1d_index(graph, node, index_node, tuple(out_meta.shape))
        if idx_1d is None:
            return False
        with graph.inserting_before(node):
            # index(self, [idx_1d]): a row gather with a single 1-D index. The
            # trailing dims become gather offset (window) dims, so the hidden
            # dim stays contiguous instead of the scalar 128x1 gather.
            new_node = graph.call_function(torch.ops.aten.index.Tensor, args=(self_node, [idx_1d]))
        self._replace(graph, node, new_node)
        logger.debug(f"RewriteRowScatterAdd: {node.name} gather -> index")
        return True

    @staticmethod
    def _replace(graph, node, new_node) -> None:
        node_meta = _get_val_meta(node)
        if node_meta is not None:
            _add_metadata_to_node(new_node, shape=tuple(node_meta.shape), dtype=node_meta.dtype)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
