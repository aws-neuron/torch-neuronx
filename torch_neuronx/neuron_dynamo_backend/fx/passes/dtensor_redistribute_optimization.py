"""Eliminate redundant index_select / slice ops from collective reassembly.

DTensor's redistribute() traces to flatten → index_select → reshape patterns that
torch-mlir lowers to element-wise stablehlo.gather. funcol.all_gather_tensor with
gather_dim > 0 produces a chunk + cat chain that lowers to N slices and a
concatenate on a different dim than the gather dim, forcing real data movement.
This pass replaces both with view/permute/reshape (or slice for standalone shards).

Four patterns are handled:
  1. Shard-to-Replicate cat:        wait_tensor → flatten → index_select x N → cat(dim>0)
  2. Replicate-to-Shard cat:        tensor → flatten → index_select x N → cat(dim=0)
  3. Replicate-to-Shard clone:      tensor → flatten → index_select → view → clone
  4. funcol all_gather slice+cat:   wait_tensor → slice x N (dim=0) → cat(dim=1)

Shard-to-Replicate occurs after all_gather reassembles shards into the full replicated tensor.
Replicate-to-Shard cat occurs before reduce_scatter chunks a replicated tensor for distribution.
Replicate-to-Shard clone occurs when a single rank's shard is needed
(i.e. not immediately passed to reduce_scatter).
funcol all_gather slice+cat occurs in sequence-parallel transformer code that calls
funcol.all_gather_tensor(x, gather_dim=1) — the chunk-and-cat reassembly is equivalent
to a single reshape since N is the slowest-varying dim of the all_gather output.
"""

import copy
import functools
import logging
import operator

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass

logger = logging.getLogger(__name__)


class DTensorRedistributeOptimization(NeuronFXPass):
    """Replace redundant DTensor redistribution indexing with view or slice operations.

    Args:
        state (NeuronFXPassState, optional): Shared state used for pass artifact dumping.
            Defaults to ``None``.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        """Optimize supported DTensor redistribution patterns.

        Args:
            graph_module (GraphModule): FX graph module optimized in place.

        Returns:
            PassResult: The same graph module and whether a redistribution pattern changed.
        """
        modified = _optimize_cat_patterns(graph_module) | _optimize_clone_pattern(graph_module)
        return PassResult(graph_module, modified)


# ── Cat Pattern Optimizer (Shard-to-Replicate + Replicate-to-Shard) ──────────


def _optimize_cat_patterns(gm: GraphModule) -> bool:
    """Replace index_select+cat patterns with view+permute+reshape.

    Scans for ``aten.cat`` nodes whose inputs follow the DTensor redistribution
    chain (view([-1]) → index_select → view(chunk_shape)), and replaces the
    entire chain with efficient reshape operations.

    Handles both Shard-to-Replicate (source is wait_tensor after all_gather)
    and Replicate-to-Shard (source is any other tensor) directions.

    Returns:
        True if any patterns were optimized.
    """
    graph = gm.graph
    modified = False

    for node in list(graph.nodes):
        if node.target != torch.ops.aten.cat.default:
            continue

        cat_inputs = node.args[0]
        if not isinstance(cat_inputs, list | tuple) or len(cat_inputs) < 2:
            continue

        cat_dim = node.args[1] if len(node.args) > 1 else 0

        result = _try_shard_to_replicate_cat(node, cat_inputs, cat_dim, graph)
        if result is None:
            result = _try_replicate_to_shard_cat(node, cat_inputs, cat_dim, graph)
        if result is None:
            result = _try_funcol_all_gather_slice_cat(node, cat_inputs, cat_dim, graph)
        if result is None:
            continue

        replacement, _source = result
        node.replace_all_uses_with(replacement)
        modified = True

    if modified:
        _cleanup(gm)
    return modified


def _try_shard_to_replicate_cat(
    cat_node: Node, cat_inputs: list, cat_dim: int, graph: Graph
) -> tuple[Node, Node] | None:
    """Try to optimize Shard-to-Replicate cat: wait_tensor → flatten → idx_sel x N → cat.

    Returns:
        (replacement_node, source_node) if the pattern matches, None otherwise.
    """
    info = _analyze_chain(cat_inputs, expect_wait_tensor=True)
    if info is None:
        return None

    source, num_chunks, chunk_shape, offsets = info
    chunk_size = _product(chunk_shape)
    if offsets != [i * chunk_size for i in range(num_chunks)]:
        return None

    # Verify source covers exactly N chunks (no extra data)
    if not hasattr(source, "meta") or "tensor_meta" not in source.meta:
        return None
    if _product(source.meta["tensor_meta"].shape) != num_chunks * chunk_size:
        return None

    batch, *_ = chunk_shape
    replacement = _build_replacement(
        graph,
        cat_node,
        source,
        num_chunks,
        chunk_shape,
        batch,
        cat_dim,
        unsqueeze_leading=[num_chunks, batch],
        n_dim_idx=0,
    )
    return replacement, source


def _try_replicate_to_shard_cat(
    cat_node: Node, cat_inputs: list, cat_dim: int, graph: Graph
) -> tuple[Node, Node] | None:
    """Try to optimize Replicate-to-Shard cat: tensor → flatten → idx_sel x N → cat(dim=0).

    Reshapes the full tensor into [N*B, S, H] for reduce_scatter to distribute.
    No per-rank shard selection here — reduce_scatter handles that downstream.

    Returns:
        (replacement_node, source_node) if the pattern matches, None otherwise.
    """
    if cat_dim != 0:
        return None

    info = _analyze_chain(cat_inputs, expect_wait_tensor=False)
    if info is None:
        return None

    source, num_chunks, chunk_shape, offsets = info
    if not hasattr(source, "meta") or "tensor_meta" not in source.meta:
        return None
    src_shape = list(source.meta["tensor_meta"].shape)
    batch, *shard_and_rest = chunk_shape

    # Replicate-to-Shard uses interleaved offsets: stride = product(dims after batch)
    shard_stride = _product(shard_and_rest)
    if offsets != [i * shard_stride for i in range(num_chunks)]:
        return None

    # Verify source sharded dim == num_chunks * shard_size
    if src_shape[0] != batch:
        return None
    if _product(src_shape) != num_chunks * _product(chunk_shape):
        return None

    replacement = _build_replacement(
        graph,
        cat_node,
        source,
        num_chunks,
        chunk_shape,
        batch,
        cat_dim=0,
        unsqueeze_leading=[src_shape[0], num_chunks],
        n_dim_idx=1,
    )
    return replacement, source


def _try_funcol_all_gather_slice_cat(
    cat_node: Node, cat_inputs: list, cat_dim: int, graph: Graph
) -> tuple[Node, Node] | None:
    """Try to optimize funcol slice+cat: wait_tensor -> slice x N (dim=0) -> cat(cat_dim>0).

    Pattern (from funcol.all_gather_tensor with gather_dim > 0):
        all_gather_into_tensor → wait_tensor : [N, *rest]
        slice(dim=0, 0:1), slice(dim=0, 1:2), ..., slice(dim=0, N-1:N)
        cat([s_0, ..., s_{N-1}], dim=cat_dim)

    Since the slice dim (0) is the slowest-varying dim of the all_gather output,
    chunk-and-cat reassembly is layout-equivalent to a single reshape that merges
    the leading N into the cat_dim. Replace the chain with one aten.view.

    Returns:
        (replacement_node, source_node) if the pattern matches, None otherwise.
    """
    if cat_dim == 0:
        return None

    num_chunks = len(cat_inputs)
    sources = []
    for i, inp in enumerate(cat_inputs):
        if not _is_op(inp, torch.ops.aten.slice.Tensor):
            return None
        # slice.Tensor signature: (input, dim, start, end[, step])
        if len(inp.args) < 4:
            return None
        slice_dim = inp.args[1]
        slice_start = inp.args[2]
        slice_end = inp.args[3]
        slice_step = inp.args[4] if len(inp.args) > 4 else 1
        if slice_dim != 0 or slice_step != 1:
            return None
        if slice_start != i or slice_end != i + 1:
            return None
        sources.append(inp.args[0])

    if len({id(s) for s in sources}) != 1:
        return None
    source = sources[0]
    if not isinstance(source, Node):
        return None
    if source.target != torch.ops._c10d_functional.wait_tensor.default:
        return None
    if not hasattr(source, "meta") or "tensor_meta" not in source.meta:
        return None

    src_shape = list(source.meta["tensor_meta"].shape)
    if len(src_shape) < 2 or src_shape[0] != num_chunks:
        return None
    if cat_dim >= len(src_shape):
        return None

    # Each slice has shape [1, src_shape[1], ..., src_shape[-1]]. cat(dim=cat_dim) of
    # num_chunks such tensors gives [1, ..., src_shape[cat_dim]*N, ..., src_shape[-1]].
    # This is layout-equivalent to a reshape only when cat_dim == 1, because then the
    # gathered N (slowest-varying dim 0 of the source) merges contiguously with
    # src_shape[1] in the destination's cat_dim. For cat_dim > 1, an interior dim
    # would have to absorb N, which requires an actual transpose.
    if cat_dim != 1:
        return None

    output_shape = list(src_shape)
    output_shape[0] = 1
    output_shape[cat_dim] = src_shape[cat_dim] * num_chunks

    if _product(output_shape) != _product(src_shape):
        return None

    with graph.inserting_before(cat_node):
        replacement = graph.call_function(torch.ops.aten.view.default, args=(source, output_shape))
        replacement.meta = _copy_meta(cat_node)

    return replacement, source


def _build_replacement(
    graph: Graph,
    cat_node: Node,
    source: Node,
    num_chunks: int,
    chunk_shape: list[int],
    batch: int,
    cat_dim: int,
    unsqueeze_leading: list[int],
    n_dim_idx: int,
) -> Node:
    """Build the replacement subgraph: view [→ permute → reshape].

    Args:
        graph: The FX graph to insert new nodes into.
        cat_node: The original cat node being replaced (used for insertion point and meta).
        source: The source tensor node (wait_tensor for Shard-to-Replicate,
            any tensor for Replicate-to-Shard).
        num_chunks: Number of chunks / TP degree (N).
        chunk_shape: Shape of each chunk, e.g. [batch, shard, hidden].
        batch: Batch size (chunk_shape[0]).
        cat_dim: Concatenation dimension of the original cat op.
        unsqueeze_leading: First two dims of the intermediate view shape.
            [N, batch] for Shard-to-Replicate (source is [N*batch, shard, hidden]).
            [batch, N] for Replicate-to-Shard (source is [batch, N*shard, hidden]).
        n_dim_idx: Index of the num_chunks (N) dimension in unsqueeze_leading.
            0 for Shard-to-Replicate, 1 for Replicate-to-Shard.
    """
    if batch == 1 and (cat_dim <= 1):
        # If batch size = 1, we can use a simple view.
        # Data is already contiguous — no need to separate batch from shard dim.
        output_shape = list(chunk_shape)
        output_shape[cat_dim] = output_shape[cat_dim] * num_chunks
        with graph.inserting_before(cat_node):
            replacement = graph.call_function(
                torch.ops.aten.view.default, args=(source, output_shape)
            )
            replacement.meta = _copy_meta(cat_node)
        return replacement

    unsqueeze_shape = unsqueeze_leading + chunk_shape[1:]
    ndim = len(unsqueeze_shape)

    if n_dim_idx == 0:
        # Shard-to-Replicate: move N from front to just before shard dim so reshape can merge them
        # [N, B, S, H] → [B, N, S, H] (for cat_dim=1)
        perm = [*range(1, cat_dim + 1), 0, *range(cat_dim + 1, ndim)]
    else:
        # Replicate-to-Shard: move N to front so reshape produces [N*B, S, H]
        # [B, N, S, H] → [N, B, S, H] (for cat_dim=0)
        perm = [n_dim_idx] + [i for i in range(ndim) if i != n_dim_idx]

    permuted_shape = [unsqueeze_shape[p] for p in perm]
    output_shape = list(chunk_shape)
    output_shape[cat_dim] = chunk_shape[cat_dim] * num_chunks

    with graph.inserting_before(cat_node):
        viewed = graph.call_function(torch.ops.aten.view.default, args=(source, unsqueeze_shape))
        viewed.meta = _make_meta(source.meta, unsqueeze_shape)

        permuted = graph.call_function(torch.ops.aten.permute.default, args=(viewed, perm))
        permuted.meta = _make_meta(viewed.meta, permuted_shape)

        replacement = graph.call_function(
            torch.ops.aten.reshape.default, args=(permuted, output_shape)
        )
        replacement.meta = _copy_meta(cat_node)

    return replacement


# ── Clone Pattern Optimizer (Replicate-to-Shard standalone) ─────────────────────────────────


def _optimize_clone_pattern(gm: GraphModule) -> bool:
    """Replace index_select+clone with slice+clone for standalone Replicate-to-Shard.

    This is where per-rank shard selection happens: slice(dim, rank*S, (rank+1)*S)
    extracts this rank's portion from the full replicated tensor.

    Returns:
        True if any patterns were optimized.
    """
    graph = gm.graph
    modified = False

    for node in list(graph.nodes):
        if node.target != torch.ops.aten.clone.default:
            continue

        view_out = node.args[0]
        if not _is_op(view_out, torch.ops.aten.view.default):
            continue
        output_shape = view_out.args[1]
        if not isinstance(output_shape, list | tuple):
            continue

        idx_sel = view_out.args[0]
        if not _is_op(idx_sel, torch.ops.aten.index_select.default):
            continue

        flatten = idx_sel.args[0]
        if not _is_op(flatten, torch.ops.aten.view.default) or flatten.args[1] != [-1]:
            continue

        source = flatten.args[0]
        if not isinstance(source, Node):
            continue
        if source.target == torch.ops._c10d_functional.wait_tensor.default:
            continue
        if not hasattr(source, "meta") or "tensor_meta" not in source.meta:
            continue

        offset = _extract_offset(idx_sel.args[2])
        if offset is None:
            continue

        input_shape = list(source.meta["tensor_meta"].shape)
        shard_dim, shard_size = _find_shard_dim(input_shape, list(output_shape))
        if shard_dim is None:
            continue

        stride = _product(input_shape[shard_dim + 1 :])
        if stride == 0 or offset % stride != 0:
            continue
        start = offset // stride

        with graph.inserting_before(node):
            sliced = graph.call_function(
                torch.ops.aten.slice.Tensor, args=(source, shard_dim, start, start + shard_size)
            )
            sliced.meta = _copy_meta(view_out)

            new_clone = graph.call_function(torch.ops.aten.clone.default, args=(sliced,))
            new_clone.meta = _copy_meta(node)

        node.replace_all_uses_with(new_clone)
        modified = True

    if modified:
        _cleanup(gm)
    return modified


# ── Pattern Analysis ─────────────────────────────────────────────────────────


def _analyze_chain(
    cat_inputs: list, expect_wait_tensor: bool
) -> tuple[Node, int, list[int], list[int]] | None:
    """Analyze a chain of cat inputs back to a common source.

    Each cat input is expected to follow the DTensor redistribution chain:
        source → view([-1]) → index_select(flat, 0, indices+offset) → view(chunk_shape)

    All inputs must share the same source node and chunk_shape.

    Args:
        cat_inputs: List of nodes feeding into the cat op.
        expect_wait_tensor: If True, source must be a wait_tensor (Shard-to-Replicate).
            If False, source must NOT be a wait_tensor (Replicate-to-Shard).

    Returns:
        (source_node, num_chunks, chunk_shape, offsets) where:
        - source_node: The common source tensor all chains trace back to.
        - num_chunks: Number of cat inputs (== TP degree).
        - chunk_shape: Shape each chunk is reshaped to, e.g. [batch, shard, hidden].
            Identical across all inputs (validated).
        - offsets: List of integer offsets from each index_select's index computation.
        Returns None if the pattern doesn't match.
    """
    sources = []
    chunk_shapes = []
    offsets = []

    for inp in cat_inputs:
        # Each input should be: view(index_select_result, chunk_shape)
        if not _is_op(inp, torch.ops.aten.view.default):
            return None
        shape = inp.args[1]
        if not isinstance(shape, list | tuple):
            return None
        chunk_shapes.append(list(shape))

        # The view's input should be an index_select on a flattened tensor
        idx_sel = inp.args[0]
        if not _is_op(idx_sel, torch.ops.aten.index_select.default):
            return None

        # Extract the constant offset: indices are computed as (arange + offset)
        offset = _extract_offset(idx_sel.args[2])
        if offset is None:
            return None
        offsets.append(offset)

        # The index_select reads from a flatten: view(source, [-1])
        flatten = idx_sel.args[0]
        if not _is_op(flatten, torch.ops.aten.view.default) or flatten.args[1] != [-1]:
            return None

        # Trace to the original source tensor
        source = flatten.args[0]
        if not isinstance(source, Node):
            return None

        # Distinguish Shard-to-Replicate (wait_tensor) from Replicate-to-Shard (other)
        is_wait = source.target == torch.ops._c10d_functional.wait_tensor.default
        if is_wait != expect_wait_tensor:
            return None
        sources.append(source)

    # All inputs must trace back to the same source tensor
    if len({id(s) for s in sources}) != 1:
        return None
    # All chunks must have identical shapes
    if not all(s == chunk_shapes[0] for s in chunk_shapes):
        return None

    return sources[0], len(cat_inputs), chunk_shapes[0], offsets


# ── Utilities ────────────────────────────────────────────────────────────────


def _extract_offset(node: Node) -> int | None:
    """Extract the constant integer offset from an add(indices, OFFSET) node."""
    if not _is_op(node, torch.ops.aten.add.Tensor) or len(node.args) < 2:
        return None
    return node.args[1] if isinstance(node.args[1], int) else None


def _find_shard_dim(
    input_shape: list[int], output_shape: list[int]
) -> tuple[int | None, int | None]:
    """Find the single dimension that differs between input and output shapes.

    Used to identify which dimension was sharded during Replicate-to-Shard redistribution.
    Returns (shard_dim, shard_size) or (None, None) if zero or multiple dims differ.
    """
    shard_dim = None
    shard_size = None
    for i, (a, b) in enumerate(zip(input_shape, output_shape, strict=False)):
        if a != b:
            if shard_dim is not None:
                return None, None
            shard_dim = i
            shard_size = b
    return shard_dim, shard_size


def _is_op(node, target) -> bool:
    """Check if node is an FX Node with the given op target."""
    return isinstance(node, Node) and node.target == target


def _product(dims: list[int]) -> int:
    """Return the product of all elements in dims."""
    return functools.reduce(operator.mul, dims, 1)


def _make_meta(src_meta: dict, new_shape: list[int]) -> dict:
    """Create a copy of node metadata with an updated tensor shape.

    Required because torch-mlir reads tensor_meta from every node during lowering.
    """
    meta = copy.copy(src_meta)
    if "tensor_meta" in meta:
        # Compute contiguous strides for the new shape
        strides = []
        s = 1
        for d in reversed(new_shape):
            strides.append(s)
            s *= d
        strides.reverse()
        meta["tensor_meta"] = copy.copy(meta["tensor_meta"])._replace(
            shape=torch.Size(new_shape), stride=tuple(strides)
        )
    if "val" in meta and isinstance(meta["val"], torch.Tensor):
        meta["val"] = torch.empty(new_shape, dtype=meta["val"].dtype, device="meta")
    return meta


def _copy_meta(node: Node) -> dict:
    """Return a shallow copy of a node's metadata dict."""
    return node.meta.copy() if node.meta else {}


def _cleanup(gm: GraphModule):
    """Remove dead nodes and recompile."""
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
