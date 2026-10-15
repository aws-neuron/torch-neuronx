"""
FX Passes: ShardIndexInjection + VocabShardingSPMD

Two passes for SPMD vocab sharding that make compiled graphs identical
across all TP ranks:

1. ShardIndexInjection: Appends a vocab_shard_index int32 scalar placeholder to every
   graph. The shard index is inferred from vocab sharding pattern constants
   (lower_bound // shard_size). The Executor injects torch.tensor(shard_index)
   at runtime via the IOMutation interface.

2. VocabShardingSPMD: Rewrites per-shard vocab sharding constants to
   expressions of vocab_shard_index:
   - Forward triad: lt/ge/bitwise_or/sub from RowwiseParallel embedding
   - Backward flat-offset: add.Tensor feeding scatter_add
   - Backward slice: slice.Tensor(grad, 0, start, end) for vocab gradient

Disable with TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE=1.
"""

import logging
import os
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, Node
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, MutationKind
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)

logger = logging.getLogger(__name__)


def _fake_i64_scalar():
    """Create a FakeTensor scalar for node metadata."""
    with FakeTensorMode():
        return torch.empty((), dtype=torch.int32)


@dataclass
class VocabTriad:
    """Store a connected vocabulary-sharding comparison triad.

    Args:
        lt_node (Node): Lower-bound comparison node.
        ge_node (Node): Upper-bound comparison node.
        bitwise_or_node (Node): Node combining the comparison results.
        sub_node (Node | None): Associated offset-subtraction node, if found.
        lower_bound (int): Inclusive lower vocabulary bound.
        upper_bound (int): Exclusive upper vocabulary bound.
    """

    lt_node: Node
    ge_node: Node
    bitwise_or_node: Node
    sub_node: Node | None
    lower_bound: int
    upper_bound: int

    @property
    def shard_size(self) -> int:
        """Compute the vocabulary shard size.

        Returns:
            int: Difference between the upper and lower bounds.
        """
        return self.upper_bound - self.lower_bound


@dataclass
class FlatOffsetAdd:
    """Store a matched constant-offset addition that feeds a scatter.

    Args:
        node (Node): ``add.Tensor`` node that applies the constant offset.
        flat_shard_size (int): Length of the matched one-dimensional int64 input.

    Note:
        The match requires a nonnegative offset divisible by ``flat_shard_size``.
        The addition must feed a dimension-zero ``scatter_add`` operation.
    """

    node: Node
    flat_shard_size: int  # N = shard_size * hidden_dim


def find_flat_offset_adds(graph: Graph) -> list[FlatOffsetAdd]:
    """Find backward flat-offset additions for row-wise embeddings.

    Args:
        graph (Graph): FX graph to search.

    Returns:
        list[FlatOffsetAdd]: Matches in graph order.

    Note:
        A match adds a nonnegative multiple of a one-dimensional int64 input's
        length and feeds dimension zero of ``scatter_add``.
    """
    results = []
    for node in graph.nodes:
        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.add.Tensor
            and len(node.args) >= 2
        ):
            continue

        const = node.args[1]
        if not isinstance(const, int) or const < 0:
            continue

        # Check first arg has 1-D shape [N]
        input_node = node.args[0]
        if not hasattr(input_node, "meta") or "val" not in input_node.meta:
            continue
        val = input_node.meta["val"]
        if val.dim() != 1 or val.shape[0] == 0:
            continue
        if val.dtype != torch.int64:
            continue

        flat_shard_size = val.shape[0]
        if const % flat_shard_size != 0:
            continue

        # Verify result feeds into scatter_add (index_select backward)
        if not any(
            u.op == "call_function"
            and u.target == torch.ops.aten.scatter_add.default
            and len(u.args) >= 2
            and u.args[1] == 0
            for u in node.users
        ):
            continue

        results.append(FlatOffsetAdd(node=node, flat_shard_size=flat_shard_size))

    return results


@dataclass
class BackwardSlice:
    """Store a backward vocabulary-gradient slice match.

    Args:
        node (Node): ``slice.Tensor`` node extracting the shard.
        start (int): Inclusive shard start index.
        shard_size (int): Number of vocabulary rows in the shard.

    Note:
        DTensor extracts a row-wise embedding shard with
        ``start = shard_index * shard_size``.
    """

    node: Node
    start: int
    shard_size: int  # end - start


def _is_reshaped_graph_input(node: Node) -> bool:
    """Whether `node` is a graph input, or a reshape of one.

    The gradient this matcher slices arrives as a graph input, so a value computed in the graph
    is not it. Requiring the element count to survive each hop excludes slicing, so a value
    carved out of an input does not pass as the input itself.
    """
    while node.op == "call_function":
        sources = [arg for arg in node.args if isinstance(arg, Node)]
        if len(sources) != 1:
            return False
        here, there = node.meta.get("val"), sources[0].meta.get("val")
        if not isinstance(here, torch.Tensor) or not isinstance(there, torch.Tensor):
            return False
        if here.numel() != there.numel():
            return False
        node = sources[0]
    return node.op == "placeholder"


def find_backward_slices(graph: Graph) -> list[BackwardSlice]:
    """Find shard-like dimension-zero slices of two-dimensional graph inputs.

    Args:
        graph (Graph): FX graph to search.

    Returns:
        list[BackwardSlice]: Structurally matched slice candidates in graph order.

    Note:
        A match has aligned nonnegative integer bounds. Its shard size is at least two and
        evenly divides a first dimension larger than one shard. The input is a placeholder
        or comes through single-input operations that preserve the element count.
    """
    results = []
    for node in graph.nodes:
        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.slice.Tensor
            and len(node.args) >= 4
        ):
            continue

        dim = node.args[1]
        start = node.args[2]
        end = node.args[3]

        if dim != 0 or not isinstance(start, int) or not isinstance(end, int):
            continue

        shard_size = end - start
        if shard_size < 2 or start < 0:
            continue
        if start % shard_size != 0:
            continue

        input_node = node.args[0]
        if not hasattr(input_node, "meta") or "val" not in input_node.meta:
            continue
        val = input_node.meta["val"]
        if val.dim() != 2:
            continue
        vocab_size = val.shape[0]
        if vocab_size % shard_size != 0 or vocab_size <= shard_size:
            continue

        if not _is_reshaped_graph_input(input_node):
            continue

        results.append(BackwardSlice(node=node, start=start, shard_size=shard_size))

    return results


def find_vocab_triads(graph: Graph) -> list[VocabTriad]:
    """Find connected vocabulary-sharding comparison triads.

    Args:
        graph (Graph): FX graph to search.

    Returns:
        list[VocabTriad]: Matches in topological order.

    Note:
        A direct ``bitwise_or(lt.Scalar, ge.Scalar)`` must compare one input against
        numeric bounds. Bounds are converted to integers before requiring a shard size
        of at least two and an aligned lower bound. The subtraction is searched only
        among that input's direct users and depth-one clones.
    """
    triads = []
    claimed_sub_nodes: set[Node] = set()
    for node in graph.nodes:
        if not (node.op == "call_function" and node.target == torch.ops.aten.bitwise_or.Tensor):
            continue

        lt_node, ge_node = node.args[0], node.args[1]

        # Verify lt.Scalar and ge.Scalar with integer constants
        if not (
            lt_node.op == "call_function"
            and lt_node.target == torch.ops.aten.lt.Scalar
            and ge_node.op == "call_function"
            and ge_node.target == torch.ops.aten.ge.Scalar
        ):
            continue

        lower = lt_node.args[1]
        upper = ge_node.args[1]
        if not (isinstance(lower, int | float) and isinstance(upper, int | float)):
            continue
        lower, upper = int(lower), int(upper)

        shard_size = upper - lower
        # Minimum shard_size of 2 avoids matching trivial patterns where
        # shard_size=1 makes lower % shard_size == 0 for any lower.
        if shard_size < 2 or lower % shard_size != 0:
            continue

        # Verify lt and ge operate on the same input
        if lt_node.args[0] != ge_node.args[0]:
            continue

        # Find the associated sub node by walking from the shared input
        shared_input = lt_node.args[0]
        sub_node = _find_sub_from_input(shared_input, lower, claimed_sub_nodes)
        if sub_node is not None:
            claimed_sub_nodes.add(sub_node)

        triads.append(
            VocabTriad(
                lt_node=lt_node,
                ge_node=ge_node,
                bitwise_or_node=node,
                sub_node=sub_node,
                lower_bound=lower,
                upper_bound=upper,
            )
        )

    return triads


def _find_sub_from_input(
    shared_input: Node, offset_val: int, exclude: set[Node] | None = None
) -> Node | None:
    """Find sub.Tensor(x, offset_val) where x is the shared_input or a clone of it.

    Walks users of shared_input looking for clone -> sub.Tensor(clone, offset_val)
    or direct sub.Tensor(shared_input, offset_val).

    Note: Only checks depth-1 clones (clone.default directly from shared_input).
    The DTensor embedding pattern always uses exactly one clone between the shared
    input and the sub node.

    Args:
        shared_input: The shared input node (e.g., tokens).
        offset_val: The integer offset to match in sub.Tensor(x, offset_val).
        exclude: Set of sub nodes already claimed by other triads.
    """
    candidates = [shared_input]
    # Also check clones of the shared input
    for user in shared_input.users:
        if user.op == "call_function" and user.target == torch.ops.aten.clone.default:
            candidates.append(user)

    for candidate in candidates:
        for user in candidate.users:
            if (
                user.op == "call_function"
                and user.target == torch.ops.aten.sub.Tensor
                and len(user.args) >= 2
                and isinstance(user.args[1], int | float)
                and int(user.args[1]) == offset_val
                and (exclude is None or user not in exclude)
            ):
                return user
    return None


def _infer_shard_index(graph: fx.Graph) -> int | None:
    """Infer the TP shard index from vocab sharding pattern constants.

    Checks all patterns (triads and flat-offset adds) and verifies they agree
    on the shard index. Returns None if patterns disagree or none are found.
    """
    indices: set[int] = set()

    for triad in find_vocab_triads(graph):
        indices.add(triad.lower_bound // triad.shard_size)

    for fa in find_flat_offset_adds(graph):
        const = fa.node.args[1]
        indices.add(const // fa.flat_shard_size)

    for bs in find_backward_slices(graph):
        indices.add(bs.start // bs.shard_size)

    if len(indices) == 1:
        return indices.pop()

    if len(indices) > 1:
        logger.warning(
            f"ShardIndexInjection: conflicting shard indices {indices}, skipping injection"
        )
        return None

    return None  # No patterns found — nothing to rewrite


class ShardIndexInjection(NeuronFXPass):
    """Inject a vocabulary-shard-index placeholder after existing graph inputs.

    Args:
        state (NeuronFXPassState | None, optional): Shared state used to record the
            added runtime input. Defaults to ``None``.

    Note:
        The pass is idempotent and skips when
        ``TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE=1`` or the graph is empty,
        unmatched, already injected, or has conflicting inferred shard indices.
    """

    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        """Inject a runtime vocabulary-shard-index graph input.

        Args:
            graph_module (fx.GraphModule): Graph module to inspect and potentially modify.

        Returns:
            PassResult: The graph module and whether the placeholder was added.
        """
        graph = graph_module.graph

        # Escape hatch
        if os.environ.get("TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE") == "1":
            return PassResult(graph_module, False)

        placeholders = [n for n in graph.nodes if n.op == "placeholder"]

        # Guard: no placeholders
        if not placeholders:
            return PassResult(graph_module, False)

        # Guard: already injected (idempotent)
        if any(n.name == "vocab_shard_index" for n in placeholders):
            return PassResult(graph_module, False)

        original_input_count = len(placeholders)

        # Guard: conflicting shard indices
        shard_index = _infer_shard_index(graph)
        if shard_index is None:
            return PassResult(graph_module, False)

        last_placeholder = placeholders[-1]
        with graph.inserting_after(last_placeholder):
            shard_node = graph.placeholder("vocab_shard_index")
            with FakeTensorMode():
                shard_node.meta["val"] = torch.empty((), dtype=torch.int32)

        graph.lint()
        graph_module.recompile()

        if self.state is not None:
            self.state.add_pass_mutations(
                pass_name="ShardIndexInjection",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[original_input_count],
                        transform_key="shard_index",
                        transform_params={"shard_index": shard_index},
                        pass_name="ShardIndexInjection",
                    )
                ],
            )

        logger.debug(f"ShardIndexInjection: shard_index={shard_index}")
        return PassResult(graph_module, True)


class VocabShardingSPMD(NeuronFXPass):
    """Rewrite rank-specific vocabulary constants as shard-index expressions.

    Args:
        state (NeuronFXPassState, optional): Shared FX-pass state used for artifact
            dumping. Defaults to ``None``.

    Note:
        For forward triads, lower and subtraction offsets become ``index * shard_size``
        and upper bounds add ``shard_size``. Backward flat offsets become
        ``index * flat_shard_size``. Backward slice bounds use an int64-cast index times
        ``shard_size`` and add ``shard_size`` for the end. ``index`` is the placeholder
        injected by ``ShardIndexInjection``.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Rewrite vocabulary-sharding patterns in an FX graph.

        Args:
            graph_module (torch.fx.GraphModule): Graph module containing the injected
                shard-index placeholder.

        Returns:
            PassResult: The graph module and whether any pattern was rewritten.
        """
        graph = graph_module.graph

        # Find the vocab_shard_index placeholder
        shard_node = None
        for node in graph.nodes:
            if node.op == "placeholder" and node.name == "vocab_shard_index":
                shard_node = node
                break
        if shard_node is None:
            return PassResult(graph_module, False)

        changed = False

        # Rewrite forward triads
        triads = find_vocab_triads(graph)
        for triad in triads:
            self._rewrite_triad(graph, triad, shard_node)
            changed = True

        # Rewrite backward flat-offset adds
        flat_adds = find_flat_offset_adds(graph)
        for fa in flat_adds:
            self._rewrite_flat_offset(graph, fa, shard_node)
            changed = True

        # Rewrite backward slices
        bslices = find_backward_slices(graph)
        for bs in bslices:
            self._rewrite_backward_slice(graph, bs, shard_node)
            changed = True

        if changed:
            graph.lint()
            graph_module.recompile()
            logger.debug(
                f"VocabShardingSPMD: rewrote {len(triads)} triads, "
                f"{len(flat_adds)} flat-offset adds, {len(bslices)} backward slices"
            )
        return PassResult(graph_module, changed)

    def _insert_offset(self, graph: Graph, before: Node, shard_node: Node, shard_size: int) -> Node:
        """Insert mul(vocab_shard_index, shard_size) before `before`."""
        with graph.inserting_before(before):
            offset = graph.call_function(torch.ops.aten.mul.Tensor, (shard_node, shard_size))
            offset.meta["val"] = _fake_i64_scalar()
        return offset

    def _rewrite_triad(self, graph: Graph, triad: VocabTriad, shard_node: Node):
        """Rewrite a single triad's constants to vocab-shard-index expressions."""
        shard_size = triad.shard_size

        offset = self._insert_offset(graph, triad.lt_node, shard_node, shard_size)

        # Rewrite lt: lt.Scalar(tokens, lower) -> lt.Tensor(tokens, offset)
        triad.lt_node.args = (triad.lt_node.args[0], offset)
        triad.lt_node.target = torch.ops.aten.lt.Tensor

        # Rewrite ge: ge.Scalar(tokens, upper) -> ge.Tensor(tokens, upper_expr)
        with graph.inserting_before(triad.ge_node):
            upper_expr = graph.call_function(torch.ops.aten.add.Tensor, (offset, shard_size))
            upper_expr.meta["val"] = _fake_i64_scalar()
        triad.ge_node.args = (triad.ge_node.args[0], upper_expr)
        triad.ge_node.target = torch.ops.aten.ge.Tensor

        # Rewrite sub: sub.Tensor(clone, lower) -> sub.Tensor(clone, offset)
        if triad.sub_node is not None:
            sub_offset = self._insert_offset(graph, triad.sub_node, shard_node, shard_size)
            triad.sub_node.args = (triad.sub_node.args[0], sub_offset)

    def _rewrite_flat_offset(self, graph: Graph, fa: FlatOffsetAdd, shard_node: Node):
        """Rewrite add.Tensor(x, C) -> add.Tensor(x, mul(vocab_shard_index, flat_shard_size))."""
        flat_shard = self._insert_offset(graph, fa.node, shard_node, fa.flat_shard_size)
        fa.node.args = (fa.node.args[0], flat_shard)

    def _rewrite_backward_slice(self, graph: Graph, bs: BackwardSlice, shard_node: Node):
        """Rewrite slice start/end to vocab_shard_index expressions.

        Cast to i64 because slice indices must match arith ops in torch-mlir lowering.
        Uses explicit i64 metadata (not _fake_i64_scalar which returns i32) because
        torch-mlir reads FakeTensor dtype to generate arith ops — i32 metadata with
        i64 computation causes 'arith.maxsi requires same type' errors.
        """
        i64_meta = FakeTensorMode().from_tensor(torch.tensor(0, dtype=torch.int64))
        with graph.inserting_before(bs.node):
            shard_i64 = graph.call_function(
                torch.ops.aten._to_copy.default, (shard_node,), {"dtype": torch.int64}
            )
            shard_i64.meta["val"] = i64_meta
            offset = graph.call_function(torch.ops.aten.mul.Tensor, (shard_i64, bs.shard_size))
            offset.meta["val"] = i64_meta
            end_expr = graph.call_function(torch.ops.aten.add.Tensor, (offset, bs.shard_size))
            end_expr.meta["val"] = i64_meta
        bs.node.args = (bs.node.args[0], 0, offset, end_expr)


__all__ = [
    "BackwardSlice",
    "FlatOffsetAdd",
    "ShardIndexInjection",
    "VocabShardingSPMD",
    "VocabTriad",
    "find_backward_slices",
    "find_flat_offset_adds",
    "find_vocab_triads",
]
