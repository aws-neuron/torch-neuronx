import logging
from typing import NamedTuple

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)
from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

logger = logging.getLogger(__name__)

# Operations that MUTATE data in-place (scatter, copy, etc.)
MUTATION_OPS = {
    # Scatter operations - write data at specified indices
    torch.ops.aten.scatter.src,
    torch.ops.aten.scatter.value,
    torch.ops.aten.scatter_add.default,
    torch.ops.aten.scatter_reduce.two,
    torch.ops.aten.select_scatter.default,
    torch.ops.aten.slice_scatter.default,
    torch.ops.aten.index_put.default,
    torch.ops.aten.index_copy.default,
    # Copy (functional version of copy_)
    torch.ops.aten.copy.default,
}

# Operations that we trace through to find mutation sources.
# Includes views (share storage) and copies (new storage).
# These don't mutate, but connect outputs back to inputs.
TRACEABLE_OPS = {
    # True view operations (share underlying storage)
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    # Copy operations (new storage, but we trace through to find source)
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
}

# All ops we trace through to find mutation lineage
TRACEABLE_MUTATION_OPS = MUTATION_OPS | TRACEABLE_OPS


class _CopyInplaceMutation(NamedTuple):
    """One ``copy_`` rewrite: the mutated input and where its result already lands.

    ``returned_output_idx`` is a slot in the pre-rewrite output tuple this mutation
    may take, or None when it needs a new one, which also covers a returned result
    whose slot an earlier mutation already claimed.
    """

    input_idx: int
    param_idx: list[int] | None
    src: Node
    returned_output_idx: int | None


class FunctionalizeCopyInplacePass(NeuronFXPass):
    """Functionalize input ``copy_`` operations and record output aliases.

    Args:
        state (NeuronFXPassState, optional): Shared FX-pass state used to record output
            mutations. Defaults to ``None``.

    Note:
        Removes ``copy_`` nodes whose destination is an input, prepends unreturned mutation
        results, and records pass-through and mutation outputs for buffer reuse.
    """

    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)
        self.mutation_info: AliasingInfo = AliasingInfo()
        self.passthrough_info: AliasingInfo = AliasingInfo()
        self.added_mutation_output_indices: list[int] = []

    def call(self, gm: GraphModule) -> PassResult:
        """Functionalize input copies and collect graph output relationships.

        Args:
            gm (GraphModule): Graph module to inspect and potentially modify.

        Returns:
            PassResult: The graph module and whether any ``copy_`` node was removed.
        """
        self.mutation_info = AliasingInfo()
        self.passthrough_info = AliasingInfo()
        self.added_mutation_output_indices = []
        graph = gm.graph

        placeholders = [n for n in graph.nodes if n.op == "placeholder"]

        output_node = self._get_output_node(graph)
        if output_node is None:
            return PassResult(gm, modified=False)

        # Handle explicit copy_ operations (graph modification)
        modified = self._handle_copy_inplace(graph, placeholders, output_node)

        # Detect pass-through outputs (inputs returned unchanged)
        self._detect_passthrough_outputs(placeholders, output_node)

        if modified:
            graph.lint()
            gm.recompile()

        # Collect all mutations to emit atomically as one PassMutations entry
        out_muts = []
        if self.added_mutation_output_indices:
            out_muts.append(
                IOMutation(
                    kind=MutationKind.ADD,
                    indices=self.added_mutation_output_indices,
                    pass_name="FunctionalizeCopyInplacePass",
                )
            )

        # Emit passthrough ALIAS mutations for outputs that are direct
        # pass-throughs of input placeholders (buffer reuse, no copy-back).
        if self.passthrough_info and self.passthrough_info.aliases:
            out_muts.append(
                IOMutation(
                    kind=MutationKind.ALIAS,
                    indices=[a.output_index for a in self.passthrough_info.aliases],
                    alias_input_indices=[a.parameter_number for a in self.passthrough_info.aliases],
                    is_mutation=False,
                    pass_name="FunctionalizeCopyInplacePass",
                )
            )

        # Emit mutation ALIAS for outputs that are mutation results of inputs
        # (buffer reuse + copy-back needed).
        if self.mutation_info and self.mutation_info.aliases:
            out_muts.append(
                IOMutation(
                    kind=MutationKind.ALIAS,
                    indices=[a.output_index for a in self.mutation_info.aliases],
                    alias_input_indices=[a.parameter_number for a in self.mutation_info.aliases],
                    is_mutation=True,
                    pass_name="FunctionalizeCopyInplacePass",
                )
            )

        if self.state is not None:
            pre_functionalize_pass_mutations = self.state.pass_mutations[:]  # noqa: F841
            if out_muts:
                self.state.add_pass_mutations(
                    pass_name="FunctionalizeCopyInplacePass",
                    output_mutations=out_muts,
                )

            # TODO: Rebase and insert pre-AOT AliasingAnalysis mutations once
            # the executor consumes them. Currently main ignores pre-AOT info
            # at execution time, so inserting them here has no effect.
            # if self.state.pre_aot_pass_mutations is not None:
            #     rebased = rebase_pre_aot_pass_mutations(
            #         self.state.pre_aot_pass_mutations, pre_functionalize_pass_mutations
            #     )
            #     if rebased is not None:
            #         self.state.pass_mutations.append(rebased)

        return PassResult(gm, modified=modified)

    def _get_output_node(self, graph: torch.fx.Graph) -> Node | None:
        """Find the output node in the graph.

        Args:
            graph (torch.fx.Graph): FX graph to search.

        Returns:
            Node | None: The output node, or None if not found.
        """
        return next((node for node in graph.nodes if node.op == "output"), None)

    def _normalize_outputs(self, output_node: Node) -> list[Node]:
        """Convert output node arguments to a normalized list.

        Args:
            output_node (Node): The graph's output node.

        Returns:
            list[Node]: List of output nodes.
        """
        outputs = output_node.args[0]
        if outputs is None:
            return []
        elif not isinstance(outputs, (list | tuple)):
            return [outputs]
        return list(outputs)

    def _is_copy_inplace(self, node: Node) -> bool:
        """Check if a node is a copy_ (in-place copy) operation.

        Args:
            node (Node): FX node to check.

        Returns:
            bool: True if node is aten.copy_ or similar in-place copy.
        """
        if node.op != "call_function":
            return False
        target = node.target
        if target == torch.ops.aten.copy_.default:
            return True
        if hasattr(target, "__name__") and "copy_" in str(target.__name__):
            return True
        return hasattr(target, "_name") and "copy_" in str(target._name)

    def _find_returned_mutation_output(
        self,
        outputs: list[Node],
        node: Node,
        src: Node,
        claimed_output_indices: set[int],
    ) -> int | None:
        """Find the output slot that already carries a copy_ node's mutation result."""
        src_value = src
        if src.op == "call_function" and src.target is torch.ops.aten.copy.default:
            copied_from = src.args[1] if len(src.args) > 1 else None
            if isinstance(copied_from, Node):
                dst_val = src.meta.get("val")
                src_val = copied_from.meta.get("val")
                if (
                    dst_val is not None
                    and src_val is not None
                    and dst_val.dtype == src_val.dtype
                    # Concrete dims only: comparing a symbolic one would plant a shape
                    # guard or raise on an unbacked symint.
                    and all(isinstance(dim, int) for dim in (*dst_val.shape, *src_val.shape))
                    and dst_val.shape == src_val.shape
                ):
                    src_value = copied_from

        return next(
            (
                i
                for i, output in enumerate(outputs)
                if (output is node or output is src or output is src_value)
                and i not in claimed_output_indices
            ),
            None,
        )

    def _handle_copy_inplace(
        self, graph: torch.fx.Graph, placeholders: list[Node], output_node: Node
    ) -> bool:
        """
        Handle explicit copy_ operations on inputs.

        Transforms: input.copy_(value) → removes copy_, and adds value to the
        outputs unless the graph already returns it. Records the mutation
        relationship for compiler optimization.

        Returns:
            True if graph was modified, False otherwise.
        """
        original_outputs = self._normalize_outputs(output_node)
        mutations: list[_CopyInplaceMutation] = []
        nodes_to_remove: list[Node] = []
        claimed_output_indices: set[int] = set()

        for node in list(graph.nodes):
            if self._is_copy_inplace(node):
                dst = node.args[0]
                src = node.args[1]

                if dst in placeholders:
                    input_idx = placeholders.index(dst)
                    returned_output_idx = self._find_returned_mutation_output(
                        original_outputs, node, src, claimed_output_indices
                    )
                    if returned_output_idx is not None:
                        claimed_output_indices.add(returned_output_idx)
                    mutations.append(
                        _CopyInplaceMutation(input_idx, None, src, returned_output_idx)
                    )
                    node.replace_all_uses_with(src)
                    nodes_to_remove.append(node)

        if not mutations:
            return False

        # Remove copy_ nodes
        for node in nodes_to_remove:
            graph.erase_node(node)

        outputs = self._normalize_outputs(output_node)
        hidden = [m for m in mutations if m.returned_output_idx is None]
        mutation_outputs = [m.src for m in hidden]
        self.added_mutation_output_indices = list(range(len(mutation_outputs)))
        for output_idx, mutation in enumerate(hidden):
            self.mutation_info.add(
                parameter_number=mutation.input_idx,
                parameter_index=mutation.param_idx,
                output_index=output_idx,
            )

        # An already-returned result keeps its slot, shifted by the prepended ones.
        for mutation in mutations:
            if mutation.returned_output_idx is not None:
                self.mutation_info.add(
                    parameter_number=mutation.input_idx,
                    parameter_index=mutation.param_idx,
                    output_index=len(mutation_outputs) + mutation.returned_output_idx,
                )

        final_outputs = mutation_outputs + outputs
        output_node.args = (tuple(final_outputs),)
        logger.debug(
            f"Prepended {len(mutation_outputs)} of {len(mutations)} mutations to outputs, "
            f"increased from {len(outputs)}->{len(final_outputs)}: {output_node.args}"
        )

        return True

    def _detect_passthrough_outputs(self, placeholders: list[Node], output_node: Node) -> None:
        """Detect outputs that are direct pass-throughs of input placeholders.

        Scans the output node's arguments and identifies any that are
        placeholder nodes (inputs returned unchanged). Records these as
        aliasing relationships in passthrough_info so the executor can reuse
        input buffers and the compiler can set input_output_alias.

        This is critical for activation checkpointing with torch.compile,
        where AOT Autograd includes input primals (weights) in the forward
        graph's return so backward can recompute the forward.

        Args:
            placeholders: List of input placeholder nodes.
            output_node: The graph's output node.
        """
        outputs = output_node.args[0]
        if outputs is None or not isinstance(outputs, list | tuple):
            return

        # Track which inputs are already aliased by mutation detection
        already_aliased = {alias.output_index for alias in self.mutation_info.aliases}

        for output_idx, output_arg in enumerate(outputs):
            if output_idx in already_aliased:
                continue
            if not isinstance(output_arg, Node):
                continue
            if output_arg.op == "placeholder" and output_arg in placeholders:
                input_idx = placeholders.index(output_arg)
                self.passthrough_info.add(
                    parameter_number=input_idx,
                    parameter_index=[],
                    output_index=output_idx,
                )

    def _trace_to_input(self, node: Node, placeholders: list[Node]) -> tuple[int, bool] | None:
        """Trace a node backward to find its source input placeholder.

        Follows traceable operations (views, clones, mutations) until
        reaching a placeholder or an untraceable node.

        Args:
            node (Node): Starting node to trace from.
            placeholders (list[Node]): List of input placeholder nodes.

        Returns:
            tuple[int, bool] | None: Tuple of (placeholder_index, went_through_mutation)
                if trace reaches a placeholder, None otherwise.
        """
        visited: set[Node] = set()
        current = node
        went_through_mutation = False

        while current not in visited:
            visited.add(current)

            # Reached an input placeholder
            if current in placeholders:
                return (placeholders.index(current), went_through_mutation)

            # Only trace through call_function nodes
            if current.op != "call_function":
                break

            # Track if we pass through a mutation operation
            if current.target in MUTATION_OPS:
                went_through_mutation = True

            # Continue tracing if this is a traceable operation
            if (
                current.target in TRACEABLE_MUTATION_OPS
                and len(current.args) > 0
                and isinstance(current.args[0], Node)
            ):
                current = current.args[0]
                continue

            break

        return None

    def get_mutation_info(self) -> AliasingInfo:
        """Return detected input-mutation output relationships.

        Returns:
            AliasingInfo: Mutation aliases collected by the latest pass invocation.
        """
        return self.mutation_info

    def get_passthrough_info(self) -> AliasingInfo:
        """Return detected input pass-through output relationships.

        Returns:
            AliasingInfo: Pass-through aliases collected by the latest pass invocation.
        """
        return self.passthrough_info
