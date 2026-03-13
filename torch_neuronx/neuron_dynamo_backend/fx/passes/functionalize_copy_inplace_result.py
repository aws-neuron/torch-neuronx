import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from torch_neuronx.neuron_dynamo_backend.utils.alias_info import AliasingInfo

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


class FunctionalizeCopyInplacePass(PassBase):
    """
    Pass that detects and records input-output mutation relationships.

    Handles copy_ pattern: Removes copy_ ops, prepends mutation value to outputs

    Records mutation info mapping outputs to their source inputs,
    enabling compiler buffer optimizations (e.g., buffer reuse).

    Note: "Mutation lineage" is tracked, not strict aliasing. A clone breaks
    aliasing but we still track that the output derives from mutating the input.
    """

    def __init__(self):
        super().__init__()
        self.mutation_info: AliasingInfo = AliasingInfo()

    def call(self, gm: GraphModule) -> PassResult:
        self.mutation_info = AliasingInfo()
        graph = gm.graph

        placeholders = [n for n in graph.nodes if n.op == "placeholder"]

        output_node = self._get_output_node(graph)
        if output_node is None:
            return PassResult(gm, modified=False)

        # Handle explicit copy_ operations (graph modification)
        modified = self._handle_copy_inplace(graph, placeholders, output_node)

        if modified:
            graph.lint()
            gm.recompile()

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

    def _handle_copy_inplace(
        self, graph: torch.fx.Graph, placeholders: list[Node], output_node: Node
    ) -> bool:
        """
        Handle explicit copy_ operations on inputs.

        Transforms: input.copy_(value) → removes copy_, adds value to outputs
        Records the mutation relationship for compiler optimization.

        Returns:
            True if graph was modified, False otherwise.
        """
        mutations: list[tuple[int, list[int] | None, Node]] = []
        nodes_to_remove: list[Node] = []

        for node in list(graph.nodes):
            if self._is_copy_inplace(node):
                dst = node.args[0]
                src = node.args[1]

                if dst in placeholders:
                    input_idx = placeholders.index(dst)
                    mutations.append((input_idx, None, src))
                    node.replace_all_uses_with(src)
                    nodes_to_remove.append(node)

        if not mutations:
            return False

        # Remove copy_ nodes
        for node in nodes_to_remove:
            graph.erase_node(node)

        # Prepend mutation values to outputs
        outputs = self._normalize_outputs(output_node)

        mutation_outputs = []
        for idx, (input_idx, param_idx, src_node) in enumerate(mutations):
            mutation_outputs.append(src_node)
            self.mutation_info.add(
                parameter_number=input_idx,
                parameter_index=param_idx,
                output_index=idx,
            )

        final_outputs = mutation_outputs + outputs
        output_node.args = (tuple(final_outputs),)

        return True

    def _detect_mutation_lineage(
        self, graph: torch.fx.Graph, placeholders: list[Node], outputs: list[Node]
    ) -> bool:
        """
        Detect outputs that derive from mutating an input.

        Traces each output backward through the graph. If the trace passes
        through a mutation operation (scatter, copy, etc.) and reaches an
        input placeholder, records this relationship.

        Note: This detects mutation *lineage*, not strict aliasing. An output
        may be a clone of a mutated input (no aliasing) but we still track
        the relationship for buffer optimization. Only the FIRST output deriving
        from each input is recorded as that is the only kind we care about.

        Returns:
            True if any mutation lineage was detected, False otherwise.
        """
        seen_inputs: set[int] = set()

        for output_idx, output_node in enumerate(outputs):
            if not isinstance(output_node, Node):
                continue

            result = self._trace_to_input(output_node, placeholders)

            if result is not None:
                input_idx, went_through_mutation = result

                # Only record if the path included a mutation operation
                if went_through_mutation and input_idx not in seen_inputs:
                    seen_inputs.add(input_idx)
                    self.mutation_info.add(
                        parameter_number=input_idx,
                        parameter_index=None,
                        output_index=output_idx,
                    )

        return len(self.mutation_info) > 0

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
        """Return the detected mutation information."""
        return self.mutation_info
