"""FX pass to ensure contiguous inputs for operations that require it."""

import torch
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPass


class EnsureContiguousInputs(NeuronFXPass):
    """Insert ``contiguous`` before split operations with noncontiguous inputs.

    Args:
        state (NeuronFXPassState, optional): Shared state used for pass artifact dumping.
            Defaults to ``None``.
    """

    def _is_contiguous(self, node: torch.fx.Node) -> bool:
        """Check if a node produces a contiguous tensor using FakeTensor metadata."""
        # Check if metadata exists and has contiguity info
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "is_contiguous"):
                return val.is_contiguous()

        return node.op == "placeholder"

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Make inputs to supported split operations contiguous.

        Args:
            graph_module (torch.fx.GraphModule): FX graph module to update.

        Returns:
            PassResult: Graph module and whether a ``contiguous`` node was inserted.
        """
        modified = False
        for node in graph_module.graph.nodes:
            # Match chunk and split operations
            if node.op == "call_function" and (
                node.target
                in (
                    torch.ops.aten.chunk.default,
                    torch.chunk,
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes.default,
                    torch.split,
                )
            ):
                input_arg = node.args[0] if node.args else None
                if (
                    input_arg
                    and isinstance(input_arg, torch.fx.Node)
                    and not self._is_contiguous(input_arg)
                ):
                    with graph_module.graph.inserting_before(node):
                        contiguous_node = graph_module.graph.call_function(
                            torch.ops.aten.contiguous.default, (input_arg,)
                        )
                    node.replace_input_with(input_arg, contiguous_node)
                    modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
