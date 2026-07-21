"""
FX pass to remove None outputs from backward graphs and record the mutation.

This pass handles the mismatch between AOTAutograd's efficient None returns for inputs
that don't require gradients and the .backward() API expectation of explicit gradient tensors.
"""

import logging

import torch
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, MutationKind
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)

logger = logging.getLogger(__name__)


class RemoveNoneOutputs(NeuronFXPass):
    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)

    def _normalize_outputs(self, output_node: torch.fx.Node) -> list[torch.fx.Node | None]:
        """Convert output node arguments to a normalized list.

        Handles both single outputs and tuple/list outputs by converting
        them to a consistent list format.
        """
        outputs = output_node.args[0]
        if not isinstance(outputs, (list | tuple)):
            return [outputs]
        return list(outputs)

    def call(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Execute the pass to remove None outputs from the graph.

        Filters out None values from the graph's output tuple and records
        an IOMutation so the executor can restore them at runtime.
        """
        logger.debug("Starting remove_none_outputs_pass")
        output_node = gm.graph.output_node()
        outputs = self._normalize_outputs(output_node)

        none_positions = [i for i, output in enumerate(outputs) if output is None]
        if not none_positions:
            return PassResult(gm, modified=False)

        non_none_outputs = [output for output in outputs if output is not None]
        output_node.args = (tuple(non_none_outputs),)
        gm.recompile()

        if self.state is not None:
            self.state.add_pass_mutations(
                pass_name="RemoveNoneOutputs",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=none_positions,
                        transform_key="none",
                        pass_name="RemoveNoneOutputs",
                    )
                ],
            )

        logger.debug(
            f"Finished remove_none_outputs_pass: removed {len(none_positions)} None outputs"
        )
        return PassResult(gm, modified=True)


__all__ = ["RemoveNoneOutputs"]
