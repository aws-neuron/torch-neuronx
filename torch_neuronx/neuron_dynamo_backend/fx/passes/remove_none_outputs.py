"""
FX pass to remove None outputs from backward graphs and preserve metadata for restoration.

This pass handles the mismatch between AOTAutograd's efficient None returns for inputs
that don't require gradients and the .backward() API expectation of explicit gradient tensors.
"""

import logging
from dataclasses import dataclass

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger = logging.getLogger(__name__)


@dataclass
class NoneOutputInfo:
    """Metadata class to track None output positions for restoration."""

    non_none_positions: list[int]
    original_output_count: int
    new_output_count: int


class RemoveNoneOutputs(PassBase):
    def __init__(self):
        super().__init__()
        self.result: NoneOutputInfo | None = None

    def _normalize_outputs(self, output_node: torch.fx.Node) -> list[torch.fx.Node | None]:
        """Convert output node arguments to a normalized list.

        Handles both single outputs and tuple/list outputs by converting
        them to a consistent list format.

        Args:
            output_node (torch.fx.Node): The graph's output node.

        Returns:
            list[torch.fx.Node | None]: List of output nodes (may contain None).
        """
        outputs = output_node.args[0]
        if not isinstance(outputs, (list | tuple)):
            return [outputs]
        return list(outputs)

    def call(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Execute the pass to remove None outputs from the graph.

        Filters out None values from the graph's output tuple and records
        metadata about the original positions for later restoration.

        Args:
            gm (torch.fx.GraphModule): Graph module to transform.

        Returns:
            PassResult: Result containing modified graph and modified=True.
        """
        logger.debug("Starting remove_none_outputs_pass")
        output_node = gm.graph.output_node()
        outputs = self._normalize_outputs(output_node)
        non_none_positions = []
        non_none_outputs = []
        for i, output in enumerate(outputs):
            if output is not None:
                non_none_positions.append(i)
                non_none_outputs.append(output)
        self.result = NoneOutputInfo(non_none_positions, len(outputs), len(non_none_outputs))
        logger.debug(f"Finished remove_non_outputs_pass: updated graph signature: {self.result}")
        output_node.args = (tuple(non_none_outputs),)
        gm.recompile()
        return PassResult(gm, modified=True)


__all__ = ["NoneOutputInfo", "RemoveNoneOutputs"]
