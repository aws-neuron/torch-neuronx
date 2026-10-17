import logging
from collections.abc import Callable
from typing import Any

import torch

from torch_neuronx.neuron_dynamo_backend._nki.ast import (
    NISAOpValue,
    NKIContext,
)
from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import lowerings

TargetType = Callable[..., Any] | str
logger = logging.getLogger(__name__)


# FX -> nki.isa lowering for point-wise operations
class NISAPointwiseSubgraphLowering(torch.fx.Interpreter):
    """Interpret an FX graph as NISA pointwise operations.

    Args:
        gm (torch.fx.GraphModule): FX graph module to interpret.
        shape (tuple, optional): Retained lowering shape. Defaults to ``(None, None)``.
    """

    ctx: NKIContext
    graph_output: NISAOpValue | None

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        shape=(None, None),
    ) -> None:
        super().__init__(gm, garbage_collect_values=True)
        self.graph_outputs = None
        self.shape = shape
        self.ctx = NKIContext()

    # TODO: currently, all intermediate buffers are allocated.
    # Should we garbage collect or leave NKI compiler to do it?
    def call_function(
        self,
        target: TargetType,
        args: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Lower an FX function call through the registered NKI lowering.

        Args:
            target (TargetType): FX call target registered in the lowering table.
            args (Any): Positional arguments for the lowering.
            kwargs (dict[str, Any]): Keyword arguments for the lowering.

        Returns:
            Any: Result returned by the registered lowering.

        Raises:
            AssertionError: If the target has no registered lowering.
        """
        assert target in lowerings, f"{target} lowering is not defined yet!"
        return lowerings[target](self.ctx, *args, **kwargs)

    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        """Record the single output produced by the interpreted FX graph.

        Args:
            target (str): Unused FX output target.
            args (tuple[Any]): One-item tuple containing the graph output.
            kwargs (dict[str, Any]): Unused FX output keyword arguments.

        Raises:
            AssertionError: If ``args`` does not contain exactly one item.

        Note:
            When the output is a list, only its first item is retained.
        """
        assert len(args) == 1
        self.graph_outputs = args[0]

        # TODO: joint_graph may return multiple extra outputs as a list with other buffer gradients.
        # We need to handle them to support score_mod with free variables.
        if isinstance(args[0], list):
            self.graph_outputs = args[0][0]
