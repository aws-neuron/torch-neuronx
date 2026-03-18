from collections.abc import Callable
from typing import Any

import torch

from torch_neuronx.neuron_dynamo_backend._nki.ast import (
    NISAOpValue,
    NKIContext,
)
from torch_neuronx.neuron_dynamo_backend._nki.fx_to_nki import lowerings

TargetType = Callable[..., Any] | str


# FX -> nki.isa lowering for point-wise operations
class NISAPointwiseSubgraphLowering(torch.fx.Interpreter):
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
        assert target in lowerings, f"{target} lowering is not defined yet!"
        return lowerings[target](self.ctx, *args, **kwargs)

    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        assert len(args) == 1
        self.graph_outputs = args[0]
