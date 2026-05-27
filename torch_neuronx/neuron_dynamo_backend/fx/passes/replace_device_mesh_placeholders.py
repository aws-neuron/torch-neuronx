"""
FX pass to replace DeviceMesh placeholder inputs with None.

Placeholders whose meta["val"] is a FakeScriptObject wrapping a DeviceMesh
are replaced with None and removed from the graph.
"""

import logging

import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)

logger = logging.getLogger(__name__)


class ReplaceDeviceMeshPlaceholders(NeuronFXPass):
    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)
        self.removed_indices: list[int] | None = None

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        removed = []
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for i, node in enumerate(placeholders):
            fso = node.meta.get("val")
            if isinstance(fso, FakeScriptObject) and isinstance(fso.real_obj, DeviceMesh):
                logger.debug(f"Found DeviceMesh input node: {node} at placeholder index {i}")
                node.replace_all_uses_with(None)
                gm.graph.erase_node(node)
                removed.append(i)
        self.removed_indices = removed
        if removed:
            gm.recompile()
        return PassResult(gm, bool(removed))


__all__ = ["ReplaceDeviceMeshPlaceholders"]
