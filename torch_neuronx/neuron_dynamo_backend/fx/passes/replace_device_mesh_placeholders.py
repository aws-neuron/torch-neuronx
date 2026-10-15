"""
FX pass to replace DeviceMesh placeholder inputs with None and remove
DeviceMesh outputs, restoring them at runtime from the original inputs.

Placeholders whose meta["val"] is a FakeScriptObject wrapping a DeviceMesh
are replaced with None and removed from the graph.  Any output that is a
pass-through of one of those removed inputs is also removed from the output
tuple; a single ADD output mutation covers all removed positions, mapping each
output index back to its original input index via ``transform_params``.
"""

import logging

import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.passes.infra.pass_base import PassResult

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, MutationKind
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)

logger = logging.getLogger(__name__)


def _is_device_mesh_fso(val) -> bool:
    return isinstance(val, FakeScriptObject) and isinstance(val.real_obj, DeviceMesh)


class ReplaceDeviceMeshPlaceholders(NeuronFXPass):
    def __init__(self, state: NeuronFXPassState | None = None):
        super().__init__(state)

    def call(self, gm: torch.fx.GraphModule) -> PassResult:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

        # Map from DeviceMesh placeholder node -> original input index.
        mesh_node_to_input_idx: dict[torch.fx.Node, int] = {
            node: i
            for i, node in enumerate(placeholders)
            if _is_device_mesh_fso(node.meta.get("val"))
        }

        removed_input_indices: list[int] = sorted(mesh_node_to_input_idx.values())
        output_index_to_input_index: dict[int, int] = {}

        # Scan outputs for DeviceMesh values before erasing placeholder nodes.
        output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
        if output_node is not None:
            raw = output_node.args[0]
            outputs: list = list(raw) if isinstance(raw, list | tuple) else [raw]
            for out_idx, out_node in enumerate(outputs):
                if not isinstance(out_node, torch.fx.Node):
                    continue
                if not _is_device_mesh_fso(out_node.meta.get("val")):
                    continue
                if out_node not in mesh_node_to_input_idx:
                    raise RuntimeError(
                        f"Output {out_node.name} is a DeviceMesh generated from an "
                        "intermediate computation. This cannot be executed in-graph "
                        "on device."
                    )
                output_index_to_input_index[out_idx] = mesh_node_to_input_idx[out_node]

            if output_index_to_input_index:
                kept = [o for i, o in enumerate(outputs) if i not in output_index_to_input_index]
                output_node.args = (tuple(kept),)

        # Remove DeviceMesh placeholder nodes.
        for node in mesh_node_to_input_idx:
            logger.debug(
                f"Found DeviceMesh input node: {node} at placeholder index "
                f"{mesh_node_to_input_idx[node]}"
            )
            node.replace_all_uses_with(None)
            gm.graph.erase_node(node)

        modified = removed_input_indices or output_index_to_input_index
        if not modified:
            return PassResult(gm, False)

        gm.recompile()

        if self.state is not None:
            in_muts = []
            out_muts = []
            if removed_input_indices:
                in_muts.append(
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=removed_input_indices,
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                )
            if output_index_to_input_index:
                out_muts.append(
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=list(output_index_to_input_index.keys()),
                        transform_key="input_passthrough",
                        transform_params={"index_to_input": output_index_to_input_index},
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                )
            self.state.add_pass_mutations(
                pass_name="ReplaceDeviceMeshPlaceholders",
                input_mutations=in_muts,
                output_mutations=out_muts,
            )

        return PassResult(gm, True)


__all__ = ["ReplaceDeviceMeshPlaceholders"]
