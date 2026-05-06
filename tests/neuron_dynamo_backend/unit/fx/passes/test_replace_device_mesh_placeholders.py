"""
Unit tests for replace_device_mesh_placeholders FX pass.

Tests the FX pass that removes DeviceMesh placeholder inputs from the graph
by replacing their uses with None and erasing the nodes.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.fx as fx
from torch._library.fake_class_registry import FakeScriptObject
from torch.distributed.device_mesh import DeviceMesh

from torch_neuronx.neuron_dynamo_backend.fx.passes.replace_device_mesh_placeholders import (
    ReplaceDeviceMeshPlaceholders,
)


def _make_device_mesh_fso():
    """Create a FakeScriptObject wrapping a DeviceMesh."""
    mesh = MagicMock(spec=DeviceMesh)
    fso = FakeScriptObject.__new__(FakeScriptObject)
    object.__setattr__(fso, "wrapped_obj", mesh)
    object.__setattr__(fso, "script_class_name", "DeviceMesh")
    object.__setattr__(fso, "real_obj", mesh)
    return fso


def _make_non_device_mesh_fso():
    """Create a FakeScriptObject wrapping a non-DeviceMesh object."""
    obj = MagicMock()
    fso = FakeScriptObject.__new__(FakeScriptObject)
    object.__setattr__(fso, "wrapped_obj", obj)
    object.__setattr__(fso, "script_class_name", "SomeOtherClass")
    object.__setattr__(fso, "real_obj", obj)
    return fso


def _build_graph_with_device_mesh_placeholders(
    num_tensor_inputs: int,
    device_mesh_indices: list[int],
):
    """
    Build a GraphModule with placeholder nodes where specified indices
    have DeviceMesh FakeScriptObject metadata.

    Non-DeviceMesh placeholders are tensor inputs that get summed for the output.
    DeviceMesh placeholders are not used by any computation node.
    """
    total = num_tensor_inputs + len(device_mesh_indices)
    graph = fx.Graph()
    placeholders = []
    for i in range(total):
        p = graph.placeholder(f"arg_{i}")
        if i in device_mesh_indices:
            p.meta["val"] = _make_device_mesh_fso()
        else:
            p.meta["val"] = torch.randn(2, 3)
        placeholders.append(p)

    # Build a simple computation using only tensor placeholders
    tensor_placeholders = [p for i, p in enumerate(placeholders) if i not in device_mesh_indices]
    if len(tensor_placeholders) == 0:
        result = None
    elif len(tensor_placeholders) == 1:
        result = tensor_placeholders[0]
    else:
        result = tensor_placeholders[0]
        for tp in tensor_placeholders[1:]:
            result = graph.call_function(torch.ops.aten.add.Tensor, (result, tp))

    graph.output(result)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm


class TestReplaceDeviceMeshPlaceholders:
    """Test class for DeviceMesh placeholder removal FX pass."""

    def test_single_device_mesh_removed(self):
        """Test removal of a single DeviceMesh placeholder."""
        # Graph: [tensor, device_mesh, tensor]
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[1]
        )

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        assert result.modified
        assert pass_instance.removed_indices == [1]

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_multiple_device_meshes_removed(self):
        """Test removal of multiple DeviceMesh placeholders."""
        # Graph: [device_mesh, tensor, device_mesh, tensor, device_mesh]
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[0, 2, 4]
        )

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        assert result.modified
        assert pass_instance.removed_indices == [0, 2, 4]

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_no_device_mesh_no_modification(self):
        """Test that graph without DeviceMesh placeholders is not modified."""
        gm = _build_graph_with_device_mesh_placeholders(num_tensor_inputs=3, device_mesh_indices=[])

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        assert not result.modified
        assert pass_instance.removed_indices == []

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 3

    def test_all_device_meshes_removed(self):
        """Test removal when all placeholders are DeviceMesh."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=0, device_mesh_indices=[0, 1]
        )

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        assert result.modified
        assert pass_instance.removed_indices == [0, 1]

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 0

    def test_non_device_mesh_fake_script_object_not_removed(self):
        """Test that FakeScriptObject wrapping non-DeviceMesh is preserved."""
        graph = fx.Graph()
        p0 = graph.placeholder("tensor_input")
        p0.meta["val"] = torch.randn(2, 3)
        p1 = graph.placeholder("other_fso_input")
        p1.meta["val"] = _make_non_device_mesh_fso()
        graph.output(p0)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        assert not result.modified
        assert pass_instance.removed_indices == []

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_removed_indices_tracks_original_positions(self):
        """Test that removed_indices reflects positions in the original placeholder list."""
        # Graph: [tensor, device_mesh, tensor, device_mesh, tensor]
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=3, device_mesh_indices=[1, 3]
        )

        pass_instance = ReplaceDeviceMeshPlaceholders()
        pass_instance.call(gm)

        assert pass_instance.removed_indices == [1, 3]

    def test_graph_executes_after_pass(self):
        """Test that the modified graph still executes correctly."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[1]
        )

        pass_instance = ReplaceDeviceMeshPlaceholders()
        result = pass_instance.call(gm)

        # Execute with only tensor inputs (DeviceMesh removed)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = result.graph_module(x, y)
        assert torch.equal(output, x + y)
