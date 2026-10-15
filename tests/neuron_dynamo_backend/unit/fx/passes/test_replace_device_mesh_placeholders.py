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

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    MutationKind,
    apply_output_mutations,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState
from torch_neuronx.neuron_dynamo_backend.fx.passes.replace_device_mesh_placeholders import (
    ReplaceDeviceMeshPlaceholders,
)


def _make_state():
    """Create a minimal NeuronFXPassState for testing."""
    return NeuronFXPassState(job=MagicMock(), model_name="test", dumping_enabled=False)


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
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[1]
        )

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        assert result.modified
        assert len(state.pass_mutations[0].input_mutations) == 1
        assert state.pass_mutations[0].input_mutations[0].kind == MutationKind.REMOVE
        assert state.pass_mutations[0].input_mutations[0].indices == [1]

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_multiple_device_meshes_removed(self):
        """Test removal of multiple DeviceMesh placeholders."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[0, 2, 4]
        )

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        assert result.modified
        assert state.pass_mutations[0].input_mutations[0].indices == [0, 2, 4]

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_no_device_mesh_no_modification(self):
        """Test that graph without DeviceMesh placeholders is not modified."""
        gm = _build_graph_with_device_mesh_placeholders(num_tensor_inputs=3, device_mesh_indices=[])

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        assert not result.modified
        assert len(state.pass_mutations) == 0

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 3

    def test_all_device_meshes_removed(self):
        """Test removal when all placeholders are DeviceMesh."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=0, device_mesh_indices=[0, 1]
        )

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        assert result.modified
        assert state.pass_mutations[0].input_mutations[0].indices == [0, 1]

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

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        assert not result.modified
        assert len(state.pass_mutations) == 0

        placeholders = [n for n in result.graph_module.graph.nodes if n.op == "placeholder"]
        assert len(placeholders) == 2

    def test_removed_indices_tracks_original_positions(self):
        """Test that mutation indices reflect positions in the original placeholder list."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=3, device_mesh_indices=[1, 3]
        )

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        pass_instance.call(gm)

        assert state.pass_mutations[0].input_mutations[0].indices == [1, 3]

    def test_graph_executes_after_pass(self):
        """Test that the modified graph still executes correctly."""
        gm = _build_graph_with_device_mesh_placeholders(
            num_tensor_inputs=2, device_mesh_indices=[1]
        )

        state = _make_state()
        pass_instance = ReplaceDeviceMeshPlaceholders(state)
        result = pass_instance.call(gm)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = result.graph_module(x, y)
        assert torch.equal(output, x + y)


class TestReplaceDeviceMeshOutputs:
    """Tests for DeviceMesh passthrough output removal."""

    def _build_graph_with_mesh_output(self, mesh_input_idx: int, mesh_output_idx: int):
        """Build a graph where a DeviceMesh placeholder is also returned as an output."""
        graph = fx.Graph()
        p_tensor = graph.placeholder("tensor_input")
        p_tensor.meta["val"] = torch.randn(2, 3)
        p_mesh = graph.placeholder("mesh_input")
        p_mesh.meta["val"] = _make_device_mesh_fso()

        compute = graph.call_function(torch.ops.aten.add.Tensor, (p_tensor, p_tensor))

        # Output both the compute result and the mesh passthrough
        if mesh_output_idx == 0:
            graph.output((p_mesh, compute))
        else:
            graph.output((compute, p_mesh))

        return fx.GraphModule(torch.nn.Module(), graph), p_mesh

    def test_device_mesh_output_removed(self):
        """DeviceMesh passthrough output is removed from the graph."""
        gm, _ = self._build_graph_with_mesh_output(mesh_input_idx=1, mesh_output_idx=1)
        state = _make_state()
        result = ReplaceDeviceMeshPlaceholders(state).call(gm)

        assert result.modified
        output_node = next(n for n in result.graph_module.graph.nodes if n.op == "output")
        outputs = output_node.args[0]
        assert len(outputs) == 1

    def test_output_mutation_recorded(self):
        """A single ADD output mutation is recorded with the correct index mapping."""
        gm, _ = self._build_graph_with_mesh_output(mesh_input_idx=1, mesh_output_idx=1)
        state = _make_state()
        ReplaceDeviceMeshPlaceholders(state).call(gm)

        assert len(state.pass_mutations[0].output_mutations) == 1
        m = state.pass_mutations[0].output_mutations[0]
        assert m.kind == MutationKind.REMOVE
        assert m.transform_key == "input_passthrough"
        assert 1 in m.indices
        # Output index 1 should map back to original input index 1
        assert m.transform_params["index_to_input"][1] == 1

    def test_output_mutation_restores_mesh_at_runtime(self):
        """apply_output_mutations re-inserts the original mesh at the correct position."""
        gm, _ = self._build_graph_with_mesh_output(mesh_input_idx=1, mesh_output_idx=1)
        state = _make_state()
        ReplaceDeviceMeshPlaceholders(state).call(gm)

        # Simulate original inputs: [tensor, mesh]
        original_mesh = MagicMock(spec=DeviceMesh)
        original_inputs = [torch.randn(2, 3), original_mesh]

        # NEFF outputs only the compute result (mesh was stripped)
        neff_outputs = [torch.randn(2, 3)]

        restored = apply_output_mutations(
            neff_outputs, state.pass_mutations, "cpu", original_inputs
        )
        assert len(restored) == 2
        assert restored[1] is original_mesh

    def test_mesh_output_at_position_zero(self):
        """DeviceMesh output at position 0 is correctly removed and mutation records index 0."""
        gm, _ = self._build_graph_with_mesh_output(mesh_input_idx=1, mesh_output_idx=0)
        state = _make_state()
        result = ReplaceDeviceMeshPlaceholders(state).call(gm)

        assert result.modified
        m = state.pass_mutations[0].output_mutations[0]
        assert 0 in m.indices
        assert m.transform_params["index_to_input"][0] == 1

    def test_non_passthrough_device_mesh_output_raises(self):
        """A DeviceMesh output that isn't a direct placeholder passthrough raises RuntimeError."""
        graph = fx.Graph()
        p_tensor = graph.placeholder("tensor_input")
        p_tensor.meta["val"] = torch.randn(2, 3)

        # Fake an intermediate node whose val is a DeviceMesh FSO (not a placeholder)
        intermediate = graph.call_function(torch.ops.aten.add.Tensor, (p_tensor, p_tensor))
        intermediate.meta["val"] = _make_device_mesh_fso()

        graph.output((p_tensor, intermediate))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with pytest.raises(RuntimeError, match="intermediate computation"):
            ReplaceDeviceMeshPlaceholders(_make_state()).call(gm)
