"""
Unit tests for hierarchy location encoding in MLIR.
"""

import pytest
import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import (
    NeuronContextCache,
    NeuronFxImporter,
    _get_hierarchy_string,
)


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SimpleLinear()
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class DeeplyNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NestedModel()
        self.decoder = nn.Linear(2, 4)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, x):
        return self.layers(x)


class MixedModel(nn.Module):
    """Model with both module ops and functional ops."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)  # functional - no module hierarchy
        x = x + 1  # functional - no module hierarchy
        return x


def get_fx_graph(model, input_shape=(2, 4)):
    """Get AOT FX graph from model."""
    x = torch.randn(*input_shape)
    captured = get_aot_graphs(model, x)
    return captured.post_aot_forward_graph


def get_nodes_with_hierarchy(gm):
    """Get nodes that have nn_module_stack metadata."""
    return [n for n in gm.graph.nodes if n.meta.get("nn_module_stack")]


def get_nodes_without_hierarchy(gm):
    """Get call_function nodes without nn_module_stack."""
    return [
        n for n in gm.graph.nodes if n.op == "call_function" and not n.meta.get("nn_module_stack")
    ]


class TestGetHierarchyString:
    """Tests for _get_hierarchy_string."""

    @pytest.fixture
    def cache(self):
        from torch_mlir import ir
        from torch_mlir.dialects import torch as torch_d

        context = ir.Context()
        torch_d.register_dialect(context)
        return NeuronContextCache(context, py_attr_tracker=None)

    def test_simple_linear_exact_hierarchy(self, cache):
        """Test exact hierarchy string for SimpleLinear."""
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        # Linear has 2 ops (weight matmul, bias add)
        assert hierarchies == ["linear|Linear", "linear|Linear"]

    def test_nested_model_exact_hierarchies(self, cache):
        """Test exact hierarchy strings for NestedModel."""
        gm = get_fx_graph(NestedModel())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "layer1|SimpleLinear>layer1.linear|Linear",
            "layer1|SimpleLinear>layer1.linear|Linear",
            "layer2|Linear",
            "layer2|Linear",
        ]

    def test_deeply_nested_exact_hierarchies(self, cache):
        """Test exact hierarchy strings for DeeplyNested."""
        gm = get_fx_graph(DeeplyNested())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "decoder|Linear",
            "decoder|Linear",
            "encoder|NestedModel>encoder.layer1|SimpleLinear>encoder.layer1.linear|Linear",
            "encoder|NestedModel>encoder.layer1|SimpleLinear>encoder.layer1.linear|Linear",
            "encoder|NestedModel>encoder.layer2|Linear",
            "encoder|NestedModel>encoder.layer2|Linear",
        ]

    def test_sequential_exact_hierarchies(self, cache):
        """Test exact hierarchy strings for SequentialModel."""
        gm = get_fx_graph(SequentialModel())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "layers|Sequential>layers.0|Linear",
            "layers|Sequential>layers.0|Linear",
            "layers|Sequential>layers.1|ReLU",
            "layers|Sequential>layers.2|Linear",
            "layers|Sequential>layers.2|Linear",
        ]

    def test_no_hierarchy_returns_none(self, cache):
        """Test nodes without nn_module_stack return None."""
        gm = get_fx_graph(MixedModel())
        nodes_without = get_nodes_without_hierarchy(gm)

        for node in nodes_without:
            assert _get_hierarchy_string(node) is None

    def test_no_lself_prefix(self, cache):
        """Test L['self']. prefix is stripped."""
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        for node in nodes:
            result = _get_hierarchy_string(node)
            assert "L['self']" not in result


class TestGetNodeLocation:
    """Tests for NeuronContextCache.get_node_location."""

    @pytest.fixture
    def cache(self):
        from torch_mlir import ir
        from torch_mlir.dialects import torch as torch_d

        context = ir.Context()
        torch_d.register_dialect(context)
        return NeuronContextCache(context, py_attr_tracker=None)

    def test_node_with_hierarchy_has_hierarchy_in_location(self, cache):
        """Test location contains hierarchy string."""
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            for node in nodes:
                loc = cache.get_node_location(node)
                assert loc is not None
                loc_str = str(loc)
                assert "linear|Linear" in loc_str

    def test_node_with_hierarchy_preserves_source_location(self, cache):
        """Test fused location contains source file info."""
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            for node in nodes:
                if node.stack_trace:  # Has source info
                    loc = cache.get_node_location(node)
                    loc_str = str(loc)
                    # Should have both hierarchy and file reference
                    assert "linear|Linear" in loc_str
                    # Fused location indicated by fused< or multiple loc components
                    assert "fused" in loc_str or ".py" in loc_str

    def test_node_without_hierarchy_still_has_source_location(self, cache):
        """Test nodes without hierarchy still get source location."""
        gm = get_fx_graph(MixedModel())
        nodes_without = get_nodes_without_hierarchy(gm)

        with cache._c:
            for node in nodes_without:
                loc = cache.get_node_location(node)
                # Should return source loc (not None) if stack_trace exists
                if node.stack_trace:
                    assert loc is not None

    def test_fused_location_structure(self, cache):
        """Test fused location has [hierarchy, source] structure."""
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            node = nodes[0]
            loc = cache.get_node_location(node)
            loc_str = str(loc)
            # Fused locations show as fused<...> or loc(fused[...])
            # Should contain hierarchy
            assert "Linear" in loc_str


class TestNeuronFxImporterMLIROutput:
    """Tests for MLIR output with hierarchy locations."""

    def _get_mlir_text(self, model, input_shape=(2, 4)):
        """Get MLIR text with locations."""
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(model, input_shape)
        importer = NeuronFxImporter()
        module = stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)

        import io

        buf = io.StringIO()
        module.operation.print(file=buf, enable_debug_info=True)
        return buf.getvalue()

    def test_simple_linear_hierarchy_in_mlir(self):
        """Test hierarchy string appears in MLIR."""
        mlir_text = self._get_mlir_text(SimpleLinear())
        assert "linear|Linear" in mlir_text

    def test_nested_hierarchy_in_mlir(self):
        """Test nested hierarchy appears in MLIR."""
        mlir_text = self._get_mlir_text(NestedModel())
        assert "layer1|SimpleLinear>layer1.linear|Linear" in mlir_text
        assert "layer2|Linear" in mlir_text

    def test_deeply_nested_hierarchy_in_mlir(self):
        """Test deep hierarchy appears in MLIR."""
        mlir_text = self._get_mlir_text(DeeplyNested())
        expected = "encoder|NestedModel>encoder.layer1|SimpleLinear>encoder.layer1.linear|Linear"
        assert expected in mlir_text

    def test_sequential_hierarchy_in_mlir(self):
        """Test Sequential indices appear in MLIR."""
        mlir_text = self._get_mlir_text(SequentialModel())
        assert "layers|Sequential>layers.0|Linear" in mlir_text

    def test_source_file_preserved_in_mlir(self):
        """Test source file references appear in MLIR."""
        mlir_text = self._get_mlir_text(SimpleLinear())
        # Coarse grained check: source locations should reference .py files
        assert ".py" in mlir_text

    def test_line_numbers_preserved_in_mlir(self):
        """Test line numbers appear in MLIR locations."""
        mlir_text = self._get_mlir_text(SimpleLinear())
        # Line numbers appear as :NN: or :NN in locations
        import re

        assert re.search(r":\d+", mlir_text)
