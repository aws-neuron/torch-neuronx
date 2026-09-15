"""
Unit tests for hierarchy location encoding in MLIR.
"""

import pytest
import torch
import torch.nn as nn

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import (
    ATEN_HIERARCHY_ENV_VAR,
    NeuronContextCache,
    NeuronFxImporter,
    _get_hierarchy_string,
)


@pytest.fixture
def aten_hierarchy_enabled(monkeypatch):
    """Turn on the opt-in aten-op hierarchy leaf for the duration of a test."""
    monkeypatch.setenv(ATEN_HIERARCHY_ENV_VAR, "1")


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
        x = torch.relu(x)
        x = x + 1
        return x


def _mock_functional_api(x):
    """Stand-in for a vLLM functional API implemented outside nn.Module."""
    return torch.relu(x * 2)


class FunctionalBlock(nn.Module):
    def forward(self, x):
        return _mock_functional_api(x)


class FunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = FunctionalBlock()

    def forward(self, x):
        return self.block(x)


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
    """Tests for _get_hierarchy_string (ClassName[attr][count]/... format)."""

    def test_simple_linear(self):
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == ["Linear[linear][0]", "Linear[linear][0]"]

    def test_nested_model(self):
        gm = get_fx_graph(NestedModel())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "Linear[layer2][0]",
            "Linear[layer2][0]",
            "SimpleLinear[layer1][0]/Linear[linear][0]",
            "SimpleLinear[layer1][0]/Linear[linear][0]",
        ]

    def test_deeply_nested(self):
        gm = get_fx_graph(DeeplyNested())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "Linear[decoder][0]",
            "Linear[decoder][0]",
            "NestedModel[encoder][0]/Linear[layer2][0]",
            "NestedModel[encoder][0]/Linear[layer2][0]",
            "NestedModel[encoder][0]/SimpleLinear[layer1][0]/Linear[linear][0]",
            "NestedModel[encoder][0]/SimpleLinear[layer1][0]/Linear[linear][0]",
        ]

    def test_sequential(self):
        gm = get_fx_graph(SequentialModel())
        nodes = get_nodes_with_hierarchy(gm)

        hierarchies = sorted([_get_hierarchy_string(n) for n in nodes])
        assert hierarchies == [
            "Sequential[layers][0]/Linear[0][0]",
            "Sequential[layers][0]/Linear[0][0]",
            "Sequential[layers][0]/Linear[2][0]",
            "Sequential[layers][0]/Linear[2][0]",
            "Sequential[layers][0]/ReLU[1][0]",
        ]

    def test_no_hierarchy_returns_none(self):
        gm = get_fx_graph(MixedModel())
        nodes_without = get_nodes_without_hierarchy(gm)

        for node in nodes_without:
            assert _get_hierarchy_string(node) is None

    def test_no_lself_prefix(self):
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        for node in nodes:
            result = _get_hierarchy_string(node)
            assert "L['self']" not in result

    def test_no_pipe_character(self):
        """Hierarchy strings must not contain '|' — routed to op_name, not module_hierarchy."""
        gm = get_fx_graph(DeeplyNested())
        nodes = get_nodes_with_hierarchy(gm)

        for node in nodes:
            result = _get_hierarchy_string(node)
            if result:
                assert "|" not in result

    def test_slash_delimited(self):
        gm = get_fx_graph(NestedModel())
        nodes = get_nodes_with_hierarchy(gm)

        for node in nodes:
            result = _get_hierarchy_string(node)
            if result and "/" in result:
                parts = result.split("/")
                assert len(parts) >= 2
                for part in parts:
                    assert "[" in part and "]" in part

    def test_captured_functional_api_includes_stack_trace_function_frame(self):
        """Mirror vLLM's module-to-free-function call structure end to end."""
        gm = get_fx_graph(FunctionalModel())

        function_nodes = [
            node
            for node in gm.graph.nodes
            if "in _mock_functional_api" in node.meta.get("stack_trace", "")
        ]
        assert function_nodes, "functional API frame was not preserved through AOT capture"

        hierarchies = {_get_hierarchy_string(node) for node in function_nodes}
        assert any(
            "FunctionalBlock[block][0]/test_hierarchy_location[_mock_functional_api]" in hierarchy
            for hierarchy in hierarchies
        )

    def test_stack_trace_interleaves_module_and_function_frames(self):
        gm = get_fx_graph(SimpleLinear())
        node = get_nodes_with_hierarchy(gm)[0]
        node.meta["nn_module_stack"] = {
            "model@0": ("L['self'].model", FunctionalModel),
            "block@0": ("L['self'].model.block", FunctionalBlock),
        }
        node.meta["stack_trace"] = """\
  File "/models/model.py", line 10, in forward
    return self.block(x)
  File "/models/block.py", line 20, in forward
    return functional_attention(x)
  File "/models/attention.py", line 30, in functional_attention
    return torch.relu(x)
"""

        assert _get_hierarchy_string(node) == (
            "FunctionalModel[model][0]/FunctionalBlock[block][0]/attention[functional_attention]"
        )

    def test_call_hierarchy_is_not_used(self):
        gm = get_fx_graph(SimpleLinear())
        node = get_nodes_with_hierarchy(gm)[0]
        node.meta["call_hierarchy"] = [{"type": "function", "name": "experimental_api", "count": 0}]
        node.meta.pop("stack_trace", None)

        assert _get_hierarchy_string(node) == "Linear[linear][0]"

    def test_function_only_stack_trace_without_module_stack(self):
        gm = get_fx_graph(MixedModel())
        node = get_nodes_without_hierarchy(gm)[0]
        node.meta["stack_trace"] = """\
  File "/models/functional.py", line 10, in functional_attention
    return torch.relu(x)
"""

        assert _get_hierarchy_string(node) == "functional[functional_attention]"

    def test_forward_module_stack_used_with_stack_trace_function(self):
        gm = get_fx_graph(MixedModel())
        node = get_nodes_without_hierarchy(gm)[0]
        node.meta["fwd_nn_module_stack"] = {
            "model@0": ("L['self'].model", FunctionalModel),
        }
        node.meta["stack_trace"] = """\
  File "/models/model.py", line 10, in forward
    return functional_attention(x)
  File "/models/attention.py", line 20, in functional_attention
    return torch.relu(x)
"""

        assert _get_hierarchy_string(node) == (
            "FunctionalModel[model][0]/attention[functional_attention]"
        )

    def test_misaligned_trace_falls_back_to_modules_then_functions(self):
        gm = get_fx_graph(SimpleLinear())
        node = get_nodes_with_hierarchy(gm)[0]
        node.meta["stack_trace"] = """\
  File "/models/helper.py", line 10, in helper
    return torch.relu(x)
"""

        assert _get_hierarchy_string(node) == "Linear[linear][0]/helper[helper]"

    def test_stack_trace_ignores_wrappers_and_synthetic_frames(self):
        gm = get_fx_graph(MixedModel())
        node = get_nodes_without_hierarchy(gm)[0]
        node.meta["stack_trace"] = """\
  File "/torch/nn/modules/module.py", line 1, in _call_impl
    return forward_call(*args)
  File "/models/model.py", line 2, in <listcomp>
    return [functional_attention(x)]
  File "/models/attention.py", line 3, in functional_attention
    return torch.relu(x)
"""

        assert _get_hierarchy_string(node) == "attention[functional_attention]"

    def test_fallback_without_stack_trace(self):
        gm = get_fx_graph(SimpleLinear())
        node = get_nodes_with_hierarchy(gm)[0]
        node.meta.pop("stack_trace", None)

        assert _get_hierarchy_string(node) == "Linear[linear][0]"


class TestAtenHierarchyLeaf:
    """Tests for the opt-in aten-op leaf (TORCH_NEURONX_ATEN_HIERARCHY)."""

    def test_no_aten_leaf_by_default(self):
        """Default (flag off) emits module-only hierarchy — no aten segment."""
        gm = get_fx_graph(SimpleLinear())
        for node in get_nodes_with_hierarchy(gm):
            assert _get_hierarchy_string(node) == "Linear[linear][0]"

    def test_aten_leaf_appended_when_enabled(self, aten_hierarchy_enabled):
        """Flag on appends aten.<op>.<overload>[<node_name>] to call_function nodes."""
        gm = get_fx_graph(SimpleLinear())
        results = [
            _get_hierarchy_string(n)
            for n in get_nodes_with_hierarchy(gm)
            if n.op == "call_function"
        ]
        assert results, "expected call_function nodes carrying a module stack"
        for r in results:
            assert r.startswith("Linear[linear][0]/aten.")
            # leaf is aten.<op>.<overload>[<node_name>]
            assert r.rsplit("/", 1)[-1].endswith("]")

    def test_leaf_only_on_call_function_nodes(self, aten_hierarchy_enabled):
        """Non-call_function nodes get no aten leaf."""
        gm = get_fx_graph(SimpleLinear())
        for node in gm.graph.nodes:
            if node.op == "call_function":
                continue
            result = _get_hierarchy_string(node)
            if result:
                assert "/aten." not in result

    def test_leaf_without_module_stack(self, aten_hierarchy_enabled):
        """A call_function node with no module stack still yields a bare aten leaf."""
        gm = get_fx_graph(MixedModel())
        for node in get_nodes_without_hierarchy(gm):
            result = _get_hierarchy_string(node)
            assert result is not None
            assert result.startswith("aten.")
            assert "/" not in result

    def test_deterministic_across_repeated_traces(self, aten_hierarchy_enabled):
        """Same model traced twice yields identical strings.

        This is the property the NEFF cache key depends on: the hierarchy is fused
        into the MLIR loc(...) and the serialized module is hashed, so a rerun of an
        unchanged model must reproduce byte-identical hierarchy strings to hit cache.
        """
        first = [
            _get_hierarchy_string(n)
            for n in get_fx_graph(SimpleLinear()).graph.nodes
            if n.op == "call_function"
        ]
        second = [
            _get_hierarchy_string(n)
            for n in get_fx_graph(SimpleLinear()).graph.nodes
            if n.op == "call_function"
        ]
        assert first == second

    def test_repeated_same_target_distinguished_by_node_name(self, aten_hierarchy_enabled):
        """Repeated calls to one submodule stay distinguishable via the leaf."""

        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x) + self.fc(x)

        gm = get_fx_graph(Repeated())
        results = [_get_hierarchy_string(n) for n in gm.graph.nodes if n.op == "call_function"]
        # node_name suffixes (addmm, addmm_1, ...) keep every op uniquely labeled.
        assert len(set(results)) == len(results)

    def test_mlir_contains_aten_leaf(self, aten_hierarchy_enabled):
        """The aten leaf reaches the printed MLIR locations."""
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(SimpleLinear())
        importer = NeuronFxImporter()
        module = stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)

        import io

        buf = io.StringIO()
        module.operation.print(file=buf, enable_debug_info=True)
        mlir_text = buf.getvalue()
        assert "Linear[linear][0]/aten." in mlir_text


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
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            for node in nodes:
                loc = cache.get_node_location(node)
                assert loc is not None
                loc_str = str(loc)
                assert "Linear[linear][0]" in loc_str

    def test_node_with_hierarchy_preserves_source_location(self, cache):
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            for node in nodes:
                if node.stack_trace:
                    loc = cache.get_node_location(node)
                    loc_str = str(loc)
                    assert "Linear[linear][0]" in loc_str
                    assert "fused" in loc_str or ".py" in loc_str

    def test_node_without_hierarchy_still_has_source_location(self, cache):
        gm = get_fx_graph(MixedModel())
        nodes_without = get_nodes_without_hierarchy(gm)

        with cache._c:
            for node in nodes_without:
                loc = cache.get_node_location(node)
                if node.stack_trace:
                    assert loc is not None

    def test_fused_location_structure(self, cache):
        gm = get_fx_graph(SimpleLinear())
        nodes = get_nodes_with_hierarchy(gm)

        with cache._c:
            node = nodes[0]
            loc = cache.get_node_location(node)
            loc_str = str(loc)
            assert "Linear" in loc_str


class TestNeuronFxImporterMLIROutput:
    """Tests for MLIR output with hierarchy locations."""

    def _get_mlir_text(self, model, input_shape=(2, 4)):
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(model, input_shape)
        importer = NeuronFxImporter()
        module = stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)

        import io

        buf = io.StringIO()
        module.operation.print(file=buf, enable_debug_info=True)
        return buf.getvalue()

    def test_simple_linear_hierarchy_in_mlir(self):
        mlir_text = self._get_mlir_text(SimpleLinear())
        assert "Linear[linear][0]" in mlir_text

    def test_nested_hierarchy_in_mlir(self):
        mlir_text = self._get_mlir_text(NestedModel())
        assert "SimpleLinear[layer1][0]/Linear[linear][0]" in mlir_text
        assert "Linear[layer2][0]" in mlir_text

    def test_deeply_nested_hierarchy_in_mlir(self):
        mlir_text = self._get_mlir_text(DeeplyNested())
        assert "NestedModel[encoder][0]/SimpleLinear[layer1][0]/Linear[linear][0]" in mlir_text

    def test_sequential_hierarchy_in_mlir(self):
        mlir_text = self._get_mlir_text(SequentialModel())
        assert "Sequential[layers][0]/Linear[0][0]" in mlir_text

    def test_source_file_preserved_in_mlir(self):
        mlir_text = self._get_mlir_text(SimpleLinear())
        assert ".py" in mlir_text

    def test_line_numbers_preserved_in_mlir(self):
        mlir_text = self._get_mlir_text(SimpleLinear())
        import re

        assert re.search(r":\d+", mlir_text)


class TestHasHierarchyFlag:
    """Tests for NeuronFxImporter.has_hierarchy flag tracking."""

    def _import_model(self, model, input_shape=(2, 4)):
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(model, input_shape)
        importer = NeuronFxImporter()
        module = stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)
        return importer, module

    def test_has_hierarchy_true_for_model_with_modules(self):
        """Flag is set when a model with nn.Module ops is imported."""
        importer, _ = self._import_model(SimpleLinear())
        assert importer.has_hierarchy is True

    def test_has_hierarchy_true_for_nested_model(self):
        """Flag is set when a model with nested nn.Modules is imported."""
        importer, _ = self._import_model(DeeplyNested())
        assert importer.has_hierarchy is True

    def test_has_hierarchy_initially_false(self):
        """Flag defaults to False before any import happens."""
        importer = NeuronFxImporter()
        assert importer.has_hierarchy is False

    def test_has_hierarchy_true_when_only_aten_suffix(self, aten_hierarchy_enabled):
        """Flag is set even without nn_module_stack — aten-op suffix counts as hierarchy."""
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(SimpleLinear())
        # Strip module-stack metadata; call_function nodes still produce aten.<op>[<name>].
        for node in gm.graph.nodes:
            node.meta.pop("nn_module_stack", None)
            node.meta.pop("fwd_nn_module_stack", None)
        importer = NeuronFxImporter()
        stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)
        assert importer.has_hierarchy is True

    def test_has_hierarchy_false_when_only_aten_suffix_and_flag_off(self):
        """With the flag off, a stripped graph yields no hierarchy at all."""
        from torch_mlir.fx import OutputType, stateless_fx_import

        gm = get_fx_graph(SimpleLinear())
        for node in gm.graph.nodes:
            node.meta.pop("nn_module_stack", None)
            node.meta.pop("fwd_nn_module_stack", None)
        importer = NeuronFxImporter()
        stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)
        assert importer.has_hierarchy is False

    def test_has_hierarchy_false_when_no_hierarchy_nor_aten_suffix(self):
        """Flag stays False when every processed node produces no hierarchy string.

        Uses placeholder/output nodes (op != call_function) with their module-stack
        stripped — neither branch of _get_hierarchy_string contributes.
        """
        from torch_mlir import ir
        from torch_mlir.dialects import torch as torch_d

        from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronContextCache

        gm = get_fx_graph(SimpleLinear())
        non_call_nodes = [n for n in gm.graph.nodes if n.op != "call_function"]
        assert non_call_nodes, "Expected placeholder/output nodes in the graph"
        for node in non_call_nodes:
            node.meta.pop("nn_module_stack", None)
            node.meta.pop("fwd_nn_module_stack", None)
            assert _get_hierarchy_string(node) is None

        context = ir.Context()
        torch_d.register_dialect(context)
        cache = NeuronContextCache(context, py_attr_tracker=None)
        with cache._c:
            for node in non_call_nodes:
                cache.get_node_location(node)
        assert cache.has_hierarchy is False


class TestHierarchySplitterCompatibleAttribute:
    """Tests for stablehlo.hierarchy_splitter_compatible module attribute after
    full FX -> StableHLO conversion (including all MLIR passes)."""

    def _convert(self, model, input_shape=(2, 4)):
        from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
            convert_fx_to_stablehlo,
        )

        gm = get_fx_graph(model, input_shape)
        x = torch.randn(*input_shape)
        stablehlo, _, _ = convert_fx_to_stablehlo(gm, [x])
        return stablehlo

    def test_attribute_set_for_model_with_modules(self):
        """Attribute is present on the final StableHLO module when hierarchy was used."""
        stablehlo = self._convert(SimpleLinear())
        assert "stablehlo.hierarchy_splitter_compatible" in stablehlo.operation.attributes

    def test_attribute_value_is_true(self):
        """Attribute value is the string 'true'."""
        stablehlo = self._convert(NestedModel())
        attr = stablehlo.operation.attributes["stablehlo.hierarchy_splitter_compatible"]
        assert str(attr.value) == "true"

    def test_attribute_set_even_without_module_stack(self, aten_hierarchy_enabled):
        """Attribute is still present when only aten-suffix hierarchy is emitted."""
        from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
            convert_fx_to_stablehlo,
        )

        gm = get_fx_graph(SimpleLinear())
        for node in gm.graph.nodes:
            node.meta.pop("nn_module_stack", None)
            node.meta.pop("fwd_nn_module_stack", None)
        x = torch.randn(2, 4)
        stablehlo, _, _ = convert_fx_to_stablehlo(gm, [x])
        assert "stablehlo.hierarchy_splitter_compatible" in stablehlo.operation.attributes
