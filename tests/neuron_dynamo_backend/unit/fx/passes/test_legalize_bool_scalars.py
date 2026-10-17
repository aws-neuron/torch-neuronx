"""
Unit tests for legalize_bool_scalars FX pass.

Tests the FX pass that replaces Python bool literals (True/False) with
integer scalars (1/0) in arithmetic op arguments only.
"""

import pytest
import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend.fx.passes.legalize_bool_scalars import (
    _ARITHMETIC_OPS,
    LegalizeBoolScalars,
)
from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import _add_metadata_to_node


@pytest.fixture
def pass_instance():
    return LegalizeBoolScalars()


def _build_graph_with_scalar_arg(scalar_val, op=torch.ops.aten.add.Tensor):
    """Build an FX graph: op(placeholder, scalar_val)."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    _add_metadata_to_node(x, shape=(), dtype=torch.float32)
    node = graph.call_function(op, args=(x, scalar_val))
    _add_metadata_to_node(node, shape=(), dtype=torch.float32)
    graph.output(node)
    return fx.GraphModule(torch.nn.Module(), graph)


class TestBoolToIntReplacement:
    def test_true_replaced_with_1(self, pass_instance):
        gm = _build_graph_with_scalar_arg(True)
        result = pass_instance.call(gm)

        assert result.modified
        add_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert add_node.args[1] == 1
        assert not isinstance(add_node.args[1], bool)

    def test_false_replaced_with_0(self, pass_instance):
        gm = _build_graph_with_scalar_arg(False)
        result = pass_instance.call(gm)

        assert result.modified
        add_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert add_node.args[1] == 0
        assert not isinstance(add_node.args[1], bool)

    def test_true_as_first_arg(self, pass_instance):
        """True as the first arg (like the graph in the issue)."""
        graph = fx.Graph()
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(True, 1.0))
        _add_metadata_to_node(add, shape=(), dtype=torch.float32)
        graph.output(add)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        add_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert add_node.args[0] == 1
        assert not isinstance(add_node.args[0], bool)

    def test_multiple_bool_args(self, pass_instance):
        graph = fx.Graph()
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(True, False))
        _add_metadata_to_node(add, shape=(), dtype=torch.float32)
        graph.output(add)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        add_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert add_node.args == (1, 0)
        assert not any(isinstance(a, bool) for a in add_node.args)

    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.floor_divide.default,
        ],
    )
    def test_various_arithmetic_ops(self, pass_instance, op):
        gm = _build_graph_with_scalar_arg(True, op=op)
        result = pass_instance.call(gm)

        assert result.modified
        node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert node.args[1] == 1
        assert not isinstance(node.args[1], bool)


class TestNoModification:
    def test_int_scalar_unchanged(self, pass_instance):
        gm = _build_graph_with_scalar_arg(1)
        result = pass_instance.call(gm)
        assert not result.modified

    def test_float_scalar_unchanged(self, pass_instance):
        gm = _build_graph_with_scalar_arg(1.0)
        result = pass_instance.call(gm)
        assert not result.modified

    def test_node_arg_unchanged(self, pass_instance):
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(), dtype=torch.float32)
        y = graph.placeholder("y")
        _add_metadata_to_node(y, shape=(), dtype=torch.float32)
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, y))
        _add_metadata_to_node(add, shape=(), dtype=torch.float32)
        graph.output(add)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)
        assert not result.modified

    def test_no_call_function_nodes(self, pass_instance):
        graph = fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)
        assert not result.modified

    def test_bool_in_non_arithmetic_op_unchanged(self, pass_instance):
        """Bool args in non-arithmetic ops like where should not be touched."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(), dtype=torch.float32)
        y = graph.placeholder("y")
        _add_metadata_to_node(y, shape=(), dtype=torch.float32)
        where = graph.call_function(torch.ops.aten.where.self, args=(True, x, y))
        _add_metadata_to_node(where, shape=(), dtype=torch.float32)
        graph.output(where)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)
        assert not result.modified
        where_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert where_node.args[0] is True


class TestMultipleNodes:
    def test_multiple_arithmetic_nodes_with_bools(self, pass_instance):
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(), dtype=torch.float32)
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, True))
        _add_metadata_to_node(add, shape=(), dtype=torch.float32)
        mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(add, False))
        _add_metadata_to_node(mul, shape=(), dtype=torch.float32)
        graph.output(mul)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert call_nodes[0].args[1] == 1
        assert call_nodes[1].args[1] == 0

    def test_mixed_arithmetic_and_non_arithmetic(self, pass_instance):
        """Bool in arithmetic op is replaced; bool in non-arithmetic op is kept."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(), dtype=torch.float32)
        y = graph.placeholder("y")
        _add_metadata_to_node(y, shape=(), dtype=torch.float32)
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, True))
        _add_metadata_to_node(add, shape=(), dtype=torch.float32)
        where = graph.call_function(torch.ops.aten.where.self, args=(True, add, y))
        _add_metadata_to_node(where, shape=(), dtype=torch.float32)
        graph.output(where)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        add_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.add.Tensor
        )
        where_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.where.self
        )
        assert add_node.args[1] == 1
        assert not isinstance(add_node.args[1], bool)
        assert where_node.args[0] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
