"""Unit tests for dynamic dimension constraint extraction from FX graphs."""

import re

import pytest
import sympy
import torch
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

from torch_neuronx.neuron_dynamo_backend.fx.dynamic_dim_constraints import (
    build_dynamic_dim_constraints,
)


def _capture_post_aot_graph(model, inputs, marks):
    """Compile model and return the post-AOTAutograd FX graph.

    Args:
        model: the module to compile.
        inputs: tuple of example input tensors.
        marks: dict mapping input index -> list of (dim, min, max) to mark dynamic.
    """
    graphs = []

    def fw_compiler(gm, sample_inputs):
        graphs.append(gm)
        return make_boxed_func(gm.forward)

    for i, dims in marks.items():
        for dim, _, _ in dims:
            torch._dynamo.mark_dynamic(inputs[i], dim)

    compiled = torch.compile(model, backend=aot_autograd(fw_compiler=fw_compiler))
    with torch.no_grad():
        compiled(*inputs)

    assert graphs, "Backend was not invoked"
    return graphs[0].graph


def _normalize(constraints):
    """Canonicalize non-deterministic sympy symbol names (s77, s27, ...) to
    s0, s1, ... by argument-index order so results compare exactly across runs.

    Returns (symbol_to_arg_index, tensor_dim_exprs) with canonical symbol names,
    or None when there are no constraints.
    """
    if constraints is None:
        return None
    ordered = sorted(constraints.symbol_to_arg_index.items(), key=lambda kv: kv[1])
    rename = {name: f"s{i}" for i, (name, _) in enumerate(ordered)}

    def canon(expr: str) -> str:
        for name in sorted(rename, key=len, reverse=True):
            expr = re.sub(rf"\b{re.escape(name)}\b", rename[name], expr)
        return expr

    symbols = {rename[name]: idx for name, idx in constraints.symbol_to_arg_index.items()}
    tensor_dims = {
        arg: {dim: canon(expr) for dim, expr in dim_map.items()}
        for arg, dim_map in constraints.tensor_dim_exprs.items()
    }
    return symbols, tensor_dims


class _AddOne(nn.Module):
    def forward(self, x):
        return x + 1.0


class _MatMul(nn.Module):
    def forward(self, x):
        w = torch.ones(4, 8, device=x.device, dtype=x.dtype)
        return torch.matmul(x, w)


class _ReshapeDouble(nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x.reshape(n * 2, 2)


class _Cat(nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=0)


class _RelatedLengths(nn.Module):
    def forward(self, x, y):
        torch._check(y.shape[0] == x.shape[0] + 1)
        return x.sum() + y.sum()


# (name, model, inputs, marks, expected) where expected is None or the
# normalized (symbol_to_arg_index, tensor_dim_exprs) pair. Symbols are canonical
# s0/s1/... ordered by arg index; SymInt inputs always precede tensor inputs.
_CASES = [
    (
        "static_no_constraints",
        _AddOne,
        (torch.randn(4, 4),),
        {},
        None,
    ),
    (
        "single_marked_dim",
        _AddOne,
        (torch.randn(4, 4),),
        {0: [(0, 2, 1024)]},
        ({"s0": 0}, {1: {0: "s0"}}),
    ),
    (
        "matmul_pins_contract_dim",
        _MatMul,
        (torch.randn(4, 4),),
        {0: [(0, 2, 1024)]},
        ({"s0": 0}, {1: {0: "s0"}}),
    ),
    (
        "reshape_input_is_bare_symbol",
        _ReshapeDouble,
        (torch.randn(4, 4),),
        {0: [(0, 2, 1024)]},
        ({"s0": 0}, {1: {0: "s0"}}),
    ),
    (
        "cat_two_dynamic_tensors",
        _Cat,
        (torch.randn(4, 8), torch.randn(3, 8)),
        {0: [(0, 2, 1024)], 1: [(0, 2, 512)]},
        ({"s0": 0, "s1": 2}, {1: {0: "s0"}, 3: {0: "s1"}}),
    ),
    (
        "middle_dim_dynamic_3d",
        _AddOne,
        (torch.randn(2, 4, 8),),
        {0: [(1, 2, 1024)]},
        ({"s0": 0}, {1: {1: "s0"}}),
    ),
    (
        "composite_dim_expr",
        _RelatedLengths,
        (torch.randn(5), torch.randn(6)),
        {0: [(0, 2, 1024)], 1: [(0, 2, 1024)]},
        ({"s0": 1}, {0: {0: "s0 + 1"}, 2: {0: "s0"}}),
    ),
]


@pytest.mark.parametrize(
    "model,inputs,marks,expected",
    [(m, i, mk, e) for _, m, i, mk, e in _CASES],
    ids=[name for name, *_ in _CASES],
)
def test_build_dynamic_dim_constraints(model, inputs, marks, expected):
    graph = _capture_post_aot_graph(model(), inputs, marks)
    constraints = build_dynamic_dim_constraints(graph)
    assert _normalize(constraints) == expected


def _make_symbolic_tensor():
    """A FakeTensor with two independent dynamic dims backed by fresh symbols.

    ``t.size(0)`` and ``t.size(1)`` are distinct bare-symbol SymInts sharing the
    tensor's ShapeEnv, so arithmetic on them yields genuine composite SymInts.
    """
    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    return fake_mode.from_tensor(
        torch.empty(6, 8),
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC],
        ),
        source=LocalSource("ph"),
    )


def _graph_with_placeholder_vals(vals):
    """Build a synthetic FX graph whose placeholders carry the given ``val`` metas."""
    graph = torch.fx.Graph()
    for i, val in enumerate(vals):
        node = graph.placeholder(f"arg{i}")
        node.meta["val"] = val
    return graph


def _const_symint(value):
    """A SymInt specialized to a constant integer (is_Integer, not is_Symbol)."""
    return torch.SymInt(SymNode(sympy.Integer(value), ShapeEnv(), int, hint=value))


def test_composite_symint_input_raises():
    # A SymInt *input* placeholder that is a composite expression (neither a bare
    # symbol nor a constant) is unsupported and must raise.
    composite = _make_symbolic_tensor().size(0) + 1  # expr "s0 + 1"
    graph = _graph_with_placeholder_vals([composite])
    with pytest.raises(AssertionError, match="non-symbol, non-constant"):
        build_dynamic_dim_constraints(graph)


def test_constant_symint_input_is_skipped():
    # A SymInt input specialized to a constant (is_Integer) is not a dynamic-dim
    # source: it is skipped, not asserted on. Only the bare symbols are mapped.
    t = _make_symbolic_tensor()
    graph = _graph_with_placeholder_vals([_const_symint(3), t.size(0), t.size(1), t])
    constraints = build_dynamic_dim_constraints(graph)
    assert _normalize(constraints) == ({"s0": 1, "s1": 2}, {3: {0: "s0", 1: "s1"}})


def test_tensor_dim_references_unmapped_symbol_raises():
    # Map only dim 0's symbol via a scalar arg; the tensor's dim 1 references a
    # second symbol with no scalar input arg, so its expression is unresolvable.
    t = _make_symbolic_tensor()
    graph = _graph_with_placeholder_vals([t.size(0), t])
    with pytest.raises(AssertionError, match="no scalar input arg"):
        build_dynamic_dim_constraints(graph)
