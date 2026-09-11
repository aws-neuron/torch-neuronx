"""
Unit tests for FX importer
"""

import ast
import operator
import re

import pytest
import sympy
import torch
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_mlir import fx
from torch_mlir import ir as mlir_ir
from torch_mlir.dialects import func as func_d
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import _emit_operation
from torch_mlir.fx import OutputType, stateless_fx_import
from torch_mlir.ir import IntegerAttr, IntegerType

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.neuron_dynamo_backend.fx.fx_hooks import CustomFxImporterHooks
from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronFxImporter


def _extract_dense_data(module_str):
    """Extract data from dense<...> patterns in an MLIR module string."""
    matches = re.findall(r"dense<([^>]+)>", module_str)
    out = []
    for m in matches:
        try:
            out.append(ast.literal_eval(m.replace("true", "True").replace("false", "False")))
        except (ValueError, SyntaxError):
            continue
    return out


class TestFxImporterTensorConstants:
    def test_neuron_device_scalar_constant(self):
        """Scalar constant created on neuron device via torch.tensor(0.0, device=x.device)."""

        class Model(nn.Module):
            def forward(self, x):
                mask = x > 0
                min_dtype = torch.tensor(-3.389e38, device=x.device, dtype=x.dtype)
                result = torch.where(
                    mask, torch.tensor(0.0, device=mask.device, dtype=x.dtype), min_dtype
                )
                return result

        model = Model().to(torch.bfloat16).to("neuron")
        x = torch.randn(1, 512, dtype=torch.bfloat16, device="neuron")

        compiled = torch.compile(model, fullgraph=True, backend="neuron")
        out = compiled(x)
        assert out.shape == (1, 512)

    def test_get_attr_constant_under_fake_tensor_mode(self, monkeypatch):
        """The fx importer runs inside FakeTensorMode; get_attr tensor
        constants are real. A real tensor whose .cpu() dispatches
        aten._to_copy would trip fake mode's real-tensor guard without
        unset_fake_temporarily in resolve_literal."""

        constant = [[1.0, 2.0], [3.0, 4.0]]

        def _dispatching_cpu(self):
            return torch.ops.aten._to_copy.default(self, device=torch.device("cpu"))

        monkeypatch.setattr(torch.Tensor, "cpu", _dispatching_cpu)

        class Model(nn.Module):
            def forward(self, x):
                return x + torch.tensor(constant, dtype=torch.float32)

        gm = get_aot_graphs(Model(), torch.ones((2, 2), dtype=torch.float32)).post_aot_forward_graph

        with FakeTensorMode(allow_non_fake_inputs=False):
            mlir_module = fx.stateless_fx_import(gm, hooks=CustomFxImporterHooks())

        module_str = str(mlir_module)
        assert "torch.vtensor.literal" in module_str
        assert constant in _extract_dense_data(module_str)


def _get_mlir_op_names(module):
    """Collect all op names from an MLIR module, resolving torch.operator names."""
    ops = []

    def _walk(op):
        ops.append(op.operation.name)
        for region in op.operation.regions:
            for block in region:
                for inner_op in block.operations:
                    _walk(inner_op)

    for op in module.body.operations:
        _walk(op)
    return ops


def _compile_and_inspect(model, example_inputs, dynamic_dims):
    """Compile model via torch.compile with a custom backend that:
    1. Captures the pre-AOT FX graph (to verify symbolic builtins/ops)
    2. Runs AOT autograd -> imports post-AOT graph into MLIR module

    Args:
        model: The model to compile.
        example_inputs: Tuple of example inputs.
        dynamic_dims: Dict mapping input index -> set of dynamic dim indices.

    Returns:
        (pre_aot_graph, mlir_module) tuple.
    """
    mlir_modules = []
    pre_aot_graphs = []

    def fw_compiler(gm, sample_inputs):
        importer = NeuronFxImporter()
        module = stateless_fx_import(gm, output_type=OutputType.RAW, fx_importer=importer)
        mlir_modules.append(module)
        return make_boxed_func(gm.forward)

    aot_backend = aot_autograd(fw_compiler=fw_compiler)

    def capture_backend(gm, sample_inputs):
        pre_aot_graphs.append(gm)
        return aot_backend(gm, sample_inputs)

    compiled = torch.compile(model, backend=capture_backend, dynamic=True)
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and i in dynamic_dims:
            for dim in dynamic_dims[i]:
                torch._dynamo.mark_dynamic(inp, dim)
    with torch.no_grad():
        compiled(*inputs)

    assert mlir_modules, "Backend was not invoked"
    return pre_aot_graphs[0], mlir_modules[0]


class TestPyBuiltinSymbolicImport:
    """Tests that pow/neg builtins on symbolic ints import successfully."""

    @pytest.mark.xfail(reason="pow.int not in ODS yet")
    def test_pow_symbolic_shape(self):
        """x.shape[0] ** 2 triggers pow builtin and lowers to torch.aten.pow.int."""

        class Model(nn.Module):
            def forward(self, x):
                scale = x.shape[0] ** 2
                return x * scale

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # FX graph must contain the pow builtin
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.pow in targets, f"Expected operator.pow in FX graph, got: {targets}"

        # MLIR must contain torch.aten.pow.int
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.pow.int" in mlir_ops
        ), f"Expected torch.aten.pow.int in MLIR, got: {mlir_ops}"

    def test_neg_symbolic_shape(self):
        """Unary -x.shape[0] triggers neg builtin and lowers to torch.aten.neg.int."""

        class Model(nn.Module):
            def forward(self, x):
                neg_batch = -(x.shape[0])
                return x + neg_batch

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # FX graph must contain the neg builtin
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.neg in targets, f"Expected operator.neg in FX graph, got: {targets}"

        # MLIR must contain torch.aten.neg.int
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.neg.int" in mlir_ops
        ), f"Expected torch.aten.neg.int in MLIR, got: {mlir_ops}"


class TestSymFloatSymIntMapping:
    """Tests that sym_float/sym_int ops route to the correct aten equivalents."""

    def test_sym_float_conversion(self):
        """float(x.shape[0]) should dispatch via sym_float in pre-AOT graph."""

        class Model(nn.Module):
            def forward(self, x):
                scale = float(x.shape[0])
                return x * scale

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # FX graph must contain torch.sym_float
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert torch.sym_float in targets, f"Expected torch.sym_float in FX graph, got: {targets}"

        # MLIR must produce a scalar_tensor (the float scalar is materialized)
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.scalar_tensor" in mlir_ops
        ), f"Expected torch.aten.scalar_tensor in MLIR, got: {mlir_ops}"

    def test_sym_int_conversion(self):
        """int() on a symbolic float should dispatch via sym_int."""

        class Model(nn.Module):
            def forward(self, x):
                # Force a sym_int call
                i = int(bool(x.shape[0]))
                return x * i

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # FX graph must contain both sym_float and sym_int
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert torch.sym_int in targets, f"Expected torch.sym_int in FX graph, got: {targets}"

        # Should import successfully - the torch.ops.aten.Int.Scalar gets folded
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.constant.int" in mlir_ops
        ), f"Expected torch.constant.int in MLIR, got {mlir_ops}"


class TestSymTypeListOperands:
    """Tests that SymInt values in list operands (e.g. reshape shape lists) are handled."""

    def test_reshape_with_symbolic_dim(self):
        """torch.reshape with a symbolic batch dim passed in the shape list."""

        class Model(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return x.reshape(batch, -1)

        x = torch.randn(4, 2, 4)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # Pre-AOT graph: the reshape op uses a symbolic dim in its shape arg.
        # After AOT, aten.view takes a list arg containing the symbolic dim.
        mlir_ops = _get_mlir_op_names(mlir_mod)
        # The list containing the symbolic dim must be constructed
        assert (
            "torch.prim.ListConstruct" in mlir_ops
        ), f"Expected torch.prim.ListConstruct (for sym dim list), got: {mlir_ops}"
        assert "torch.aten.view" in mlir_ops, f"Expected torch.aten.view in MLIR, got: {mlir_ops}"

    def test_expand_with_symbolic_dims(self):
        """torch.expand with multiple symbolic dims in the size list."""

        class Model(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                seq = x.shape[1]
                # Expand a [1, 1, D] tensor to [batch, seq, D]
                ones = torch.ones(1, 1, x.shape[2])
                return ones.expand(batch, seq, -1) + x

        x = torch.randn(4, 8, 16)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0, 1}})

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.prim.ListConstruct" in mlir_ops
        ), f"Expected torch.prim.ListConstruct (for sym dim list), got: {mlir_ops}"


class TestSymToValueResolution:
    """Tests that symbolic placeholders are resolved when passed as raw scalar arguments."""

    def test_add_scalar_with_symbolic_batch(self):
        """x + x.shape[0] should resolve the symbolic batch dim as a scalar."""

        class Model(nn.Module):
            def forward(self, x):
                return x + x.shape[0]

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # Pre-AOT graph must have an add with a symbolic arg
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.add in targets, f"Expected operator.add in FX graph, got: {targets}"

        # MLIR must have add.Scalar (symbolic dim used as scalar)
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.add.Scalar" in mlir_ops
        ), f"Expected torch.aten.add.Scalar in MLIR, got: {mlir_ops}"

        # Verify the scalar operand of add.Scalar is the symbolic block arg
        func = next(iter(mlir_mod.body.operations))
        block = func.body.blocks[0]
        block_args = list(block.arguments)
        int_block_args = [a for a in block_args if str(a.type) == "!torch.int"]
        assert int_block_args, "Expected !torch.int block args for symbolic dims"

        add_scalar_ops = [
            op for op in block.operations if op.operation.name == "torch.aten.add.Scalar"
        ]
        assert add_scalar_ops, f"Expected at least 1 add.Scalar op, got {block.operations}"
        op = add_scalar_ops[0]
        operands = list(op.operation.operands)
        # operands[0] is the tensor, operands[1] is the scalar
        scalar_operand = operands[1]
        assert scalar_operand in int_block_args, (
            "add.Scalar scalar operand should be a symbolic block arg "
            "(proving _sym_to_value resolved correctly), but got operand "
            "from a non-block-arg source"
        )


class TestScalarArgDetection:
    """Tests that SymInt/SymFloat args trigger Tensor->Scalar op conversion."""

    def test_mul_tensor_with_sym_scalar(self):
        """x * x.shape[0] should become aten.mul.Scalar, not aten.mul.Tensor."""

        class Model(nn.Module):
            def forward(self, x):
                return x * x.shape[0]

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # Pre-AOT graph must have the mul with a symbolic arg
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.mul in targets, f"Expected operator.mul in FX graph, got: {targets}"

        # MLIR must use the Scalar variant
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.mul.Scalar" in mlir_ops
        ), f"Expected torch.aten.mul.Scalar in MLIR, got: {mlir_ops}"
        assert (
            "torch.aten.mul.Tensor" not in mlir_ops
        ), f"Should NOT have torch.aten.mul.Tensor (should be Scalar), got: {mlir_ops}"

    def test_sub_tensor_with_sym_scalar(self):
        """x - x.shape[0] should become aten.sub.Scalar."""

        class Model(nn.Module):
            def forward(self, x):
                return x - x.shape[0]

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.sub.Scalar" in mlir_ops
        ), f"Expected torch.aten.sub.Scalar in MLIR, got: {mlir_ops}"


class TestSymbolicExprCompilation:
    """Tests that derived symbolic expressions compile to aten int ops via _import_symbolic_expr."""

    def test_add_expr(self):
        """x.shape[0] + 1 compiles to torch.aten.add.int."""

        class Model(nn.Module):
            def forward(self, x):
                return x * (x.shape[0] + 1)

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # Pre-AOT graph must contain the add builtin on symbolic values
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.add in targets, f"Expected operator.add in FX graph, got: {targets}"

        # MLIR must contain torch.aten.add.int (int+int) and mul.Scalar (tensor*int)
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.add.int" in mlir_ops
        ), f"Expected torch.aten.add.int in MLIR, got: {mlir_ops}"
        assert (
            "torch.aten.mul.Scalar" in mlir_ops
        ), f"Expected torch.aten.mul.Scalar in MLIR, got: {mlir_ops}"

    def test_mul_expr(self):
        """x.shape[0] * 2 compiles to torch.aten.mul.int."""

        class Model(nn.Module):
            def forward(self, x):
                return x * (x.shape[0] * 2)

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.mul in targets, f"Expected operator.mul in FX graph, got: {targets}"

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.mul.int" in mlir_ops
        ), f"Expected torch.aten.mul.int in MLIR, got: {mlir_ops}"

    def test_compound_expr(self):
        """x.shape[0] * 2 + 1 compiles to both mul.int and add.int."""

        class Model(nn.Module):
            def forward(self, x):
                return x * (x.shape[0] * 2 + 1)

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert operator.mul in targets, f"Expected operator.mul in FX graph, got: {targets}"
        assert operator.add in targets, f"Expected operator.add in FX graph, got: {targets}"

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.mul.int" in mlir_ops
        ), f"Expected torch.aten.mul.int in MLIR, got: {mlir_ops}"
        assert (
            "torch.aten.add.int" in mlir_ops
        ), f"Expected torch.aten.add.int in MLIR, got: {mlir_ops}"


class TestDivTensorModeScalar:
    """Tests that div.Tensor_mode with scalar arg converts to div.Scalar_mode."""

    def test_div_tensor_mode_with_sym_scalar(self):
        """torch.div(x, shape_val, rounding_mode='trunc') should use Scalar variant."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.div(x, x.shape[0], rounding_mode="trunc")

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        # Pre-AOT graph must contain torch.div
        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        assert any(
            "div" in str(t) for t in targets
        ), f"Expected a div op in FX graph, got: {targets}"

        # MLIR must use the Scalar_mode variant
        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.div.Scalar_mode" in mlir_ops
        ), f"Expected torch.aten.div.Scalar_mode in MLIR, got: {mlir_ops}"

    def test_div_tensor_mode_floor(self):
        """torch.div(x, shape_val, rounding_mode='floor') should also use Scalar variant."""

        class Model(nn.Module):
            def forward(self, x):
                return torch.div(x, x.shape[0], rounding_mode="floor")

        x = torch.randn(4, 8)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.aten.div.Scalar_mode" in mlir_ops
        ), f"Expected torch.aten.div.Scalar_mode in MLIR, got: {mlir_ops}"


class TestImportSymbolicExprUnit:
    """Unit tests for _import_symbolic_expr by directly constructing SymInts
    with known sympy expressions and calling the method."""

    def _setup_importer(self):
        """Set up a NeuronFxImporter with a function that has one !torch.int arg,
        and a GraphNodeImporter whose _sym_to_value maps a sympy symbol to that arg.

        Returns (gni, loc, block, s0) where:
          - gni: GraphNodeImporter ready for _import_symbolic_expr calls
          - loc: MLIR location
          - block: the entry block (to inspect generated ops)
          - s0: sympy.Symbol mapped to block arg[0]
        """
        from torch_mlir.extras.fx_importer import GraphNodeImporter

        importer = NeuronFxImporter()
        context = importer._c
        with context:
            loc = mlir_ir.Location.unknown()
            with loc:
                int_type = importer._cc.torch_int_type
                func_type = mlir_ir.FunctionType.get([int_type], [int_type])
                func_op = mlir_ir.Operation.create(
                    "func.func",
                    results=[],
                    operands=[],
                    loc=loc,
                    regions=1,
                    attributes={
                        "sym_name": mlir_ir.StringAttr.get("test"),
                        "function_type": mlir_ir.TypeAttr.get(func_type),
                    },
                )
                entry_block = func_op.regions[0].blocks.append(int_type)

                gni = GraphNodeImporter(importer, context, importer._cc, entry_block)

                # Create a sympy symbol and map it to block arg
                s0 = sympy.Symbol("s0", integer=True)
                gni._sym_to_value[str(s0)] = entry_block.arguments[0]

                return gni, loc, entry_block, s0

    @staticmethod
    def _make_sym_arg(expr):
        """Create a mock object with .node = expr that _import_symbolic_expr can process."""

        class MockSymArg:
            def __init__(self, e):
                self.node = e

            def __str__(self):
                return str(self.node)

        return MockSymArg(expr)

    @pytest.mark.parametrize(
        "expr_fn,expected_ops,chained_pair",
        [
            pytest.param(
                lambda s: s + 2,
                ["torch.aten.add.int"],
                None,
                id="add",
            ),
            pytest.param(
                lambda s: s * 3,
                ["torch.aten.mul.int"],
                None,
                id="mul",
            ),
            pytest.param(
                lambda s: s**2,
                ["torch.aten.pow.int"],
                None,
                id="pow",
                marks=pytest.mark.xfail(reason="pow.int not in ODS yet"),
            ),
            pytest.param(
                lambda s: s * 2 + 1,
                ["torch.aten.mul.int", "torch.aten.add.int"],
                ("torch.aten.mul.int", "torch.aten.add.int"),
                id="compound_mul_add",
            ),
            pytest.param(
                lambda s: (s + 3) * (s + 1),
                ["torch.aten.add.int", "torch.aten.mul.int"],
                ("torch.aten.add.int", "torch.aten.mul.int"),
                id="compound_add_mul",
            ),
        ],
    )
    def test_symbolic_expr(self, expr_fn, expected_ops, chained_pair):
        """_import_symbolic_expr produces the expected ops with correct operand wiring."""
        gni, loc, block, s0 = self._setup_importer()

        arg = self._make_sym_arg(expr_fn(s0))
        with gni._c, loc, mlir_ir.InsertionPoint(block):
            gni._import_symbolic_expr(loc, arg)

        op_names = [op.operation.name for op in block.operations]
        assert "torch.constant.int" in op_names, f"Expected constant.int, got: {op_names}"
        for expected in expected_ops:
            assert expected in op_names, f"Expected {expected}, got: {op_names}"

        # For simple expressions: verify the primary op uses block arg (s0)
        block_arg = block.arguments[0]
        if chained_pair is None:
            primary_op = expected_ops[0]
            matched = [op for op in block.operations if op.operation.name == primary_op]
            assert matched, f"No {primary_op} ops found, got: {op_names}"
            operands = list(matched[0].operation.operands)
            assert block_arg in operands, f"{primary_op} should use block arg (s0) as an operand"
        else:
            # For compound expressions: verify the second op consumes the first op's result
            first_name, second_name = chained_pair
            first_ops = [op for op in block.operations if op.operation.name == first_name]
            assert first_ops, f"No {first_name} ops found, got: {op_names}"
            first_result = first_ops[0].operation.results[0]

            second_ops = [op for op in block.operations if op.operation.name == second_name]
            assert second_ops, f"No {second_name} ops found, got: {op_names}"
            second_operands = list(second_ops[0].operation.operands)
            assert (
                first_result in second_operands
            ), f"{second_name} should consume {first_name} result (chained expression)"

    def test_caches_result(self):
        """_import_symbolic_expr caches the result in _sym_to_value."""
        gni, loc, block, s0 = self._setup_importer()

        arg = self._make_sym_arg(s0 + 5)
        with gni._c, loc, mlir_ir.InsertionPoint(block):
            result = gni._import_symbolic_expr(loc, arg)

        # The result should be cached under str(arg)
        assert str(arg) in gni._sym_to_value, f"Expected '{arg}' in _sym_to_value after import"
        assert gni._sym_to_value[str(arg)] == result

    def test_rejects_sym_float(self):
        """_import_symbolic_expr rejects SymFloat expressions (only int is supported)."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        gni, loc, block, s0 = self._setup_importer()

        shape_env = ShapeEnv()
        sym_float_arg = shape_env.create_unbacked_symfloat()
        with (
            gni._c,
            loc,
            mlir_ir.InsertionPoint(block),
            pytest.raises(AssertionError, match="only supports integer expressions"),
        ):
            gni._import_symbolic_expr(loc, sym_float_arg)


class TestSymTypeListDerivedExpr:
    """Tests that derived SymInt expressions in list operands are handled correctly."""

    def test_expand_with_derived_symbolic_dim(self):
        """torch.expand with a derived symbolic expression (batch + 1) in the size list."""

        class Model(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return x.sum(dim=1).unsqueeze(1).expand(batch, batch + 1)

        x = torch.randn(4, 5)
        gm, mlir_mod = _compile_and_inspect(Model(), (x,), dynamic_dims={0: {0}})

        mlir_ops = _get_mlir_op_names(mlir_mod)
        assert (
            "torch.prim.ListConstruct" in mlir_ops
        ), f"Expected torch.prim.ListConstruct (for sym dim list), got: {mlir_ops}"
        assert (
            "torch.aten.add.int" in mlir_ops
        ), f"Expected torch.aten.add.int (for derived batch+1 expr), got: {mlir_ops}"
