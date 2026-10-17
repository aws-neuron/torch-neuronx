"""Mechanism tests for the Neuron-scoped graph breaks. No Neuron device needed.

Dynamo makes the graph-break decision during tracing, so a plain CPU backend is
enough to exercise it. ``_is_neuron_compile`` reads the active backend name from
the ``InstructionTranslator``, which only reports ``"neuron"`` for a real
``backend="neuron"`` compile, so these tests patch it. The real backend
detection is covered in ``e2e/test_graph_break_ops_e2e.py``.

Every test installs its own registry through the ``registry`` fixture. No test
depends on which ops the production block lists hold, because both lists change
over time: the collected list follows the installed torch and torch-mlir, and
the configured list grows as ops gain a limit. The collector tests below assert
the collection *rules* over the whole result instead of naming an op.
"""

import pytest
import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from torch_neuronx.neuron_dynamo_backend import graph_break_ops as gb
from torch_neuronx.neuron_dynamo_backend import graph_break_target_ops as targets

LIMIT = gb.SORT_SIZE_LIMIT

# A stand-in op for the unconditional (collected-list) behaviour. Any op works;
# the registry below decides the behaviour, not the production lists.
_UNCONDITIONAL_FN = torch.matrix_exp
_UNCONDITIONAL_METHOD = "matrix_exp"


@pytest.fixture(scope="module")
def collected():
    return targets.collect_unsupported_ops()


@pytest.fixture(autouse=True)
def _reset_dynamo():
    # ``torch_neuronx`` installs the hooks on import. Ask again so a failure to
    # install cannot make a "no break expected" test pass for the wrong reason.
    gb._apply_dynamo_graph_break_patch()
    assert gb._patch_applied
    dynamo.reset()
    counters.clear()
    yield
    dynamo.reset()
    counters.clear()


@pytest.fixture
def registry(monkeypatch):
    """Install a test-owned registry in place of the production block lists."""
    conditional = {
        torch.sort: gb.BlockedOp("sort", "too long", gb._sort_dim_supported),
        torch.argsort: gb.BlockedOp("argsort", "too long", gb._sort_dim_supported),
    }
    by_fn = dict(conditional)
    by_fn[_UNCONDITIONAL_FN] = gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering")
    by_method = {op.name: op for op in conditional.values()}
    by_method[_UNCONDITIONAL_METHOD] = gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering")
    monkeypatch.setattr(gb, "_BY_FN", by_fn)
    monkeypatch.setattr(gb, "_BY_METHOD", by_method)


@pytest.fixture
def on_neuron(monkeypatch, registry):
    """Make both hooks believe the traced frame is bound for backend="neuron"."""
    monkeypatch.setattr(gb, "_is_neuron_compile", lambda tx=None: True)


def _spy_backend():
    def backend(gm, example_inputs):
        backend.subgraphs.append(gm)
        return gm.forward

    backend.subgraphs = []
    return backend


def _breaks() -> int:
    return sum(counters.get("graph_break", {}).values())


def _run(fn, *args):
    backend = _spy_backend()
    counters.clear()
    out = torch.compile(fn, backend=backend)(*args)
    return out, backend


def _in_graph(backend, needle: str) -> bool:
    return any(needle in str(node.target) for gm in backend.subgraphs for node in gm.graph.nodes)


class TestConditionalBreaks:
    """A conditional entry breaks only when its predicate reports False."""

    def test_small_sort_stays_in_graph(self, on_neuron):
        x = torch.randn(2, 8)
        out, backend = _run(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, torch.sort(x)[0] * 2.0)

    def test_large_sort_breaks_and_matches_eager(self, on_neuron):
        x = torch.randn(2, LIMIT + 1)
        out, backend = _run(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() >= 1
        assert not _in_graph(backend, "sort")
        torch.testing.assert_close(out, torch.sort(x)[0] * 2.0)

    def test_large_argsort_breaks(self, on_neuron):
        x = torch.randn(2, LIMIT + 1)
        out, _ = _run(lambda t: torch.argsort(t) + 1, x)

        assert _breaks() >= 1
        torch.testing.assert_close(out, torch.argsort(x) + 1)

    def test_sort_checks_the_requested_dim(self, on_neuron):
        """A long dimension that the call does not sort must not break."""
        x = torch.randn(LIMIT + 1, 4)

        out, backend = _run(lambda t: torch.sort(t, dim=1)[0] * 2.0, x)
        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, torch.sort(x, dim=1)[0] * 2.0)

        counters.clear()
        out, _ = _run(lambda t: torch.sort(t, dim=0)[0] * 2.0, x)
        assert _breaks() >= 1
        torch.testing.assert_close(out, torch.sort(x, dim=0)[0] * 2.0)

    def test_sort_accepts_a_positional_dim(self, on_neuron):
        x = torch.randn(LIMIT + 1, 4)
        out, _ = _run(lambda t: torch.sort(t, 0)[0] * 2.0, x)

        assert _breaks() >= 1
        torch.testing.assert_close(out, torch.sort(x, 0)[0] * 2.0)

    def test_break_reason_names_the_op(self, on_neuron):
        x = torch.randn(2, LIMIT + 1)
        _run(lambda t: torch.sort(t)[0], x)

        assert "torch.sort" in " ".join(counters.get("graph_break", {}))


class TestUnconditionalBreaks:
    """An entry with no predicate breaks every call. The collected list is this."""

    def test_every_call_breaks(self, on_neuron):
        x = torch.randn(5, 5)
        out, backend = _run(lambda t: _UNCONDITIONAL_FN(t) * 2.0, x)

        assert _breaks() >= 1
        assert not _in_graph(backend, _UNCONDITIONAL_METHOD)
        torch.testing.assert_close(out, _UNCONDITIONAL_FN(x) * 2.0)

    def test_method_form_breaks(self, on_neuron):
        x = torch.randn(5, 5)
        out, backend = _run(lambda t: getattr(t, _UNCONDITIONAL_METHOD)() * 2.0, x)

        assert _breaks() >= 1
        assert not _in_graph(backend, _UNCONDITIONAL_METHOD)
        torch.testing.assert_close(out, _UNCONDITIONAL_FN(x) * 2.0)

    def test_unregistered_op_stays_in_graph(self, on_neuron):
        x = torch.randn(5, 5)
        out, backend = _run(lambda t: torch.linalg.inv(t) * 2.0, x)

        assert _breaks() == 0
        torch.testing.assert_close(out, torch.linalg.inv(x) * 2.0)


class TestUnconditionalLookupHook:
    """The ``trace_rules.lookup`` patch catches C-extension built-ins that Dynamo
    re-classifies through the frame-restart path and would otherwise bypass the
    two call-time hooks."""

    def test_returns_skip_variable_when_neuron_active(self, on_neuron, monkeypatch):
        from torch._dynamo import trace_rules
        from torch._dynamo.variables.functions import SkipFunctionVariable

        monkeypatch.setitem(
            gb.UNCONDITIONAL_TARGET_OPS,
            _UNCONDITIONAL_FN,
            gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering"),
        )
        assert trace_rules.lookup(_UNCONDITIONAL_FN) is SkipFunctionVariable

    def test_falls_through_under_non_neuron_backend(self, registry, monkeypatch):
        from torch._dynamo import trace_rules
        from torch._dynamo.variables.functions import SkipFunctionVariable

        monkeypatch.setattr(gb, "_is_neuron_compile", lambda tx=None: False)
        monkeypatch.setitem(
            gb.UNCONDITIONAL_TARGET_OPS,
            _UNCONDITIONAL_FN,
            gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering"),
        )
        assert trace_rules.lookup(_UNCONDITIONAL_FN) is not SkipFunctionVariable

    def test_falls_through_when_no_tx_active(self, registry, monkeypatch):
        """A bare call outside ``torch.compile`` has no active ``InstructionTranslator``,
        so ``_is_neuron_compile`` returns False and the hook falls through."""
        from torch._dynamo import trace_rules
        from torch._dynamo.variables.functions import SkipFunctionVariable

        monkeypatch.setitem(
            gb.UNCONDITIONAL_TARGET_OPS,
            _UNCONDITIONAL_FN,
            gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering"),
        )
        assert trace_rules.lookup(_UNCONDITIONAL_FN) is not SkipFunctionVariable


class TestTensorMethodBreaks:
    """Dynamo routes a method through ``call_method``, which needs its own hook."""

    def test_small_sort_method_stays_in_graph(self, on_neuron):
        x = torch.randn(2, 8)
        out, backend = _run(lambda t: t.sort()[0] * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, x.sort()[0] * 2.0)

    def test_large_sort_method_breaks(self, on_neuron):
        x = torch.randn(2, LIMIT + 1)
        out, _ = _run(lambda t: t.sort()[0] * 2.0, x)

        assert _breaks() >= 1
        torch.testing.assert_close(out, x.sort()[0] * 2.0)


class TestBackendScoping:
    """Nothing breaks unless the traced frame is bound for backend="neuron"."""

    def test_no_break_for_a_non_neuron_backend(self, registry):
        x = torch.randn(2, LIMIT + 1)
        out, backend = _run(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, torch.sort(x)[0] * 2.0)

    def test_no_method_break_for_a_non_neuron_backend(self, registry):
        x = torch.randn(2, LIMIT + 1)
        out, backend = _run(lambda t: t.sort()[0] * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, x.sort()[0] * 2.0)

    def test_no_unconditional_break_for_a_non_neuron_backend(self, registry):
        x = torch.randn(5, 5)
        out, backend = _run(lambda t: _UNCONDITIONAL_FN(t) * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, _UNCONDITIONAL_METHOD)


class TestDynamicShapes:
    def test_symbolic_sort_dim_does_not_break(self, on_neuron):
        """An unresolved symbolic size leaves the op in the graph and adds no guard."""
        x = torch.randn(2, LIMIT + 1)
        dynamo.mark_dynamic(x, 1)

        out, backend = _run(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() == 0
        assert _in_graph(backend, "sort")
        torch.testing.assert_close(out, torch.sort(x)[0] * 2.0)


class TestShouldBreak:
    def test_entry_without_a_predicate_always_breaks(self):
        assert gb._should_break(gb.BlockedOp("any_op", "reason"), [], {})

    def test_predicate_failure_does_not_break(self):
        """A predicate that cannot judge the call must not break the graph."""

        def supported(*_args, **_kwargs):
            raise RuntimeError("unexpected signature")

        assert not gb._should_break(gb.BlockedOp("any_op", "reason", supported), [], {})

    def test_above_limit_only_for_a_known_large_int(self):
        assert gb._above_limit(LIMIT + 1, LIMIT)
        assert not gb._above_limit(LIMIT, LIMIT)
        assert not gb._above_limit(None, LIMIT)

    def test_sort_predicate_ignores_an_unresolved_argument(self):
        assert gb._sort_dim_supported(None, -1)
        assert gb._sort_dim_supported(torch.randn(2, LIMIT + 1), None)

    def test_sort_predicate_bound_is_inclusive(self):
        """A sorted dimension of exactly SORT_SIZE_LIMIT compiles."""
        assert gb._sort_dim_supported(torch.randn(2, LIMIT), -1)
        assert not gb._sort_dim_supported(torch.randn(2, LIMIT + 1), -1)


class TestRegistryMerge:
    """A configured limit must never be masked by an unconditional entry."""

    def test_conditional_entry_wins_over_the_collected_list(self, monkeypatch):
        monkeypatch.setattr(gb, "DEFAULT_TARGET_OPS", frozenset({torch.sort}))
        monkeypatch.setattr(gb, "DEFAULT_TARGET_METHODS", frozenset({"sort"}))

        by_fn, by_method = gb._build_registry()

        assert by_fn[torch.sort].supported is gb._sort_dim_supported
        assert by_method["sort"].supported is gb._sort_dim_supported

    def test_collected_entries_get_no_predicate(self, monkeypatch):
        monkeypatch.setattr(gb, "DEFAULT_TARGET_OPS", frozenset({torch.matrix_exp}))
        monkeypatch.setattr(gb, "DEFAULT_TARGET_METHODS", frozenset({"matrix_exp"}))

        by_fn, by_method = gb._build_registry()

        assert by_fn[torch.matrix_exp].supported is None
        assert by_method["matrix_exp"].supported is None

    def test_every_configured_op_is_in_the_registry(self):
        for fn, op in gb.CONDITIONAL_TARGET_OPS.items():
            assert gb._BY_FN[fn] is op
            assert gb._BY_METHOD[op.name] is op


class TestCollectionRules:
    """Assert the collection rules over the whole result, not named ops.

    Both block lists follow the installed torch, torch-mlir, and Neuron
    decompositions, so naming an op here would make the test a snapshot of the
    current dependency versions.
    """

    def test_collection_returns_callables_and_names(self, collected):
        functions, methods = collected

        assert functions, "expected at least one uncovered free function"
        assert all(callable(fn) for fn in functions)
        assert all(isinstance(name, str) for name in methods)

    def test_no_collected_op_is_decomposed_above_the_backend(self, collected):
        """A CompositeImplicitAutograd op never reaches the backend."""
        functions, methods = collected

        for base in {targets._aten_base_name(fn) for fn in functions} - {None}:
            assert not targets._is_decomposed_above_backend(base), base
        for name in methods:
            assert not targets._is_decomposed_above_backend(name), name

    def test_every_collected_op_exists_in_the_dispatcher(self, collected):
        """An op the dispatcher does not know can never need a Neuron lowering."""
        functions, _ = collected

        for base in {targets._aten_base_name(fn) for fn in functions} - {None}:
            assert targets._overload_keys(base), base

    def test_every_method_entry_returns_a_tensor(self, collected):
        """Dynamo folds a metadata query to a constant, so a break would be wrong."""
        _, methods = collected

        for name in methods:
            assert hasattr(torch.Tensor, name), name
            assert targets._returns_tensor(name), name

    def test_no_covered_op_is_collected(self, collected):
        """An op that torch-mlir or a Neuron decomposition covers must be absent."""
        functions, _ = collected
        covered = targets._torch_mlir_ops() | targets._decomposed_ops()
        covered_bases = {name.split(".", 1)[0] for name in covered}

        for base in {targets._aten_base_name(fn) for fn in functions} - {None}:
            assert base not in covered_bases, base
