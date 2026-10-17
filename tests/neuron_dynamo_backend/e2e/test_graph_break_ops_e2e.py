"""E2E tests for the Neuron-scoped graph breaks through ``torch.compile``.

These run the real ``backend="neuron"`` path, so they also cover the backend
detection that the CPU tests in ``../test_graph_break_ops.py`` patch out.

Every test installs its own registry through the ``registry`` fixture, so no test
depends on which ops the production block lists hold.

Runs on trn2. Requires the Neuron runtime.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.exc import Unsupported
from torch._dynamo.utils import counters

from torch_neuronx.neuron_dynamo_backend import graph_break_ops as gb

LIMIT = gb.SORT_SIZE_LIMIT

# Neuron lowers this op, so a break here isolates the mechanism.
_UNCONDITIONAL_FN = torch.sigmoid
_UNCONDITIONAL_METHOD = "sigmoid"


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()
    counters.clear()


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Install a test-owned registry in place of the production block lists."""
    conditional = {
        torch.sort: gb.BlockedOp("sort", "too long", gb._sort_dim_supported),
        torch.argsort: gb.BlockedOp("argsort", "too long", gb._sort_dim_supported),
    }
    by_fn = dict(conditional)
    by_method = {op.name: op for op in conditional.values()}
    monkeypatch.setattr(gb, "_BY_FN", by_fn)
    monkeypatch.setattr(gb, "_BY_METHOD", by_method)
    return by_fn, by_method


@pytest.fixture
def block_unconditionally(registry):
    """Add an entry with no predicate, the way the collected list behaves."""
    by_fn, by_method = registry
    op = gb.BlockedOp(_UNCONDITIONAL_METHOD, "no lowering")
    by_fn[_UNCONDITIONAL_FN] = op
    by_method[_UNCONDITIONAL_METHOD] = op


def _breaks() -> int:
    return sum(counters.get("graph_break", {}).values())


def _randn(*shape):
    """Seed on the CPU. Device RNG is not part of what these tests check."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0))


def _compile_on_neuron(fn, x):
    with torch.no_grad():
        reference = fn(x)
        counters.clear()
        out = torch.compile(fn, backend="neuron")(x.to("neuron"))
    return out, reference


class TestConditionalBreaksE2E:
    def test_large_sort_breaks_and_matches_cpu(self):
        x = _randn(2, LIMIT + 1)

        out, reference = _compile_on_neuron(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() >= 1
        assert "torch.sort" in " ".join(counters.get("graph_break", {}))
        torch.testing.assert_close(out.cpu(), reference)

    def test_small_sort_stays_in_graph(self):
        x = _randn(2, 128)

        out, reference = _compile_on_neuron(lambda t: torch.sort(t)[0] * 2.0, x)

        assert _breaks() == 0
        torch.testing.assert_close(out.cpu(), reference)

    def test_large_argsort_breaks_and_matches_cpu(self):
        x = _randn(2, LIMIT + 1)

        out, reference = _compile_on_neuron(lambda t: torch.argsort(t) + 1, x)

        assert _breaks() >= 1
        torch.testing.assert_close(out.cpu(), reference)

    def test_fullgraph_true_raises_on_a_blocked_call(self):
        x = _randn(2, LIMIT + 1)
        compiled = torch.compile(lambda t: torch.sort(t)[0], backend="neuron", fullgraph=True)

        with torch.no_grad(), pytest.raises(Unsupported):
            compiled(x.to("neuron"))


class TestUnconditionalBreaksE2E:
    """The collected block list is entries with no predicate."""

    def test_every_call_breaks_and_matches_cpu(self, block_unconditionally):
        x = _randn(4, 32)

        out, reference = _compile_on_neuron(lambda t: _UNCONDITIONAL_FN(t) * 2.0, x)

        assert _breaks() >= 1
        assert _UNCONDITIONAL_METHOD in " ".join(counters.get("graph_break", {}))
        torch.testing.assert_close(out.cpu(), reference)

    def test_method_form_breaks_and_matches_cpu(self, block_unconditionally):
        x = _randn(4, 32)

        out, reference = _compile_on_neuron(lambda t: getattr(t, _UNCONDITIONAL_METHOD)() * 2.0, x)

        assert _breaks() >= 1
        torch.testing.assert_close(out.cpu(), reference)

    def test_same_op_stays_in_graph_when_not_registered(self):
        """Without the registry entry the op compiles, so the break is the only variable."""
        x = _randn(4, 32)

        out, reference = _compile_on_neuron(lambda t: _UNCONDITIONAL_FN(t) * 2.0, x)

        assert _breaks() == 0
        torch.testing.assert_close(out.cpu(), reference)


class TestMethodBreaksE2E:
    """Dynamo routes ``x.sort()`` through a different hook than ``torch.sort``."""

    def test_large_sort_method_breaks_and_matches_cpu(self):
        x = _randn(2, LIMIT + 1)

        out, reference = _compile_on_neuron(lambda t: t.sort()[0] * 2.0, x)

        assert _breaks() >= 1
        assert "Tensor.sort" in " ".join(counters.get("graph_break", {}))
        torch.testing.assert_close(out.cpu(), reference)


class TestBackendScopingE2E:
    def test_eager_backend_does_not_break(self):
        """The block list must apply to backend="neuron" only."""
        x = _randn(2, LIMIT + 1)

        def fn(t):
            return torch.sort(t)[0] * 2.0

        with torch.no_grad():
            counters.clear()
            out = torch.compile(fn, backend="eager")(x)

        assert _breaks() == 0
        torch.testing.assert_close(out, fn(x))

    def test_eager_backend_does_not_break_the_method_form(self):
        x = _randn(2, LIMIT + 1)

        def fn(t):
            return t.sort()[0] * 2.0

        with torch.no_grad():
            counters.clear()
            out = torch.compile(fn, backend="eager")(x)

        assert _breaks() == 0
        torch.testing.assert_close(out, fn(x))

    def test_eager_backend_does_not_break_an_unconditional_entry(self, block_unconditionally):
        x = _randn(4, 32)

        def fn(t):
            return _UNCONDITIONAL_FN(t) * 2.0

        with torch.no_grad():
            counters.clear()
            out = torch.compile(fn, backend="eager")(x)

        assert _breaks() == 0
        torch.testing.assert_close(out, fn(x))
