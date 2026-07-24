# ruff: noqa: N806

"""Unit tests for factory ``.out`` decompositions and FakeTensor overrides.

Covers ``full_like.out``, ``ones_like.out``, ``zeros_like.out``, ``new_full.out``,
``new_ones.out``, and ``new_zeros.out``.

Background: ``torch._subclasses.fake_impls.constructors()`` injects
``device=meta`` into every factory op's kwargs before re-dispatching. The
``.default`` variants declare ``device`` in their schema, so the injection is
harmless; the ``.out`` variants don't (they inherit device from ``out=``), so
Dynamo FakeTensor propagation fails with ``expected at most N argument(s) but
received N+1``. ``torch_neuronx.neuron_dynamo_backend.decompositions``
overrides the FakeTensor op-impl entry and registers a plain
``out.copy_(<functional_result>)`` decomposition on ``neuron_decompositions``
so AOT autograd can lower the ``.out`` call.
"""

import pytest
import torch
from torch._subclasses import fake_impls
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_neuronx.neuron_dynamo_backend import decompositions

aten = torch.ops.aten

# Table of (op, decomp_fn, sample_positional_args, expected_fn) tuples. The
# ``expected_fn`` receives the same positional args as the decomp and returns
# the reference tensor produced by eager PyTorch.
_FACTORY_OUT_OPS = [
    pytest.param(
        aten.full_like.out,
        decompositions._factory_out_decomp_full_like,
        lambda: (torch.randn(4, 5), 3.14),
        lambda self, fill_value: torch.full_like(self, fill_value),
        id="full_like.out",
    ),
    pytest.param(
        aten.ones_like.out,
        decompositions._factory_out_decomp_ones_like,
        lambda: (torch.randn(4, 5),),
        lambda self: torch.ones_like(self),
        id="ones_like.out",
    ),
    pytest.param(
        aten.zeros_like.out,
        decompositions._factory_out_decomp_zeros_like,
        lambda: (torch.randn(4, 5),),
        lambda self: torch.zeros_like(self),
        id="zeros_like.out",
    ),
    pytest.param(
        aten.new_full.out,
        decompositions._factory_out_decomp_new_full,
        lambda: (torch.randn(4, 5), [4, 5], -2.5),
        lambda self, size, fill_value: self.new_full(size, fill_value),
        id="new_full.out",
    ),
    pytest.param(
        aten.new_ones.out,
        decompositions._factory_out_decomp_new_ones,
        lambda: (torch.randn(4, 5), [4, 5]),
        lambda self, size: self.new_ones(size),
        id="new_ones.out",
    ),
    pytest.param(
        aten.new_zeros.out,
        decompositions._factory_out_decomp_new_zeros,
        lambda: (torch.randn(4, 5), [4, 5]),
        lambda self, size: self.new_zeros(size),
        id="new_zeros.out",
    ),
]


class TestFactoryOutRegistration:
    """Every factory ``.out`` overload is registered in both dispatch tables."""

    @pytest.mark.parametrize("op,decomp_fn,_build,_expected", _FACTORY_OUT_OPS)
    def test_decomp_registered(self, op, decomp_fn, _build, _expected):
        """The decomposition function is bound to ``neuron_decompositions``."""
        assert op in decompositions.neuron_decompositions
        assert decompositions.neuron_decompositions[op] is decomp_fn

    @pytest.mark.parametrize("op,_decomp_fn,_build,_expected", _FACTORY_OUT_OPS)
    def test_fake_impl_registered(self, op, _decomp_fn, _build, _expected):
        """The FakeTensor override is bound to ``op_implementations_dict`` and
        points at our routing wrapper (not the buggy ``constructors``)."""
        assert op in fake_impls.op_implementations_dict
        assert fake_impls.op_implementations_dict[op] is decompositions._factory_out_fake_impl


class TestFactoryOutDecompositionNumerics:
    """The decomp writes the expected value into ``out`` and returns it."""

    @pytest.mark.parametrize("_op,decomp_fn,build,expected_fn", _FACTORY_OUT_OPS)
    def test_matches_aten(self, _op, decomp_fn, build, expected_fn):
        torch.manual_seed(0)
        args = build()
        # A pre-populated out buffer with garbage values — the decomp must
        # overwrite it in place.
        out = torch.full_like(args[0], float("nan"))
        result = decomp_fn(*args, out=out)
        expected = expected_fn(*args)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("_op,decomp_fn,build,_expected", _FACTORY_OUT_OPS)
    def test_writes_into_out_and_returns_it(self, _op, decomp_fn, build, _expected):
        """``out`` is returned by identity (in-place semantics)."""
        args = build()
        out = torch.empty_like(args[0])
        r = decomp_fn(*args, out=out)
        assert r.data_ptr() == out.data_ptr()


class TestFactoryOutFakeTensorPropagation:
    """Dynamo-style FakeTensor propagation succeeds for every ``.out`` overload.

    Sidesteps the upstream ``fake_impls.constructors()`` bug that injects
    ``device=meta`` into a schema that doesn't accept it. Without the
    override in ``decompositions.py``, each call here would raise
    ``RuntimeError: aten::<op>() expected at most N argument(s) but received
    N+1 argument(s)``.
    """

    @pytest.mark.parametrize("op,_decomp_fn,build,_expected", _FACTORY_OUT_OPS)
    def test_fake_call_succeeds(self, op, _decomp_fn, build, _expected):
        args = build()
        out = torch.empty_like(args[0])
        with FakeTensorMode(allow_non_fake_inputs=True):
            r = op(*args, out=out)
        assert r.shape == args[0].shape
        assert r.dtype == args[0].dtype

    def test_torch_compile_traces_through(self):
        """``torch.compile(aot_eager)`` traces past the ``.out`` call end-to-end.

        Exercises the whole path: FakeTensor override -> AOT decomposition
        -> reference backend execution. If the fake override regresses,
        Dynamo trace fails; if the decomposition regresses, AOT lowering
        raises. This is the single test that catches either failure mode.
        """
        torch.manual_seed(0)

        def fn(x):
            out = torch.empty_like(x)
            return torch.ops.aten.ones_like.out(x, out=out)

        x = torch.randn(4, 5)
        expected = torch.ones_like(x)
        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), expected)
