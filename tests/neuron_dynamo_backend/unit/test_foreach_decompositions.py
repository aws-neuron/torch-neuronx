import pytest
import torch
from torch._subclasses import fake_impls
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_neuronx.neuron_dynamo_backend import decompositions

aten = torch.ops.aten


def _build_functional_inputs(n=4, shape=(8, 8)):
    """Build inputs for the functional ``.Tensor`` variants."""
    torch.manual_seed(0)
    self_list = [torch.randn(*shape) for _ in range(n)]
    t1_list = [torch.randn(*shape) for _ in range(n)]
    t2_list = [torch.randn(*shape) + 2.0 for _ in range(n)]  # avoid div-by-zero
    scalars = torch.randn(n)
    return self_list, t1_list, t2_list, scalars


def _clone_list(tensors):
    return [t.clone() for t in tensors]


# ─── functional (returns new list) ────────────────────────────────────────

_ADDC_FUNCTIONAL_OPS = [
    pytest.param(
        aten._foreach_addcdiv.Tensor,
        decompositions._foreach_addcdiv_tensor_decomp,
        id="_foreach_addcdiv.Tensor",
    ),
    pytest.param(
        aten._foreach_addcmul.Tensor,
        decompositions._foreach_addcmul_tensor_decomp,
        id="_foreach_addcmul.Tensor",
    ),
]

# ─── .Tensor_out (writes into out=; unit return) ──────────────────────────

_OUT_OPS = [
    pytest.param(
        aten._foreach_addcdiv.Tensor_out,
        decompositions._foreach_addcdiv_tensor_out_decomp,
        id="_foreach_addcdiv.Tensor_out",
    ),
    pytest.param(
        aten._foreach_addcmul.Tensor_out,
        decompositions._foreach_addcmul_tensor_out_decomp,
        id="_foreach_addcmul.Tensor_out",
    ),
]

# ─── _.Tensor (in-place; unit return) ─────────────────────────────────────

_INPLACE_OPS = [
    pytest.param(
        aten._foreach_addcdiv_.Tensor,
        decompositions._foreach_addcdiv_inplace_tensor_decomp,
        id="_foreach_addcdiv_.Tensor",
    ),
    pytest.param(
        aten._foreach_addcmul_.Tensor,
        decompositions._foreach_addcmul_inplace_tensor_decomp,
        id="_foreach_addcmul_.Tensor",
    ),
]

_ADDC_ALL_OPS = _ADDC_FUNCTIONAL_OPS + _OUT_OPS + _INPLACE_OPS


class TestForeachAddcRegistration:
    """Every foreach ``.Tensor*`` overload is registered in both tables."""

    @pytest.mark.parametrize("op,decomp_fn", _ADDC_ALL_OPS)
    def test_decomp_registered(self, op, decomp_fn):
        """The decomposition function is bound to ``neuron_decompositions``."""
        assert op in decompositions.neuron_decompositions
        assert decompositions.neuron_decompositions[op] is decomp_fn

    @pytest.mark.parametrize("op,_decomp_fn", _ADDC_ALL_OPS)
    def test_fake_impl_registered(self, op, _decomp_fn):
        """The FakeTensor override is bound to ``op_implementations_dict`` and
        points at our shape-only synthesizer (not the buggy meta path)."""
        assert op in fake_impls.op_implementations_dict
        assert (
            fake_impls.op_implementations_dict[op] is decompositions._foreach_addc_tensor_fake_impl
        )


class TestForeachAddcDecompositionNumerics:
    """The decomposition math matches eager for every variant."""

    @pytest.mark.parametrize("op,decomp_fn", _ADDC_FUNCTIONAL_OPS)
    def test_functional_matches_aten(self, op, decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        result = decomp_fn(self_list, t1_list, t2_list, scalars)
        expected = op(self_list, t1_list, t2_list, scalars)
        for r, e in zip(result, expected, strict=True):
            torch.testing.assert_close(r, e)

    @pytest.mark.parametrize("op,decomp_fn", _OUT_OPS)
    def test_out_variant_matches_aten_and_writes_out(self, op, decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        out = [torch.full_like(t, float("nan")) for t in self_list]
        # Snapshot pointers before to confirm no reallocation.
        out_ptrs = [t.data_ptr() for t in out]
        decomp_fn(self_list, t1_list, t2_list, scalars, out=out)
        # ``out`` was written in place — pointers unchanged.
        assert [t.data_ptr() for t in out] == out_ptrs
        # Compare against the functional variant of the same op (``.Tensor``
        # sibling). The aten ``.Tensor_out`` schema returns ``()`` so we can't
        # collect an expected value from it directly.
        functional_op = op.overloadpacket.Tensor
        expected = functional_op(self_list, t1_list, t2_list, scalars)
        for r, e in zip(out, expected, strict=True):
            torch.testing.assert_close(r, e)

    @pytest.mark.parametrize("op,decomp_fn", _INPLACE_OPS)
    def test_inplace_variant_matches_aten_and_mutates_self(self, op, decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        # Reference: eager op mutates ``self_ref`` in place.
        self_ref = _clone_list(self_list)
        op(self_ref, t1_list, t2_list, scalars)
        # Decomp: mutates ``self_dec`` in place.
        self_dec = _clone_list(self_list)
        self_dec_ptrs = [t.data_ptr() for t in self_dec]
        decomp_fn(self_dec, t1_list, t2_list, scalars)
        # Pointers unchanged (true in-place mutation).
        assert [t.data_ptr() for t in self_dec] == self_dec_ptrs
        for r, e in zip(self_dec, self_ref, strict=True):
            torch.testing.assert_close(r, e)


class TestForeachAddcFakeTensorPropagation:
    """Dynamo-style FakeTensor propagation succeeds for every overload.

    Without the override in ``decompositions.py``, each call here would
    raise ``RuntimeError: Expected scalars to be on CPU, got meta instead``.
    """

    @pytest.mark.parametrize("op,_decomp_fn", _ADDC_FUNCTIONAL_OPS)
    def test_fake_call_functional(self, op, _decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        with FakeTensorMode(allow_non_fake_inputs=True):
            r = op(self_list, t1_list, t2_list, scalars)
        assert isinstance(r, list)
        assert len(r) == len(self_list)
        for out_t, self_t in zip(r, self_list, strict=True):
            assert out_t.shape == self_t.shape
            assert out_t.dtype == self_t.dtype

    @pytest.mark.parametrize("op,_decomp_fn", _OUT_OPS)
    def test_fake_call_out(self, op, _decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        out = [torch.empty_like(t) for t in self_list]
        with FakeTensorMode(allow_non_fake_inputs=True):
            r = op(self_list, t1_list, t2_list, scalars, out=out)
        # The ``.Tensor_out`` schema declares a unit return ``()``.
        assert r is None or r == ()

    @pytest.mark.parametrize("op,_decomp_fn", _INPLACE_OPS)
    def test_fake_call_inplace(self, op, _decomp_fn):
        self_list, t1_list, t2_list, scalars = _build_functional_inputs()
        with FakeTensorMode(allow_non_fake_inputs=True):
            r = op(self_list, t1_list, t2_list, scalars)
        assert r is None or r == ()

    def test_torch_compile_traces_through_functional(self):
        """``torch.compile(aot_eager)`` traces the whole path end-to-end
        for the functional variant. Exercises the FakeTensor override AND
        the decomposition; a regression in either fails this test.
        """
        torch.manual_seed(0)

        def fn(a0, a1, b0, b1, c0, c1, s):
            return torch.ops.aten._foreach_addcdiv.Tensor([a0, a1], [b0, b1], [c0, c1], s)

        args = (
            torch.randn(4, 4),
            torch.randn(4, 4),
            torch.randn(4, 4),
            torch.randn(4, 4),
            torch.randn(4, 4) + 2.0,
            torch.randn(4, 4) + 2.0,
            torch.tensor([0.5, 0.7]),
        )
        expected = fn(*args)
        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)(*args)
        for r, e in zip(compiled, expected, strict=True):
            torch.testing.assert_close(r, e)


def _build_tensors(n=4, shape=(6, 6)):
    torch.manual_seed(0)
    return [torch.randn(*shape) for _ in range(n)]


_SCALARS = [0.5, 1.0, -0.2, 3.0]


# ─── Functional ``.ScalarList`` variants ──────────────────────────────────

_SCALARLIST_FUNCTIONAL_OPS = [
    pytest.param(
        aten._foreach_clamp_max.ScalarList,
        decompositions._foreach_clamp_max_scalarlist_decomp,
        lambda ts, ss: [t.clamp(max=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_max.ScalarList",
    ),
    pytest.param(
        aten._foreach_clamp_min.ScalarList,
        decompositions._foreach_clamp_min_scalarlist_decomp,
        lambda ts, ss: [t.clamp(min=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_min.ScalarList",
    ),
    pytest.param(
        aten._foreach_maximum.ScalarList,
        decompositions._foreach_maximum_scalarlist_decomp,
        lambda ts, ss: [
            torch.maximum(t, torch.full_like(t, s)) for t, s in zip(ts, ss, strict=True)
        ],
        id="_foreach_maximum.ScalarList",
    ),
    pytest.param(
        aten._foreach_minimum.ScalarList,
        decompositions._foreach_minimum_scalarlist_decomp,
        lambda ts, ss: [
            torch.minimum(t, torch.full_like(t, s)) for t, s in zip(ts, ss, strict=True)
        ],
        id="_foreach_minimum.ScalarList",
    ),
    pytest.param(
        aten._foreach_pow.ScalarList,
        decompositions._foreach_pow_scalarlist_decomp,
        lambda ts, ss: [torch.pow(t, s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_pow.ScalarList",
    ),
]

# ─── ``.ScalarList_out`` variants (unary math, writes into out=[...]) ────

_OUT_UNARY_OPS = [
    pytest.param(
        aten._foreach_clamp_max.ScalarList_out,
        decompositions._foreach_clamp_max_scalarlist_out_decomp,
        lambda ts, ss: [t.clamp(max=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_max.ScalarList_out",
    ),
    pytest.param(
        aten._foreach_clamp_min.ScalarList_out,
        decompositions._foreach_clamp_min_scalarlist_out_decomp,
        lambda ts, ss: [t.clamp(min=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_min.ScalarList_out",
    ),
    pytest.param(
        aten._foreach_maximum.ScalarList_out,
        decompositions._foreach_maximum_scalarlist_out_decomp,
        lambda ts, ss: [
            torch.maximum(t, torch.full_like(t, s)) for t, s in zip(ts, ss, strict=True)
        ],
        id="_foreach_maximum.ScalarList_out",
    ),
    pytest.param(
        aten._foreach_minimum.ScalarList_out,
        decompositions._foreach_minimum_scalarlist_out_decomp,
        lambda ts, ss: [
            torch.minimum(t, torch.full_like(t, s)) for t, s in zip(ts, ss, strict=True)
        ],
        id="_foreach_minimum.ScalarList_out",
    ),
    pytest.param(
        aten._foreach_pow.ScalarList_out,
        decompositions._foreach_pow_scalarlist_out_decomp,
        lambda ts, ss: [torch.pow(t, s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_pow.ScalarList_out",
    ),
]

# ─── ``_.ScalarList`` in-place variants (mutate self) ────────────────────

_INPLACE_UNARY_OPS = [
    pytest.param(
        aten._foreach_clamp_max_.ScalarList,
        decompositions._foreach_clamp_max_inplace_scalarlist_decomp,
        lambda ts, ss: [t.clamp(max=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_max_.ScalarList",
    ),
    pytest.param(
        aten._foreach_clamp_min_.ScalarList,
        decompositions._foreach_clamp_min_inplace_scalarlist_decomp,
        lambda ts, ss: [t.clamp(min=s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_clamp_min_.ScalarList",
    ),
    pytest.param(
        aten._foreach_pow_.ScalarList,
        decompositions._foreach_pow_inplace_scalarlist_decomp,
        lambda ts, ss: [torch.pow(t, s) for t, s in zip(ts, ss, strict=True)],
        id="_foreach_pow_.ScalarList",
    ),
]

# ─── lerp family (3-arg: self, tensors1, weights) ────────────────────────

_LERP_OPS = [
    pytest.param(
        aten._foreach_lerp.ScalarList,
        decompositions._foreach_lerp_scalarlist_decomp,
        "functional",
        id="_foreach_lerp.ScalarList",
    ),
    pytest.param(
        aten._foreach_lerp.ScalarList_out,
        decompositions._foreach_lerp_scalarlist_out_decomp,
        "out",
        id="_foreach_lerp.ScalarList_out",
    ),
    pytest.param(
        aten._foreach_lerp_.ScalarList,
        decompositions._foreach_lerp_inplace_scalarlist_decomp,
        "inplace",
        id="_foreach_lerp_.ScalarList",
    ),
]

# ─── Registration ────────────────────────────────────────────────────────


class TestForeachScalarListRegistration:
    """Every registered ScalarList decomp is bound to ``neuron_decompositions``."""

    @pytest.mark.parametrize(
        "op,decomp_fn,_expected", _SCALARLIST_FUNCTIONAL_OPS + _OUT_UNARY_OPS + _INPLACE_UNARY_OPS
    )
    def test_unary_decomp_registered(self, op, decomp_fn, _expected):
        assert op in decompositions.neuron_decompositions
        assert decompositions.neuron_decompositions[op] is decomp_fn

    @pytest.mark.parametrize("op,decomp_fn,_kind", _LERP_OPS)
    def test_lerp_decomp_registered(self, op, decomp_fn, _kind):
        assert op in decompositions.neuron_decompositions
        assert decompositions.neuron_decompositions[op] is decomp_fn


# ─── Numerics ────────────────────────────────────────────────────────────


class TestForeachScalarListDecompositionNumerics:
    """Each decomp's math matches eager for the corresponding aten op."""

    @pytest.mark.parametrize("op,decomp_fn,expected_fn", _SCALARLIST_FUNCTIONAL_OPS)
    def test_functional_matches_aten(self, op, decomp_fn, expected_fn):
        tensors = _build_tensors()
        result = decomp_fn(tensors, _SCALARS)
        expected = op(tensors, _SCALARS)
        for r, e in zip(result, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)
        # Sanity vs. hand-rolled reference (catches upstream aten semantic drift).
        for r, e in zip(result, expected_fn(tensors, _SCALARS), strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    @pytest.mark.parametrize("op,decomp_fn,expected_fn", _OUT_UNARY_OPS)
    def test_out_matches_aten_and_writes_out(self, op, decomp_fn, expected_fn):
        tensors = _build_tensors()
        out = [torch.full_like(t, float("nan")) for t in tensors]
        out_ptrs = [t.data_ptr() for t in out]
        decomp_fn(tensors, _SCALARS, out=out)
        # ``out`` was written in place — pointers unchanged.
        assert [t.data_ptr() for t in out] == out_ptrs
        # Values match the hand-rolled reference (the aten ``.ScalarList_out``
        # schema returns ``()``, so we can't compare against its return value).
        for r, e in zip(out, expected_fn(tensors, _SCALARS), strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    @pytest.mark.parametrize("op,decomp_fn,expected_fn", _INPLACE_UNARY_OPS)
    def test_inplace_matches_aten_and_mutates_self(self, op, decomp_fn, expected_fn):
        # Eager reference: mutate self_ref in place.
        self_ref = _build_tensors()
        op(self_ref, _SCALARS)
        # Decomp: mutate self_dec in place.
        self_dec = _build_tensors()
        self_dec_ptrs = [t.data_ptr() for t in self_dec]
        decomp_fn(self_dec, _SCALARS)
        assert [t.data_ptr() for t in self_dec] == self_dec_ptrs
        for r, e in zip(self_dec, self_ref, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_lerp_functional_matches_aten(self):
        starts = _build_tensors()
        ends = _build_tensors(n=4, shape=(6, 6))
        # Rebuild ends with a different seed so start != end (otherwise lerp is
        # trivially self).
        torch.manual_seed(1)
        ends = [torch.randn(6, 6) for _ in range(4)]
        result = decompositions._foreach_lerp_scalarlist_decomp(starts, ends, _SCALARS)
        expected = aten._foreach_lerp.ScalarList(starts, ends, _SCALARS)
        for r, e in zip(result, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_lerp_out_matches_aten_and_writes_out(self):
        starts = _build_tensors()
        torch.manual_seed(1)
        ends = [torch.randn(6, 6) for _ in range(4)]
        out = [torch.full_like(t, float("nan")) for t in starts]
        out_ptrs = [t.data_ptr() for t in out]
        decompositions._foreach_lerp_scalarlist_out_decomp(starts, ends, _SCALARS, out=out)
        assert [t.data_ptr() for t in out] == out_ptrs
        expected = aten._foreach_lerp.ScalarList(starts, ends, _SCALARS)
        for r, e in zip(out, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_lerp_inplace_matches_aten_and_mutates_self(self):
        starts_ref = _build_tensors()
        torch.manual_seed(1)
        ends = [torch.randn(6, 6) for _ in range(4)]
        aten._foreach_lerp_.ScalarList(starts_ref, ends, _SCALARS)

        starts_dec = _build_tensors()
        starts_dec_ptrs = [t.data_ptr() for t in starts_dec]
        decompositions._foreach_lerp_inplace_scalarlist_decomp(starts_dec, ends, _SCALARS)
        assert [t.data_ptr() for t in starts_dec] == starts_dec_ptrs
        for r, e in zip(starts_dec, starts_ref, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)


# ─── End-to-end torch.compile trace ──────────────────────────────────────


class TestForeachScalarListCompile:
    """``torch.compile(aot_eager)`` traces past every ScalarList overload
    with the decomposition applied. If AOT autograd stops applying the decomp
    (e.g., functionalization change or table-plumbing regression), these fail
    immediately.
    """

    def test_functional_traces_through(self):
        tensors = _build_tensors()

        def fn(*ts):
            return aten._foreach_clamp_max.ScalarList(list(ts), _SCALARS)

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)(*tensors)
        expected = aten._foreach_clamp_max.ScalarList(tensors, _SCALARS)
        for r, e in zip(compiled, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_out_traces_through(self):
        """The ``.ScalarList_out`` overload — verifies functionalization
        rewrites out= into functional + copy_ and both decomps compose."""
        tensors = _build_tensors()

        def fn(*ts):
            outs = [torch.empty_like(t) for t in ts]
            aten._foreach_maximum.ScalarList_out(list(ts), _SCALARS, out=outs)
            return outs

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)(*tensors)
        expected = [
            torch.maximum(t, torch.full_like(t, s)) for t, s in zip(tensors, _SCALARS, strict=True)
        ]
        for r, e in zip(compiled, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)

    def test_lerp_traces_through(self):
        starts = _build_tensors()
        torch.manual_seed(1)
        ends = [torch.randn(6, 6) for _ in range(4)]

        def fn(*args):
            n = len(args) // 2
            return aten._foreach_lerp.ScalarList(list(args[:n]), list(args[n:]), _SCALARS)

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)(*starts, *ends)
        expected = aten._foreach_lerp.ScalarList(starts, ends, _SCALARS)
        for r, e in zip(compiled, expected, strict=True):
            torch.testing.assert_close(r, e, atol=1e-6, rtol=1e-6, equal_nan=True)
