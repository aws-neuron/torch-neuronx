# ruff: noqa: N806

"""
Unit tests for special math function decompositions:
i0, i0e, i1, i1e, bessel_j0, bessel_j1, spherical_bessel_j0,
ndtri, log_ndtr, laguerre_polynomial_l, polygamma, erfcx.
"""

from typing import ClassVar

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# ── Special-function decompositions ──────────────────────────────────────────


def _special_cpu_reference(fn, x_cpu):
    """CPU reference; some special ops (bessel_j0/j1, spherical_bessel_j0,
    ndtri, log_ndtr, laguerre_polynomial_l) have no bf16/fp16 CPU kernel — run
    them in fp32 and cast back so torch.testing.assert_close defaults apply.
    """
    if x_cpu.dtype in (torch.bfloat16, torch.float16):
        return fn(x_cpu.float()).to(x_cpu.dtype)
    return fn(x_cpu)


# A&S polynomial evaluation on Neuron loses a ULP or two vs CPU fp32; error
# scales with function magnitude (i0(9)~1094 → abs err ~1e-2). bf16 defaults
# (rtol=1.6e-2) dominate. Still tighter than kaiser_window's E2E (rtol=1e-3).
_AS_POLY_RTOL = 1e-5
_AS_POLY_ATOL = 1e-3


def _zero_crossing_tol(dtype):
    """Relative error explodes at function zeros (sinc, ndtri near 0.5,
    log_ndtr for large positive x); bf16 quantization dominates for those dtypes."""
    if dtype in (torch.bfloat16, torch.float16):
        return {"rtol": 2e-2, "atol": 2e-2}
    return {"rtol": 1e-4, "atol": 1e-3}


class TestSpecialFunctionDecompositionsRegistered:
    @pytest.mark.parametrize(
        "op",
        [
            torch.ops.aten.i0.default,
            torch.ops.aten.special_i0e.default,
            torch.ops.aten.special_i1.default,
            torch.ops.aten.special_bessel_j0.default,
            torch.ops.aten.special_bessel_j1.default,
            torch.ops.aten.special_spherical_bessel_j0.default,
            torch.ops.aten.special_ndtri.default,
            torch.ops.aten.special_log_ndtr.default,
            torch.ops.aten.special_laguerre_polynomial_l.default,
            torch.ops.aten.special_laguerre_polynomial_l.n_scalar,
            torch.ops.aten.special_laguerre_polynomial_l.x_scalar,
            torch.ops.aten.polygamma.default,
            torch.ops.aten.special_erfcx.default,
            torch.ops.aten.igamma.default,
            torch.ops.aten.igammac.default,
            torch.ops.aten.special_gammainc.default,
            torch.ops.aten.special_gammaincc.default,
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


class TestErfcxDecomposition:
    """erfcx(x) = exp(x^2) * erfc(x).

    Bounded as x -> +inf, but grows as ~2*exp(x^2) as x -> -inf. A prior
    decomposition clamped exp(x^2) and branched on signed x, which truncated
    the large-negative tail. These CPU-only tests check the decomposition math
    against torch.special.erfcx; they need no Neuron device.
    """

    def test_erfcx_matches_reference_across_range(self):
        # Covers the large-negative tail (|x| up to 5) that the clamp broke.
        x = torch.linspace(-5.0, 5.0, 4001, dtype=torch.float32)
        got = decompositions.special_erfcx_decomp(x)
        ref = torch.special.erfcx(x)
        torch.testing.assert_close(got, ref, rtol=3e-3, atol=1e-5)

    def test_erfcx_negative_tail_not_truncated(self):
        # Regression: the old clamp returned ~2*e^10 (~4.4e4) at x=-5; true ~1.4e11.
        x = torch.tensor([-5.0], dtype=torch.float32)
        got = decompositions.special_erfcx_decomp(x)
        assert got.item() > 1e10
        torch.testing.assert_close(got, torch.special.erfcx(x), rtol=1e-3, atol=0.0)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestI0FamilyDecompositionsE2E:
    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_i0_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.i0.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 3.0  # cross both A&S branches
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=_AS_POLY_RTOL, atol=_AS_POLY_ATOL)

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_i0e_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_i0e.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 3.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=_AS_POLY_RTOL, atol=_AS_POLY_ATOL)

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_i1_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_i1.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 3.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=_AS_POLY_RTOL, atol=_AS_POLY_ATOL)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBesselJDecompositionsE2E:
    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_bessel_j0_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_bessel_j0.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 3.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=_AS_POLY_RTOL, atol=_AS_POLY_ATOL)

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_bessel_j1_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_bessel_j1.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 3.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=_AS_POLY_RTOL, atol=_AS_POLY_ATOL)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSphericalBesselJ0DecompositionE2E:
    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_spherical_bessel_j0_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_spherical_bessel_j0.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 4.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, **_zero_crossing_tol(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_spherical_bessel_j0_near_zero(self, dtype):
        """Straddle the internal 1e-4 branch threshold at x=0."""

        def fn(x):
            return torch.ops.aten.special_spherical_bessel_j0.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.tensor([0.0, 1e-6, 1e-3, 0.5, 1.0], dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestNdtriLogNdtrDecompositionsE2E:
    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_ndtri_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(p):
            return torch.ops.aten.special_ndtri.default(p)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        # Keep p bounded away from 0 and 1 (where ndtri diverges).
        p_cpu = torch.rand(shape, dtype=dtype) * 0.9 + 0.05
        result = compiled_fn(p_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, p_cpu)
        torch.testing.assert_close(result.cpu(), expected, **_zero_crossing_tol(dtype))

    @pytest.mark.parametrize(
        "shape",
        [(8,), (4, 8), (2, 4, 8)],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_special_log_ndtr_e2e(self, shape, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_log_ndtr.default(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(shape, dtype=dtype) * 2.0
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected, **_zero_crossing_tol(dtype))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLaguerrePolynomialDecompositionE2E:
    # .n_scalar takes a compile-time scalar order (exact unrolled recurrence).
    # .default and .x_scalar take a per-element tensor order n: the effective
    # order is trunc(n) (n<=-1 -> 0), evaluated by a fixed-depth recurrence +
    # select (exact for trunc(n) <= _LAGUERRE_MAX_ORDER).

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_laguerre_n_scalar_e2e(self, n, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.special_laguerre_polynomial_l.n_scalar(x, n)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.randn(4, 8, dtype=dtype)
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected)

    def test_laguerre_tensor_n_default_e2e(self):
        """.default: per-element tensor order. Covers trunc semantics
        (non-integer n), the n<=-1 -> 0 branch, and x=0 (L_n(0)=1). fp32 —
        the coverage sample is fp32; bf16 order-recurrence error is exercised
        via .n_scalar."""
        torch.manual_seed(0)

        def fn(x, n):
            return torch.ops.aten.special_laguerre_polynomial_l.default(x, n)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        # x,n span the sample domain -> effective orders trunc(n) in [-5,5].
        x_cpu = torch.randn(16, 16) * 1.6
        n_cpu = torch.rand(16, 16) * 10.0 - 5.0
        # pin a few edge cells: x=0, exact integer, negative order.
        x_cpu[0, 0] = 0.0
        n_cpu[0, 0] = 3.0
        n_cpu[0, 1] = -1.0
        n_cpu[0, 2] = 2.7
        result = compiled_fn(x_cpu.to("neuron"), n_cpu.to("neuron"))
        expected = torch.special.laguerre_polynomial_l(x_cpu, n_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_laguerre_tensor_n_x_scalar_e2e(self):
        """.x_scalar: scalar x, per-element tensor order n."""
        torch.manual_seed(0)

        def fn(n):
            return torch.ops.aten.special_laguerre_polynomial_l.x_scalar(0.5, n)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        n_cpu = torch.rand(16, 16) * 10.0 - 5.0
        result = compiled_fn(n_cpu.to("neuron"))
        expected = torch.special.laguerre_polynomial_l(torch.tensor(0.5), n_cpu)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_laguerre_tensor_n_order_cap(self):
        """Orders above _LAGUERRE_MAX_ORDER are out of contract: the fixed-depth
        unroll can't reach them, so those lanes return NaN rather than a stale
        (silently-wrong) L_MAX_ORDER value. Orders at/below the cap stay exact,
        and the mask is per-element (a mix of in- and out-of-range orders NaNs
        only the out-of-range lanes)."""
        from torch_neuronx.neuron_dynamo_backend.decompositions import _LAGUERRE_MAX_ORDER

        cap = _LAGUERRE_MAX_ORDER

        def fn(x, n):
            return torch.ops.aten.special_laguerre_polynomial_l.default(x, n)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        # Contract is on the effective order trunc(n): cap and cap+0.9 both
        # trunc to cap (in range); cap+1 and cap+1.9 trunc past it (out of range).
        n_cpu = torch.tensor([0.0, 1.0, float(cap), cap + 0.9, cap + 1.0, cap + 1.9])
        x_cpu = torch.full_like(n_cpu, 0.5)
        result = compiled_fn(x_cpu.to("neuron"), n_cpu.to("neuron")).cpu()

        in_range = n_cpu.trunc() <= cap
        # In-contract lanes match eager exactly.
        torch.testing.assert_close(
            result[in_range],
            torch.special.laguerre_polynomial_l(x_cpu, n_cpu)[in_range],
            rtol=1e-4,
            atol=1e-4,
        )
        # Out-of-contract lanes are NaN (loud), not a plausible finite value.
        assert torch.isnan(result[~in_range]).all(), result[~in_range]


def _binary_special_cpu_reference(fn, a_cpu, x_cpu):
    """CPU reference for a binary special op; igamma/igammac have no bf16/fp16
    CPU kernel, so compute in fp32 and cast back (mirrors
    ``_special_cpu_reference``)."""
    if a_cpu.dtype in (torch.bfloat16, torch.float16):
        return fn(a_cpu.float(), x_cpu.float()).to(a_cpu.dtype)
    return fn(a_cpu, x_cpu)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestIncompleteGammaDecompositionsE2E:
    """igamma / igammac and their special_gammainc / special_gammaincc aliases.

    Both the power-series branch (x < a+1) and the continued-fraction branch
    (x >= a+1) are exercised; ``igamma`` computes P(a,x), ``igammac`` computes
    Q(a,x) = 1 - P(a,x).
    """

    # (a_range, x_range, label) — series-only, CF-only, and a straddling mix.
    _DOMAINS: ClassVar = [
        ((2.0, 6.0), (0.0, 1.0), "series"),  # x < a+1 everywhere
        ((0.5, 2.0), (5.0, 10.0), "cf"),  # x >= a+1 everywhere
        ((0.05, 6.0), (0.0, 8.0), "mixed"),  # straddles the branch boundary
    ]

    @pytest.mark.parametrize("a_range,x_range,label", _DOMAINS)
    @pytest.mark.parametrize(
        "op,ref",
        [
            (torch.ops.aten.igamma.default, torch.igamma),
            (torch.ops.aten.igammac.default, torch.igammac),
            (torch.ops.aten.special_gammainc.default, torch.special.gammainc),
            (torch.ops.aten.special_gammaincc.default, torch.special.gammaincc),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_incomplete_gamma_e2e(self, op, ref, a_range, x_range, label, dtype):
        torch.manual_seed(0)

        def fn(a, x):
            return op(a, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        a_lo, a_hi = a_range
        x_lo, x_hi = x_range
        a_cpu = (torch.rand(4, 8, dtype=dtype) * (a_hi - a_lo) + a_lo).clamp_min(1e-2)
        x_cpu = torch.rand(4, 8, dtype=dtype) * (x_hi - x_lo) + x_lo
        result = compiled_fn(a_cpu.to("neuron"), x_cpu.to("neuron"))
        expected = _binary_special_cpu_reference(fn, a_cpu, x_cpu)
        # Series (x<a+1) and CF (x>=a+1) each lose a ULP or two on device vs the
        # CPU fp32 kernel; regularized values live in [0, 1] so the absolute
        # floor keeps the ~0 tail from tripping the rel test. bf16 is looser
        # because input quantization (~3 decimal digits) dominates.
        if dtype in (torch.bfloat16, torch.float16):
            rtol, atol = 2e-2, 2e-2
        else:
            rtol, atol = 1e-4, 1e-4
        torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_incomplete_gamma_x_zero_edge(self, dtype):
        """At x=0, P(a,0)=0 and Q(a,0)=1 for all a>0."""

        def lower(a, x):
            return torch.ops.aten.igamma.default(a, x)

        def upper(a, x):
            return torch.ops.aten.igammac.default(a, x)

        a_cpu = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=dtype)
        x_cpu = torch.zeros(4, dtype=dtype)
        lo = torch.compile(lower, backend="neuron", fullgraph=True)(
            a_cpu.to("neuron"), x_cpu.to("neuron")
        )
        up = torch.compile(upper, backend="neuron", fullgraph=True)(
            a_cpu.to("neuron"), x_cpu.to("neuron")
        )
        torch.testing.assert_close(lo.cpu(), torch.zeros(4, dtype=dtype))
        torch.testing.assert_close(up.cpu(), torch.ones(4, dtype=dtype))

    def test_incomplete_gamma_nan_domain(self):
        """The NaN domain must match eager exactly: NaN iff (a<0) or (x<0) or
        (a==0 & x==0); a==0,x>0 -> P=1,Q=0. A one-sided NaN vs eager (finite on
        one side, NaN on the other) is the failure this guards against — it is
        what surfaced when the domain mask used ``a>0`` instead of handling the
        ``a==0, x>0`` boundary.
        """
        # a, x chosen to hit every domain region including both signs and zeros.
        a = torch.tensor([-2.0, -0.5, 0.0, 0.0, 1.0, 2.0, -1.0, 0.5, 3.0])
        x = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.5])
        for op, ref in [
            (torch.ops.aten.igamma.default, torch.igamma),
            (torch.ops.aten.igammac.default, torch.igammac),
        ]:
            compiled = torch.compile(lambda a, x, op=op: op(a, x), backend="neuron", fullgraph=True)
            got = compiled(a.to("neuron"), x.to("neuron")).cpu()
            exp = ref(a, x)
            # NaN placement must match exactly (no one-sided NaN).
            assert torch.equal(
                torch.isnan(got), torch.isnan(exp)
            ), f"{op}: NaN mask mismatch got={torch.isnan(got)} exp={torch.isnan(exp)}"
            torch.testing.assert_close(got, exp, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestPolygammaDecompositionE2E:
    # n == 0 (digamma) is not covered; the decomp returns NotImplemented and
    # aten.digamma runs, which has its own correctness issues on Neuron.

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_polygamma_e2e(self, n, dtype):
        torch.manual_seed(0)

        def fn(x):
            return torch.ops.aten.polygamma.default(n, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        # polygamma is defined for x > 0 (Hurwitz zeta domain).
        x_cpu = torch.rand((4, 8), dtype=dtype) * 3.0 + 0.5
        result = compiled_fn(x_cpu.to("neuron"))
        expected = _special_cpu_reference(fn, x_cpu)
        torch.testing.assert_close(result.cpu(), expected)
