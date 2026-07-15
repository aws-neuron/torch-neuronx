# ruff: noqa: N806

"""
Unit tests for special math function decompositions:
i0, i0e, i1, i1e, bessel_j0, bessel_j1, spherical_bessel_j0,
ndtri, log_ndtr, laguerre_polynomial_l, polygamma.
"""

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
        ],
    )
    def test_registered(self, op):
        assert op in decompositions.neuron_decompositions


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
    # .default and .x_scalar overloads take tensor n; the decomp bails to
    # NotImplemented in a torch.compile trace (see _extract_int_scalar) so
    # only .n_scalar is exercised e2e.

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
