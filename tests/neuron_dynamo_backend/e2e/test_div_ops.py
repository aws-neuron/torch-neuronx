"""
E2E tests for div decompositions (div.Tensor_mode and div.Scalar_mode with rounding_mode).
"""

import pytest
import torch

SHAPES = [
    pytest.param((64,), id="1d"),
    pytest.param((16, 32), id="2d"),
    pytest.param((8, 17), id="2d-odd"),
    pytest.param((8, 16, 32), id="3d"),
    pytest.param((256, 256), id="large-2d"),
]

DTYPES = [
    pytest.param(torch.float32, id="f32"),
    pytest.param(torch.float16, id="f16"),
    pytest.param(torch.bfloat16, id="bf16"),
]

ROUNDING_MODES = [
    pytest.param(None, id="none"),
    pytest.param("trunc", id="trunc"),
    pytest.param("floor", id="floor"),
]

# Boundary classes cover every float dtype: the correction runs in fp32 (fp16/bf16 upcast
# and downcast once) so it applies across all of them.
BOUNDARY_DTYPES = DTYPES


def _boundary_grid(dtype):
    """Every (num, den) pair over a grid of fractional values, so many quotients land
    exactly on / just under integer boundaries. Shared by the boundary classes below."""
    vals = torch.arange(-16, 16, 0.5)
    num, den = torch.cartesian_prod(vals, vals).unbind(1)
    keep = den != 0
    return num[keep].to(dtype), den[keep].to(dtype)


class TestDivBoundaryE2E:
    """div.Tensor_mode at integer-boundary quotients, for the modes the fix covers (None +
    floor). The torch-mlir floor lowering applies an exact-boundary correction for fp32; a
    naive floor(x/y) lands one integer off when the fp32 quotient rounds an exact boundary.
    floor_divide / remainder have their own classes below.
    """

    @pytest.mark.parametrize("dtype", BOUNDARY_DTYPES)
    @pytest.mark.parametrize(
        "rounding_mode", [pytest.param(None, id="none"), pytest.param("floor", id="floor")]
    )
    def test_boundary(self, dtype, rounding_mode):
        num, den = _boundary_grid(dtype)

        def fn(a, b):
            return torch.div(a, b, rounding_mode=rounding_mode)

        with torch.no_grad():
            cpu_out = fn(num, den)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(num.to("neuron"), den.to("neuron"))
        if rounding_mode is None:
            rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
            torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)
        else:
            torch.testing.assert_close(neuron_out.cpu(), cpu_out, equal_nan=True)


class TestFloorDivideBoundaryE2E:
    """floor_divide at integer boundaries under compile -- fences the floor_divide decomp.

    The decomposition would otherwise floor an fp32-rounded quotient, one integer off at a
    boundary quotient; the Neuron floor lowering applies the exact-boundary correction (fp32).
    Fails on the naive decomposition.
    """

    @pytest.mark.parametrize("dtype", BOUNDARY_DTYPES)
    def test_boundary(self, dtype):
        num, den = _boundary_grid(dtype)

        def fn(a, b):
            return torch.floor_divide(a, b)

        with torch.no_grad():
            cpu_out = fn(num, den)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(num.to("neuron"), den.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, equal_nan=True)


class TestRemainderBoundaryE2E:
    """remainder at integer boundaries under compile -- fences the remainder decomp.

    A boundary quotient that floors one integer off shifts remainder = x - floor(x/y)*y by
    ~y; the Neuron decomposition recovers the exact floor. Fails on the naive decomposition.
    """

    @pytest.mark.parametrize("dtype", BOUNDARY_DTYPES)
    def test_boundary(self, dtype):
        num, den = _boundary_grid(dtype)

        def fn(a, b):
            return torch.remainder(a, b)

        with torch.no_grad():
            cpu_out = fn(num, den)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(num.to("neuron"), den.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, equal_nan=True)


INTEGER_DTYPES = [
    pytest.param(torch.int8, id="i8"),
    pytest.param(torch.int16, id="i16"),
    pytest.param(torch.int32, id="i32"),
    pytest.param(torch.int64, id="i64"),
]


def _integer_grid(dtype):
    """Every (num, den) pair over small values (mixed-sign floor semantics) plus, for
    >= 32-bit dtypes, values past 2**24 (the fp32-datapath precision path)."""
    info = torch.iinfo(dtype)
    vals = list(range(-9, 10))
    if info.bits >= 32:
        vals += [2**24 + 1, -(2**24 + 1), 100000001, -100000001, 2**30, -(2**30)]
    vals = torch.tensor([v for v in vals if info.min <= v <= info.max], dtype=dtype)
    num, den = torch.cartesian_prod(vals, vals).unbind(1)
    keep = den != 0
    return num[keep], den[keep]


class TestIntegerRemainderE2E:
    """Signed integer remainder under compile -- Python floor-division semantics.

    remainder decomposed to x - floor(intdiv(x, y)) * y, but flooring an already-truncated
    integer quotient is a no-op, so it collapsed to C-style trunc remainder (wrong sign for
    mixed-sign operands, e.g. -7 % 3 -> -1 instead of 2). Now it routes through div(floor).
    Fails on the old lowering. (Integer floor_divide itself is fixed separately.)
    """

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_remainder(self, dtype):
        num, den = _integer_grid(dtype)

        def fn(a, b):
            return torch.remainder(a, b)

        with torch.no_grad():
            cpu_out = fn(num, den)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(num.to("neuron"), den.to("neuron"))
        assert torch.equal(neuron_out.cpu(), cpu_out)


class TestDivTensorModeE2E:
    """E2E tests for aten.div.Tensor_mode (tensor / tensor with rounding_mode)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
    def test_div_tensor_mode(self, shape, dtype, rounding_mode):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype)
        # Large divisor keeps quotients well away from integer boundaries,
        # avoiding rounding-mode sensitivity from reduced-precision arithmetic.
        y = torch.randn(shape, dtype=dtype).abs().clamp(min=4.0)

        def fn(a, b):
            return torch.div(a, b, rounding_mode=rounding_mode)

        with torch.no_grad():
            cpu_out = fn(x, y)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), y.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)


class TestDivScalarModeE2E:
    """E2E tests for aten.div.Scalar_mode (tensor / scalar with rounding_mode)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
    def test_div_scalar_mode(self, shape, dtype, rounding_mode):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype)

        def fn(a):
            return torch.div(a, 8.0, rounding_mode=rounding_mode)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)
