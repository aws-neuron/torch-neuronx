"""End-to-end compile tests for ops lowered natively in torch-mlir
(TorchToStablehlo), as opposed to Python decompositions.

These ops are primitive special functions with no lowerable decomposition: PyTorch
`_refs` route them to `prims.*` ops that torch-mlir cannot legalize. Instead
we emit the corresponding CHLO op (e.g. `chlo.lgamma`) in the Torch->StableHLO
conversion and let the upstream `chlo-legalize-to-stablehlo` pass finish lowering it.

The IR-level checks live in the torch-mlir LIT suite
(`third_party/torch-mlir/test/Conversion/TorchToStablehlo/special_functions.mlir`);
these tests assert the ops compile and run correctly on the Neuron device.
"""

from typing import ClassVar

import pytest
import torch

_DTYPES = [
    (torch.float32, 1e-4, 1e-4),
    (torch.bfloat16, 1e-2, 1e-2),
    (torch.float16, 1e-2, 1e-2),
]


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestSpecialFunctionLoweringE2E:
    """Special-function ops greened via native CHLO lowerings."""

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_erfc_e2e(self, dtype, rtol, atol):
        def fn(x):
            return torch.erfc(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(-2.0, 2.0, 32, dtype=dtype).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)

    def test_erfc_tail_accuracy_e2e(self):
        # f32 only: tail values ~1e-5..1e-17 underflow in bf16/f16, so the tail check
        # is only meaningful in single precision.
        def fn(x):
            return torch.erfc(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(3.0, 6.0, 32).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        expected = fn(x_cpu)
        assert (result.cpu() > 0).all()
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-20)

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_erfinv_e2e(self, dtype, rtol, atol):
        def fn(x):
            return torch.erfinv(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(-0.95, 0.95, 32, dtype=dtype).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_lgamma_e2e(self, dtype, rtol, atol):
        def fn(x):
            return torch.lgamma(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(0.2, 8.0, 32, dtype=dtype).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_digamma_e2e(self, dtype, rtol, atol):
        def fn(x):
            return torch.digamma(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(0.2, 8.0, 32, dtype=dtype).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_special_i1e_e2e(self, dtype, rtol, atol):
        def fn(x):
            return torch.special.i1e(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(0.2, 8.0, 32, dtype=dtype).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
    def test_nextafter_e2e(self, dtype, rtol, atol):
        def fn(x, y):
            return torch.nextafter(x, y)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(0.2, 8.0, 32, dtype=dtype).reshape(4, 8)
        y_cpu = (torch.rand(4, 8) + 1.1).to(dtype)
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=rtol, atol=atol)

    # zeta is f32 only
    def test_special_zeta_e2e(self):
        def fn(x, y):
            return torch.special.zeta(x, y)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(4, 8) + 1.1  # zeta(x, q): x > 1
        y_cpu = torch.rand(4, 8) + 1.1
        result = compiled_fn(x_cpu.to("neuron"), y_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu, y_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestCompositeSpecialFunctionE2E:
    """Composite special functions decomposed onto ops that lower natively above."""

    @pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
    def test_mvlgamma_e2e(self, p):
        def fn(x):
            return torch.mvlgamma(x, p)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        # mvlgamma requires self > (p - 1) / 2
        x_cpu = torch.linspace(2.1, 6.0, 32).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    # Includes large +x
    # piecewise decomp stays bounded there (asymptotic series).
    @pytest.mark.parametrize("lo,hi", [(-1.5, 1.5), (2.0, 12.0), (10.0, 30.0)])
    def test_erfcx_e2e(self, lo, hi):
        def fn(x):
            return torch.special.erfcx(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.linspace(lo, hi, 32).reshape(4, 8)
        result = compiled_fn(x_cpu.to("neuron"))
        assert torch.isfinite(result.cpu()).all()  # no overflow to inf/nan
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-3, atol=1e-3)

    def test_special_zeta_other_scalar_e2e(self):
        def fn(x):
            return torch.special.zeta(x, 2.0)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(4, 8) + 1.1  # zeta(x, q): x > 1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)

    def test_special_zeta_self_scalar_e2e(self):
        def fn(x):
            return torch.special.zeta(2.0, x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.rand(4, 8) + 1.1
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestRoundFloorInt32ClampRegressionE2E:
    """
    stablehlo floor, ceil, and round_nearest_even lower through an int32
    conversion in neuronx-cc. That conversion clamps a large fp32 value to
    +/-2^31 and returns 2^31 for nan. For |x| >= 2^23 an fp32 has no fractional
    bits, so the op is the identity. The torch-mlir lowering must return the
    input for those values and for nan. torch.trunc has no direct pattern and
    decomposes into floor/ceil, so it must inherit the fix.
    """

    # Exact integers in fp32. The values with |x| > 2^31 hit the old clamp, and
    # the large negatives caught floor's sign flip. inf and nan check the
    # passthrough guards.
    _VALUES: ClassVar[list[float]] = [
        2.0**23,
        2.0**23 + 512,
        2.0**24,
        1.0e9,
        3.0e9,
        1.0e10,
        1.0e15,
        -(2.0**24),
        -3.0e9,
        -1.0e15,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]

    @pytest.mark.parametrize("op", [torch.round, torch.floor, torch.ceil, torch.trunc])
    def test_large_magnitude_identity_e2e(self, op):
        def fn(x):
            return op(x)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.tensor(self._VALUES, dtype=torch.float32)
        result = compiled_fn(x_cpu.to("neuron"))
        # The values are already integers, so the op must return them unchanged.
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=0, atol=0, equal_nan=True)

    def test_round_decimals_zero_e2e(self):
        # round(decimals=0) routes through the same round lowering.
        def fn(x):
            return torch.round(x, decimals=0)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x_cpu = torch.tensor(self._VALUES, dtype=torch.float32)
        result = compiled_fn(x_cpu.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x_cpu), rtol=0, atol=0, equal_nan=True)
