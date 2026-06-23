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
