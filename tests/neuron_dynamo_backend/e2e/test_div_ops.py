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
