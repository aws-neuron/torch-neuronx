"""E2E tests for aten.var decomposition (var.dim and var.correction)."""

import pytest
import torch

SHAPES = [
    pytest.param((64,), id="1d"),
    pytest.param((8, 16), id="2d"),
    pytest.param((4, 8, 16), id="3d"),
]

DTYPES = [
    pytest.param(torch.float32, id="f32"),
    pytest.param(torch.bfloat16, id="bf16"),
]


class TestVarDimE2E:
    """E2E tests for aten.var.dim (unbiased and biased)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_var_dim_unbiased(self, shape, dtype):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype)

        def fn(a):
            return torch.var(a, dim=-1, unbiased=True)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        rtol, atol = (5e-2, 5e-3) if dtype == torch.bfloat16 else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_var_dim_biased(self, shape, dtype):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=dtype)

        def fn(a):
            return torch.var(a, dim=-1, unbiased=False)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        rtol, atol = (5e-2, 5e-3) if dtype == torch.bfloat16 else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)


class TestVarCorrectionE2E:
    """E2E tests for aten.var.correction with explicit correction values."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("correction", [0, 1, 2])
    def test_var_correction_int(self, shape, correction):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.float32)

        def fn(a):
            return torch.var(a, dim=-1, correction=correction)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_var_correction_multi_dim(self, shape):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.float32)

        def fn(a):
            return torch.var(a, dim=list(range(len(shape))), correction=1)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_var_keepdim(self, shape):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.float32)

        def fn(a):
            return torch.var(a, dim=-1, correction=1, keepdim=True)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-4, atol=1e-5)
