"""
E2E tests for sort-related decompositions (argsort, sort, median, nanmedian).
"""

import logging

import pytest
import torch

logger = logging.getLogger(__name__)

# --- argsort configs ---
# (shape, dim, descending, dtype)
ARGSORT_CONFIGS = [
    pytest.param((64,), -1, False, torch.float32, id="1d-asc-f32"),
    pytest.param((64,), -1, True, torch.float32, id="1d-desc-f32"),
    pytest.param((16, 32), -1, False, torch.float32, id="2d-lastdim-asc-f32"),
    pytest.param((16, 32), 0, False, torch.float32, id="2d-dim0-asc-f32"),
    pytest.param((8, 16, 32), 1, False, torch.float32, id="3d-dim1-asc-f32"),
    pytest.param((8, 16, 32), -1, True, torch.float32, id="3d-lastdim-desc-f32"),
    pytest.param((256, 256), -1, False, torch.float32, id="large-2d-f32"),
    pytest.param((16, 32), -1, False, torch.float16, id="2d-f16"),
    pytest.param((16, 32), -1, False, torch.bfloat16, id="2d-bf16"),
    pytest.param(
        (16, 32),
        -1,
        False,
        torch.int64,
        id="2d-i64",
        marks=pytest.mark.xfail(reason="AwsNeuronTopK does not support integer types"),
    ),
    pytest.param((8, 17), -1, False, torch.float32, id="2d-odd-lastdim-f32"),
    pytest.param((13, 32), 0, False, torch.float32, id="2d-odd-dim0-f32"),
    pytest.param((4, 7, 16), 1, True, torch.float32, id="3d-odd-dim1-desc-f32"),
    pytest.param((), -1, False, torch.float32, id="0d-f32"),
]

# --- sort configs ---
# (shape, dim, descending, dtype)
SORT_CONFIGS = [
    pytest.param((64,), -1, False, torch.float32, id="1d-asc-f32"),
    pytest.param((64,), -1, True, torch.float32, id="1d-desc-f32"),
    pytest.param((16, 32), -1, False, torch.float32, id="2d-lastdim-asc-f32"),
    pytest.param((16, 32), 0, False, torch.float32, id="2d-dim0-asc-f32"),
    pytest.param((8, 16, 32), 1, True, torch.float32, id="3d-dim1-desc-f32"),
    pytest.param((256, 256), -1, False, torch.float32, id="large-2d-f32"),
    pytest.param((16, 32), -1, False, torch.float16, id="2d-f16"),
    pytest.param((16, 32), -1, False, torch.bfloat16, id="2d-bf16"),
    pytest.param((16, 32), -1, False, torch.int32, id="2d-i32"),
    pytest.param((16, 32), -1, False, torch.int64, id="2d-i64"),
    pytest.param((8, 17), -1, False, torch.float32, id="2d-odd-lastdim-f32"),
    pytest.param((13, 32), 0, False, torch.float32, id="2d-odd-dim0-f32"),
    pytest.param((4, 7, 16), 1, True, torch.float32, id="3d-odd-dim1-desc-f32"),
    pytest.param((), -1, False, torch.float32, id="0d-f32"),
    pytest.param(
        (10, 18000),
        -1,
        False,
        torch.float32,
        id="large-lastdim-f32",
        marks=pytest.mark.xfail(reason="sort dim size > 16384 not supported by Neuron"),
    ),
]

# --- median configs ---
# (shape, dim, keepdim, dtype)
MEDIAN_DIM_CONFIGS = [
    pytest.param((64,), 0, False, torch.float32, id="1d-f32"),
    pytest.param((16, 32), -1, False, torch.float32, id="2d-lastdim-f32"),
    pytest.param((16, 32), 0, False, torch.float32, id="2d-dim0-f32"),
    pytest.param((16, 32), -1, True, torch.float32, id="2d-keepdim-f32"),
    pytest.param((8, 16, 32), 1, False, torch.float32, id="3d-dim1-f32"),
    pytest.param((256, 4096), 1, False, torch.float32, id="large-2d-f32"),
    pytest.param((16, 32), -1, False, torch.float16, id="2d-f16"),
    pytest.param((16, 32), -1, False, torch.bfloat16, id="2d-bf16"),
    pytest.param((16, 32), -1, False, torch.int64, id="2d-i64"),
    pytest.param((8, 17), -1, False, torch.float32, id="2d-odd-lastdim-f32"),
    pytest.param((13, 32), 0, False, torch.float32, id="2d-odd-dim0-f32"),
    pytest.param((4, 9, 16), 1, False, torch.float32, id="3d-odd-dim1-f32"),
    pytest.param((), 0, False, torch.float32, id="0d-f32"),
]

# (shape, dtype)
MEDIAN_SCALAR_CONFIGS = [
    pytest.param((64,), torch.float32, id="1d-f32"),
    pytest.param((16, 32), torch.float32, id="2d-f32"),
    pytest.param((16, 32), torch.float16, id="2d-f16"),
    pytest.param((16, 32), torch.int64, id="2d-i64"),
]

# --- nanmedian configs ---
# (shape, dim, keepdim, dtype, nan_ratio)
NANMEDIAN_DIM_CONFIGS = [
    pytest.param((64,), 0, False, torch.float32, 0.0, id="1d-no_nan-f32"),
    pytest.param((64,), 0, False, torch.float32, 0.3, id="1d-some_nan-f32"),
    pytest.param((64,), 0, False, torch.float32, 1.0, id="1d-all_nan-f32"),
    pytest.param((16, 32), -1, False, torch.float32, 0.2, id="2d-lastdim-f32"),
    pytest.param((16, 32), 0, False, torch.float32, 0.2, id="2d-dim0-f32"),
    pytest.param((16, 32), -1, True, torch.float32, 0.2, id="2d-keepdim-f32"),
    pytest.param((8, 16, 32), 1, False, torch.float32, 0.1, id="3d-dim1-f32"),
    pytest.param((16, 32), -1, False, torch.float16, 0.2, id="2d-f16"),
    pytest.param((16, 32), -1, False, torch.bfloat16, 0.2, id="2d-bf16"),
    pytest.param((), 0, False, torch.float32, 0.0, id="0d-f32"),
]

# (shape, dtype, nan_ratio)
NANMEDIAN_SCALAR_CONFIGS = [
    pytest.param(
        (64,),
        torch.float32,
        0.0,
        id="1d-no_nan-f32",
        marks=pytest.mark.xfail(reason="nanmedian scalar hits neuronx-cc reduce-on-i64 bug"),
    ),
    pytest.param(
        (64,),
        torch.float32,
        0.3,
        id="1d-some_nan-f32",
        marks=pytest.mark.xfail(reason="nanmedian scalar hits neuronx-cc reduce-on-i64 bug"),
    ),
    pytest.param(
        (64,),
        torch.float32,
        1.0,
        id="1d-all_nan-f32",
        marks=pytest.mark.xfail(reason="nanmedian scalar hits neuronx-cc reduce-on-i64 bug"),
    ),
    pytest.param(
        (16, 32),
        torch.float32,
        0.2,
        id="2d-f32",
        marks=pytest.mark.xfail(reason="nanmedian scalar hits neuronx-cc reduce-on-i64 bug"),
    ),
]


def _inject_nans(x, ratio):
    """Randomly set a fraction of elements to NaN."""
    if ratio <= 0:
        return x
    mask = torch.rand_like(x, dtype=torch.float32) < ratio
    return x.masked_fill(mask, float("nan"))


class TestArgsortE2E:
    """E2E tests for argsort decomposition."""

    @pytest.mark.parametrize("shape,dim,descending,dtype", ARGSORT_CONFIGS)
    def test_argsort(self, shape, dim, descending, dtype):
        torch.manual_seed(42)
        if dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype)
        else:
            x = torch.randint(-100, 100, shape, dtype=dtype)

        def fn(t):
            return torch.argsort(t, dim=dim, descending=descending)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        if dtype in (torch.float16, torch.bfloat16):
            # bf16/fp16 have ties; verify gathered values are sorted
            neuron_gathered = torch.gather(x, dim, neuron_out.cpu())
            cpu_gathered = torch.gather(x, dim, cpu_out)
            torch.testing.assert_close(neuron_gathered, cpu_gathered, rtol=0, atol=0)
        else:
            torch.testing.assert_close(neuron_out.cpu(), cpu_out)


class TestSortE2E:
    """E2E tests for sort decomposition."""

    @pytest.mark.parametrize("shape,dim,descending,dtype", SORT_CONFIGS)
    def test_sort(self, shape, dim, descending, dtype):
        torch.manual_seed(42)
        if dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype)
        else:
            x = torch.randint(-100, 100, shape, dtype=dtype)

        def fn(t):
            return torch.sort(t, dim=dim, descending=descending)

        with torch.no_grad():
            cpu_values, cpu_indices = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_values, neuron_indices = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_values.cpu(), cpu_values, rtol=rtol, atol=atol)
        # ties (low-precision floats, integers): topk indices may differ, so check gathered values.
        if dtype in (torch.float16, torch.bfloat16) or not dtype.is_floating_point:
            neuron_gathered = torch.gather(x, dim, neuron_indices.cpu())
            torch.testing.assert_close(neuron_gathered, neuron_values.cpu(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(neuron_indices.cpu(), cpu_indices)


class TestMedianE2E:
    """E2E tests for median decomposition."""

    @pytest.mark.parametrize("shape,dim,keepdim,dtype", MEDIAN_DIM_CONFIGS)
    def test_median_dim(self, shape, dim, keepdim, dtype):
        torch.manual_seed(42)
        if dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype)
        else:
            x = torch.randint(-100, 100, shape, dtype=dtype)

        def fn(t):
            return torch.median(t, dim=dim, keepdim=keepdim)

        with torch.no_grad():
            cpu_values, cpu_indices = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_values, neuron_indices = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_values.cpu(), cpu_values, rtol=rtol, atol=atol)
        torch.testing.assert_close(neuron_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize("shape,dtype", MEDIAN_SCALAR_CONFIGS)
    def test_median_scalar(self, shape, dtype):
        torch.manual_seed(42)
        if dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype)
        else:
            x = torch.randint(-100, 100, shape, dtype=dtype)

        def fn(t):
            return torch.median(t)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol)


class TestNanmedianE2E:
    """E2E tests for nanmedian decomposition."""

    @pytest.mark.parametrize("shape,dim,keepdim,dtype,nan_ratio", NANMEDIAN_DIM_CONFIGS)
    def test_nanmedian_dim(self, shape, dim, keepdim, dtype, nan_ratio):
        torch.manual_seed(42)
        x = _inject_nans(torch.randn(shape, dtype=dtype), nan_ratio)

        def fn(t):
            return torch.nanmedian(t, dim=dim, keepdim=keepdim)

        with torch.no_grad():
            cpu_values, cpu_indices = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_values, neuron_indices = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(
            neuron_values.cpu(), cpu_values, rtol=rtol, atol=atol, equal_nan=True
        )
        # Indices are undefined for all-NaN slices
        if nan_ratio < 1.0:
            torch.testing.assert_close(neuron_indices.cpu(), cpu_indices)

    @pytest.mark.parametrize("shape,dtype,nan_ratio", NANMEDIAN_SCALAR_CONFIGS)
    def test_nanmedian_scalar(self, shape, dtype, nan_ratio):
        torch.manual_seed(42)
        x = _inject_nans(torch.randn(shape, dtype=dtype), nan_ratio)

        def fn(t):
            return torch.nanmedian(t)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        rtol, atol = (1e-2, 1e-3) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-5)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=rtol, atol=atol, equal_nan=True)
