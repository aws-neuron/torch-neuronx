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


class TestTopkNanRegression:
    """Regression for the @AwsNeuronTopK index clamp (TorchToNeuron).

    On all-NaN input the Neuron topk custom call returns index -1 for every
    slot. The lowering clamps the i64 index with max(idx, 0) before the
    inverse permute, so a downstream scatter never receives an out-of-range
    column (NRT 1006 indirect-DMA OOB on the MoE-router-weight gradient).

    Do not compare against CPU: torch.topk on NaN is undefined. The
    contract this test verifies is that no -1 sentinel escapes and the
    downstream scatter runs to completion on device.
    """

    @pytest.mark.parametrize(
        "shape,k,dtype",
        [
            pytest.param((96, 512), 10, torch.bfloat16, id="moe-router-bf16"),
            pytest.param((96, 512), 10, torch.float32, id="moe-router-f32"),
            pytest.param((8, 128), 4, torch.bfloat16, id="small-bf16"),
        ],
    )
    def test_topk_all_nan_indices_in_range(self, shape, k, dtype):
        """Every returned index must be in [0, n) on all-NaN input."""
        x = torch.full(shape, float("nan"), dtype=dtype)

        def fn(t):
            return torch.topk(t, k=k)

        with torch.no_grad():
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            _, neuron_indices = compiled_fn(x.to("neuron"))

        n = shape[-1]
        idx = neuron_indices.cpu()
        assert idx.dtype == torch.int64
        assert idx.shape == (*shape[:-1], k)
        assert bool((idx >= 0).all()), f"topk index below 0: min={idx.min().item()}"
        assert bool((idx < n).all()), f"topk index >= n: max={idx.max().item()}"

    def test_topk_all_nan_scatter_regression(self):
        """Feed all-NaN topk indices into scatter and expect no device fault.

        Before the clamp
            zeros[96,n].scatter_(1, topk(nan)[1], ones)
        tripped NRT 1006 on the router-weight gradient. With the clamp the
        same graph must run to completion on device.
        """
        n = 512
        k = 10
        x = torch.full((96, n), float("nan"), dtype=torch.bfloat16)

        def fn(t):
            _, indices = torch.topk(t, k=k)
            zeros = torch.zeros((96, n), dtype=torch.bfloat16, device=t.device)
            ones = torch.ones((96, k), dtype=torch.bfloat16, device=t.device)
            return zeros.scatter(1, indices, ones)

        with torch.no_grad():
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            out = compiled_fn(x.to("neuron"))

        out_cpu = out.cpu()
        assert out.shape == (96, n)
        assert bool(torch.isfinite(out_cpu).all()), "scatter output has non-finite values"


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
