# ruff: noqa: N806

"""End-to-end compile tests for fractional_max_pool2d / fractional_max_pool3d.

These ops have no PyTorch decomposition and no torch-mlir lowering. We register
Python decomps that port aten's generate_intervals (FractionalMaxPooling.h):
pooling-region starts are derived from the random_samples input, windows are
fixed size (= kernel), and the result is a per-(N,C) gather + windowed max.
Output shape is static (output_size is given); only the gather indices are
data-dependent, which lowers fine on Neuron.

Note on random_samples layout (matches aten): 2d uses [...,0]=W, [...,1]=H;
3d uses [...,0]=T(depth), [...,1]=H, [...,2]=W.

Each op is driven through torch.compile(backend="neuron") and compared against
the eager CPU reference, using the same random_samples for both so the
pseudo-random regions match.
"""

import pytest
import torch

_TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2}


def _close(a, b, dtype):
    tol = _TOL.get(dtype, 1e-4)
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestFractionalMaxPoolDecompositionsE2E:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("kernel,out_hw", [((2, 2), (4, 4)), ((2, 2), (3, 3))])
    def test_fractional_max_pool2d_e2e(self, kernel, out_hw, dtype):
        def fn(x, samples):
            return torch.ops.aten.fractional_max_pool2d.default(x, kernel, out_hw, samples)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x = torch.randn(2, 3, 16, 16, dtype=dtype)
        samples = torch.rand(2, 3, 2, dtype=dtype)
        ref_vals, ref_idx = fn(x, samples)
        out_vals, out_idx = compiled_fn(x.to("neuron"), samples.to("neuron"))
        _close(out_vals.cpu(), ref_vals, dtype)
        assert torch.equal(out_idx.cpu(), ref_idx)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("kernel,out_dhw", [((2, 2, 2), (4, 4, 4)), ((2, 2, 2), (2, 3, 3))])
    def test_fractional_max_pool3d_e2e(self, kernel, out_dhw, dtype):
        def fn(x, samples):
            return torch.ops.aten.fractional_max_pool3d.default(x, kernel, out_dhw, samples)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, 8, 8, dtype=dtype)
        samples = torch.rand(2, 3, 3, dtype=dtype)
        ref_vals, ref_idx = fn(x, samples)
        out_vals, out_idx = compiled_fn(x.to("neuron"), samples.to("neuron"))
        _close(out_vals.cpu(), ref_vals, dtype)
        assert torch.equal(out_idx.cpu(), ref_idx)
