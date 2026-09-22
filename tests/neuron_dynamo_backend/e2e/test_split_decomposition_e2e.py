"""
E2E tests for split decomposition (aten.split.Tensor and aten.split_with_sizes.default)
"""

import pytest
import torch

# (input_shape, split_size_or_sizes, dim, dtype, tol)
SPLIT_TENSOR_CONFIGS = [
    pytest.param((12,), 4, 0, torch.float32, 1e-4, id="1d-even-f32"),
    pytest.param((10,), 3, 0, torch.float32, 1e-4, id="1d-uneven-f32"),
    pytest.param((4, 6), 2, 1, torch.float32, 1e-4, id="2d-dim1-f32"),
    pytest.param((4, 6), 3, 0, torch.float32, 1e-4, id="2d-dim0-f32"),
    pytest.param((2, 3, 8), 4, 2, torch.float32, 1e-4, id="3d-dim2-f32"),
    pytest.param((2, 3, 8), 4, -1, torch.float32, 1e-4, id="3d-neg_dim-f32"),
    pytest.param((12,), 4, 0, torch.float16, 5e-2, id="1d-even-f16"),
    pytest.param((12,), 4, 0, torch.bfloat16, 5e-2, id="1d-even-bf16"),
    pytest.param((4, 6), 2, 1, torch.int32, 0, id="2d-dim1-i32"),
]

SPLIT_WITH_SIZES_CONFIGS = [
    pytest.param((12,), [3, 4, 5], 0, torch.float32, 1e-4, id="1d-sizes-f32"),
    pytest.param((4, 6), [1, 2, 3], 1, torch.float32, 1e-4, id="2d-dim1-sizes-f32"),
    pytest.param((4, 6), [1, 3], 0, torch.float32, 1e-4, id="2d-dim0-sizes-f32"),
    pytest.param((2, 3, 8), [2, 6], 2, torch.float32, 1e-4, id="3d-dim2-sizes-f32"),
    pytest.param((2, 3, 8), [2, 6], -1, torch.float32, 1e-4, id="3d-neg_dim-sizes-f32"),
    pytest.param((12,), [3, 4, 5], 0, torch.float16, 5e-2, id="1d-sizes-f16"),
    pytest.param((12,), [3, 4, 5], 0, torch.bfloat16, 5e-2, id="1d-sizes-bf16"),
    pytest.param((4, 6), [1, 2, 3], 1, torch.int32, 0, id="2d-dim1-sizes-i32"),
]


class TestSplitDecompositionE2E:
    """E2E tests verifying split decompositions match CPU results via torch.compile."""

    @pytest.mark.parametrize("shape,split_size,dim,dtype,tol", SPLIT_TENSOR_CONFIGS)
    def test_split_tensor(self, shape, split_size, dim, dtype, tol):
        torch.manual_seed(0)
        if dtype in (torch.int32,):
            x = torch.randint(-100, 100, shape, dtype=dtype)
        else:
            x = torch.randn(shape, dtype=dtype)

        def fn(t):
            return torch.split(t, split_size, dim=dim)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(x.to("neuron"))

        assert len(neuron_out) == len(cpu_out)
        for c, n in zip(cpu_out, neuron_out, strict=False):
            torch.testing.assert_close(n.cpu(), c, rtol=tol, atol=tol)

    @pytest.mark.parametrize("shape,split_sizes,dim,dtype,tol", SPLIT_WITH_SIZES_CONFIGS)
    def test_split_with_sizes(self, shape, split_sizes, dim, dtype, tol):
        torch.manual_seed(0)
        if dtype in (torch.int32,):
            x = torch.randint(-100, 100, shape, dtype=dtype)
        else:
            x = torch.randn(shape, dtype=dtype)

        def fn(t):
            return torch.split(t, split_sizes, dim=dim)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(x.to("neuron"))

        assert len(neuron_out) == len(cpu_out)
        for c, n in zip(cpu_out, neuron_out, strict=False):
            torch.testing.assert_close(n.cpu(), c, rtol=tol, atol=tol)
