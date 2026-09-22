"""
E2E tests for select_scatter decomposition (negative index normalization)
"""

import pytest
import torch

# (input_shape, src_shape, dim, index, dtype, tol)
SELECT_SCATTER_CONFIGS = [
    pytest.param((4, 5), (5,), 0, -1, torch.float32, 1e-4, id="2d-dim0-neg1-f32"),
    pytest.param((4, 5), (4,), 1, -1, torch.float32, 1e-4, id="2d-dim1-neg1-f32"),
    pytest.param((4, 5), (5,), 0, -3, torch.float32, 1e-4, id="2d-dim0-neg3-f32"),
    pytest.param((3, 4, 5), (3, 5), 1, -2, torch.float32, 1e-4, id="3d-dim1-neg2-f32"),
    pytest.param((3, 4, 5), (3, 4), -1, -1, torch.float32, 1e-4, id="3d-negdim-neg1-f32"),
    pytest.param((3, 4, 5), (3, 4), -1, 3, torch.float32, 1e-4, id="3d-negdim-pos-f32"),
    pytest.param((4, 5), (5,), 0, -1, torch.bfloat16, 5e-2, id="2d-dim0-neg1-bf16"),
    pytest.param((4, 5), (5,), 0, 2, torch.float32, 1e-4, id="2d-dim0-pos-f32"),
    pytest.param((4, 5), (4,), 1, 0, torch.float32, 1e-4, id="2d-dim1-zero-f32"),
]


class TestSelectScatterDecompositionE2E:
    """E2E tests verifying select_scatter with negative indices matches CPU results."""

    @pytest.mark.parametrize("input_shape,src_shape,dim,index,dtype,tol", SELECT_SCATTER_CONFIGS)
    def test_select_scatter(self, input_shape, src_shape, dim, index, dtype, tol):
        torch.manual_seed(0)
        x = torch.randn(input_shape, dtype=dtype)
        src = torch.randn(src_shape, dtype=dtype)

        def fn(t, s):
            return t.select_scatter(s, dim, index)

        with torch.no_grad():
            cpu_out = fn(x, src)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(x.to("neuron"), src.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)
