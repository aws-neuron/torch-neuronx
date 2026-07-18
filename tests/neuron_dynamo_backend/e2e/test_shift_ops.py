"""
E2E tests for bitwise shift ops (__rshift__ and __lshift__, Tensor and Scalar variants).
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
    pytest.param(torch.int32, id="i32"),
    pytest.param(torch.int64, id="i64"),
]

SHIFT_AMOUNTS = [1, 4, 8]


class TestRshiftTensorE2E:
    """E2E tests for aten.__rshift__.Tensor (tensor >> tensor)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", SHIFT_AMOUNTS)
    def test_rshift_tensor(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)
        # Shift amount must be non-negative and < bitwidth
        s = torch.full(shape, shift, dtype=dtype)

        def fn(a, b):
            return a >> b

        with torch.no_grad():
            cpu_out = fn(x, s)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), s.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
