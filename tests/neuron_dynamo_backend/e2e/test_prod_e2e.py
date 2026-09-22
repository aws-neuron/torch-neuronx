"""E2E tests for aten::prod under torch.compile(backend="neuron").

prod lowers as a multiply-reduction (StableHLO reduce). fullgraph=True asserts
the whole call lowers on-device without a graph break. Covers full and dim
reductions across dtypes; matched against CPU. Integer inputs use small values
so the product stays within Neuron's int32 in-kernel accumulation range.
"""

import pytest
import torch

FLOAT_TOL = {
    torch.float32: {},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-3},
    torch.float16: {"rtol": 1.6e-2, "atol": 1e-3},
}


class TestProdE2E:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64])
    def test_full_reduction(self, dtype):
        torch.manual_seed(0)
        if dtype.is_floating_point:
            x = (torch.rand(64) + 0.5).to(dtype)
            tol = FLOAT_TOL[dtype]
        else:
            x = torch.arange(1, 7, dtype=dtype)  # product 720, fits int32
            tol = {}

        def fn(t):
            return torch.prod(t)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(x.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, **tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_dim_reduction(self, dtype, keepdim):
        torch.manual_seed(0)
        if dtype.is_floating_point:
            x = (torch.rand(8, 16) + 0.5).to(dtype)
            tol = FLOAT_TOL[dtype]
        else:
            x = torch.randint(1, 3, (8, 16), dtype=dtype)  # per-row product <= 2^16, fits int32
            tol = {}

        def fn(t):
            return torch.prod(t, dim=1, keepdim=keepdim)

        with torch.no_grad():
            cpu_out = fn(x)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(x.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, **tol)
