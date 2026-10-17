"""End-to-end compile tests for gcd / lcm, lowered via fixed-count unrolled Euclid.

gcd has no decomposition in the stack and no torch-mlir lowering; we register a
Python decomp that unrolls Euclid a fixed (dtype-width-bounded) number of steps over
already-lowering elementwise ops. lcm = |a // gcd(a,b) * b|.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestGcdLcmLoweringE2E:
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int16, torch.int8, torch.int64])
    def test_gcd_e2e(self, dtype):
        def fn(x, y):
            return torch.gcd(x, y)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        info = torch.iinfo(dtype)
        # cap the range at 2^31-1 (and the dtype's own max) so int64 stays in 32 bits
        hi = min(info.max // 2, 2**31 - 1)
        lo = max(info.min // 2, -(2**31) + 1)
        x = torch.randint(lo, hi, (4, 8)).to(dtype)
        y = torch.randint(lo, hi, (4, 8)).to(dtype)
        x[0, 0] = 0  # gcd(0, n) == n
        result = compiled_fn(x.to("neuron"), y.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x, y))

    @pytest.mark.parametrize(
        "dtype,hi", [(torch.int32, 100), (torch.int16, 100), (torch.int8, 12), (torch.int64, 100)]
    )
    def test_lcm_e2e(self, dtype, hi):
        def fn(x, y):
            return torch.lcm(x, y)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        x = torch.randint(1, hi, (4, 8)).to(dtype)
        y = torch.randint(1, hi, (4, 8)).to(dtype)
        result = compiled_fn(x.to("neuron"), y.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x, y))

    def test_lcm_exact_division_e2e(self):
        def fn(x, y):
            return torch.lcm(x, y)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        x = torch.tensor([[293, 41, 656, 100]], dtype=torch.int32)
        y = torch.tensor([[293, 287, 164, 10]], dtype=torch.int32)
        result = compiled_fn(x.to("neuron"), y.to("neuron"))
        torch.testing.assert_close(result.cpu(), fn(x, y))
