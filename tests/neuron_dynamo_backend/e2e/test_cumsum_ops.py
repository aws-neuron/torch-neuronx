"""
E2E tests for the static-N cumsum lowering (torch.compile / neuron backend).
"""

import logging

import pytest
import torch

logger = logging.getLogger(__name__)


CUMSUM_1D_CASES = [
    pytest.param(128, True, id="128-numerics"),
    pytest.param(32768, False, id="32768-crosses-4gb"),
    pytest.param(49152, False, id="49152-production-regression"),
]


class TestCumsumStaticNRegression:
    """Regression for the static-N cumsum tril lowering (TorchToStablehlo).

    The static branch of ``ConvertAtenOp<AtenCumsumOp>`` used to build the
    [N, N] triangular ones matrix as a dense constant. For the Qwen3-VL /
    HiDream-O1-Image ``mask.flatten().to(f32).cumsum(0) - 1`` path (N=49152),
    the [N, N] f32 tril intermediate is 49152 * 49152 * 4 = 9.66 GB, well
    past the compiler input parse limit and the per-core HBM budget.
    The fix builds the tril via ``iota + compare + convert``, so the
    compiler can fuse the tril-@-matmul and never materialize a physical
    [N, N] intermediate.

    The regression signal is end-to-end run on device, not compile-only:
    the intermediate must be fused away at runtime for the graph to survive.
    """

    @pytest.mark.parametrize("n,checks_numerics", CUMSUM_1D_CASES)
    def test_cumsum_1d_static_shape(self, n, checks_numerics):
        """Static 1-D cumsum across three N.

        N=128 is a small-shape numerics sanity check against CPU. N=32768 is
        the smallest 1-D shape whose f32 tril intermediate crosses 4 GB
        (32768**2 * 4 = 4.29 GB). N=49152 matches the exact Qwen3-VL /
        HiDream-O1-Image failure shape (9.66 GB intermediate). For the large
        N, running to completion is the assertion; a 49152-element CPU
        compare is slow and adds no regression signal beyond survival.
        """

        def fn(x):
            return torch.cumsum(x, dim=0)

        x_cpu = torch.randn(n, dtype=torch.float32)
        with torch.no_grad():
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x_cpu.to("neuron"))

        assert neuron_out.shape == (n,)
        assert neuron_out.dtype == torch.float32
        if checks_numerics:
            cpu_out = fn(x_cpu)
            torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-5, atol=1e-5)

    def test_cumsum_masked_scatter_index_pattern(self):
        """The Qwen3-VL / HiDream-O1-Image ``get_placeholder_mask`` fragment.

        ``mask.flatten().to(f32).cumsum(0) - 1`` computes scatter indices
        from a boolean mask. This is the exact graph that hit exit 70;
        the compiled graph must run end-to-end on device.
        """

        def fn(mask):
            return mask.flatten().to(torch.float32).cumsum(0) - 1.0

        torch.manual_seed(0)
        mask_cpu = torch.rand(1, 96, 512) > 0.5
        with torch.no_grad():
            cpu_out = fn(mask_cpu)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(mask_cpu.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)


_CUMMAXMIN_SCENARIOS = [
    pytest.param(lambda: torch.randn(3, 5), id="random"),
    pytest.param(lambda: torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]]), id="ties"),
    pytest.param(lambda: torch.tensor([[1.0, float("nan"), 2.0, 0.5, 3.0]]), id="nan"),
]


class TestCummaxCumminCompile:
    """cummax/cummin on the compile path decompose to the Hillis-Steele scan."""

    @pytest.mark.parametrize("fn", [torch.cummax, torch.cummin], ids=["cummax", "cummin"])
    @pytest.mark.parametrize("make_x", _CUMMAXMIN_SCENARIOS)
    def test_compile(self, fn, make_x):
        def g(t):
            v, idx = fn(t, 1)
            return v + 1.0, idx

        x = make_x()
        cpu_v, cpu_i = g(x)
        torch._dynamo.reset()
        neuron_v, neuron_i = torch.compile(g, backend="neuron", fullgraph=True)(x.to("neuron"))
        torch.testing.assert_close(neuron_v.cpu(), cpu_v, equal_nan=True)
        torch.testing.assert_close(neuron_i.cpu(), cpu_i)

    @pytest.mark.parametrize("fn", [torch.cummax, torch.cummin], ids=["cummax", "cummin"])
    def test_compile_out(self, fn):
        # Only cummax.default/cummin.default register a decomposition, so the .out
        # overload must be functionalized to .default to hit it. This pins that
        # behavior; a regression would give the backend an unregistered .out.
        def g(t):
            v = torch.empty_like(t)
            i = torch.empty(t.shape, dtype=torch.int64, device=t.device)
            fn(t, 1, out=(v, i))
            return v, i

        x = torch.randn(3, 5)
        cpu_v, cpu_i = g(x)
        torch._dynamo.reset()
        neuron_v, neuron_i = torch.compile(g, backend="neuron", fullgraph=True)(x.to("neuron"))
        torch.testing.assert_close(neuron_v.cpu(), cpu_v)
        torch.testing.assert_close(neuron_i.cpu(), cpu_i)

    @pytest.mark.parametrize("fn", [torch.cummax, torch.cummin], ids=["cummax", "cummin"])
    def test_compile_backward(self, fn):
        def g(t):
            v, _ = fn(t, 1)
            return v

        x = torch.randn(3, 5, requires_grad=True)
        grad = torch.randn(3, 5)
        (g(x) * grad).sum().backward()
        cpu_grad = x.grad.clone()

        torch._dynamo.reset()
        xn = x.detach().to("neuron").requires_grad_(True)
        compiled = torch.compile(g, backend="neuron", fullgraph=True)
        (compiled(xn) * grad.to("neuron")).sum().backward()
        torch.testing.assert_close(xn.grad.cpu(), cpu_grad)
