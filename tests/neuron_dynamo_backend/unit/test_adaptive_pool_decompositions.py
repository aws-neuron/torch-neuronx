# ruff: noqa: N806

"""End-to-end compile tests for adaptive pooling ops, lowered via custom decomps.

These ops fail torch-mlir legalization in the stack. We register Python decomps:
- _adaptive_avg_pool2d/3d_backward: separable per-dim backward matmuls (handles
  non-divisible output sizes the native lowering rejects).
- adaptive_max_pool2d/3d (+ backward): per-output window gather + masked max over
  non-uniform adaptive windows; backward scatter-adds grads to the winners.

Each op is driven through torch.compile(backend="neuron") and compared against
the eager CPU reference. Adaptive-max / adaptive-avg lack CPU bfloat16 kernels for
some variants, so the reference is computed in float32 and the neuron path runs in
the target dtype.
"""

import pytest
import torch

_TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2}


def _close(a, b, dtype):
    tol = _TOL.get(dtype, 1e-4)
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestAdaptivePoolDecompositionsE2E:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("out_hw", [(4, 4), (3, 3)])  # incl. non-divisible (3,3)
    def test_adaptive_avg_pool2d_backward_e2e(self, out_hw, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, out_hw).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu().float(), x_cpu.grad, dtype)

    # Divisible output sizes only: this routes through the forward adaptive_avg_pool3d
    # (via autograd), which emits a torch.runtime.assert on non-divisible 3d sizes that
    # does not yet legalize. The backward decomp itself handles non-divisible sizes.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("out_dhw", [(2, 4, 4), (1, 2, 2)])
    def test_adaptive_avg_pool3d_backward_e2e(self, out_dhw, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_avg_pool3d(x, out_dhw).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu().float(), x_cpu.grad, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("out_hw", [(4, 4), (3, 3)])
    def test_adaptive_max_pool2d_e2e(self, out_hw, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_max_pool2d(x, out_hw)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron")
        _close(compiled_fn(x_neuron).cpu().float(), fn(x_cpu), dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("out_dhw", [(2, 4, 4), (2, 3, 3)])
    def test_adaptive_max_pool3d_e2e(self, out_dhw, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_max_pool3d(x, out_dhw)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron")
        _close(compiled_fn(x_neuron).cpu().float(), fn(x_cpu), dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaptive_max_pool2d_backward_e2e(self, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_max_pool2d(x, (4, 4)).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 8, 8, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu().float(), x_cpu.grad, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaptive_max_pool3d_backward_e2e(self, dtype):
        def fn(x):
            return torch.nn.functional.adaptive_max_pool3d(x, (2, 4, 4)).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to(dtype).to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu().float(), x_cpu.grad, dtype)
