# ruff: noqa: N806

"""End-to-end compile tests for max_pool2d backward and max_unpool3d.

- max_pool2d_with_indices_backward: the PyTorch composite default lowered to a
  graph that crashed neuronx-cc (InsertIOTransposes); we register an explicit
  scatter-add decomp that compiles cleanly.
- max_unpool3d: lowered via PyTorch's builtin decomposition (enabled in the table).
"""

import pytest
import torch

_TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2}


def _close(a, b, dtype):
    tol = _TOL.get(dtype, 1e-4)
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestMaxPoolUnpoolDecompositionsE2E:
    # Unpadded cases go through the full autograd pipeline (forward picks the
    # argmax, our decomp handles the backward). Padding is covered separately by
    # test_max_pool2d_backward_padded_e2e: under padding the -inf-padded windows
    # produce ties whose argmax can break differently across backends, so an
    # end-to-end 0/1 grad-mask comparison is brittle there — the padded test
    # instead feeds fixed indices to isolate the backward decomp.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((1, 2, 8, 8), 2, 2, 0),  # non-overlapping
            ((2, 3, 7, 7), 3, 1, 0),  # overlapping windows
        ],
    )
    def test_max_pool2d_backward_e2e(self, shape, kernel, stride, padding, dtype):
        def fn(x):
            return torch.nn.functional.max_pool2d(x, kernel, stride, padding).sum()

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_cpu = torch.randn(*shape, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        fn(x_cpu).backward()
        compiled_fn(x_neuron).backward()
        _close(x_neuron.grad.cpu(), x_cpu.grad, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_max_pool2d_backward_padded_e2e(self, dtype):
        """Padded max_pool2d backward, tested directly with fixed indices.

        The indices come from the CPU forward, so both paths scatter to the same
        positions — this validates the backward decomp's handling of padded
        (unpadded-frame) indices without depending on cross-backend argmax
        tie-breaking in the forward.
        """
        torch.manual_seed(0)
        k, s, p, d = [2, 2], [2, 2], [1, 1], [1, 1]
        x = torch.randn(1, 2, 8, 8, dtype=torch.float32)
        _, idx = torch.ops.aten.max_pool2d_with_indices.default(x, k, s, p, d, False)
        grad_output = torch.randn(*idx.shape, dtype=torch.float32)

        def fn(grad_output, x, idx):
            return torch.ops.aten.max_pool2d_with_indices_backward.default(
                grad_output, x, k, s, p, d, False, idx
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        ref = fn(grad_output, x, idx)
        out = compiled_fn(
            grad_output.to(dtype).to("neuron"), x.to(dtype).to("neuron"), idx.to("neuron")
        )
        _close(out.cpu().float(), ref.float() if dtype == torch.bfloat16 else ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_max_unpool3d_e2e(self, dtype):
        # Build pooling indices from a max_pool3d_with_indices, then unpool.
        torch.manual_seed(0)
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
        vals, idx = torch.ops.aten.max_pool3d_with_indices.default(
            x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False
        )

        def fn(v, i):
            return torch.nn.functional.max_unpool3d(
                v, i, [2, 2, 2], [2, 2, 2], [0, 0, 0], [4, 4, 4]
            )

        compiled_fn = torch.compile(fn, backend="neuron")
        ref = fn(vals, idx)
        out = compiled_fn(vals.to(dtype).to("neuron"), idx.to("neuron"))
        _close(out.cpu().float(), ref.float() if dtype == torch.bfloat16 else ref, dtype)
