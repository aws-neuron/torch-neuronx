"""E2E tests for aten::baddbmm under torch.compile(backend="neuron").

Eager and torch.compile lower through different machinery: eager goes via
``@register_aten`` in ``python_ops/torch_mlir/ops/linear_algebra.py``, while
the compile path decomposes ``torch.aten.baddbmm`` into ``bmm``/``mul``/``add``
in torch-mlir's ``DecomposeComplexOps.cpp``. An op can work in one and hard-fail
in the other, so both need coverage — the eager side lives in
``tests/python_ops/test_baddbmm.py``.
"""

import pytest
import torch

# (batch, n, m, p) — out = beta * input + alpha * (batch1[b,n,m] @ batch2[b,m,p])
SHAPES = [
    pytest.param((1, 4, 6, 5), id="batch1"),
    pytest.param((3, 4, 6, 5), id="small"),
    pytest.param((8, 16, 32, 8), id="medium"),
    pytest.param((2, 128, 64, 128), id="large"),
]

DTYPES = [
    pytest.param(torch.float32, id="f32"),
    pytest.param(torch.bfloat16, id="bf16"),
]

TOL = {torch.float32: 1e-3, torch.bfloat16: 5e-2}


class TestBaddbmmE2E:
    """torch.compile coverage for the default overload."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_baddbmm_default(self, shape):
        b, n, m, p = shape
        torch.manual_seed(42)
        inp = torch.randn(b, n, p)
        b1 = torch.randn(b, n, m)
        b2 = torch.randn(b, m, p)

        def fn(i, x, y):
            return torch.baddbmm(i, x, y)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_baddbmm_dtypes(self, dtype):
        torch.manual_seed(42)
        inp = torch.randn(2, 4, 8, dtype=dtype)
        b1 = torch.randn(2, 4, 6, dtype=dtype)
        b2 = torch.randn(2, 6, 8, dtype=dtype)

        def fn(i, x, y):
            return torch.baddbmm(i, x, y)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        assert neuron_out.dtype == cpu_out.dtype
        tol = TOL[dtype]
        torch.testing.assert_close(neuron_out.cpu().float(), cpu_out.float(), rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "beta,alpha",
        [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 2.0), (-1.0, 3.0)],
    )
    def test_baddbmm_scalar_beta_alpha(self, beta, alpha):
        """Baked scalar beta/alpha, the form transformer code actually emits."""
        torch.manual_seed(42)
        inp = torch.randn(3, 4, 5)
        b1 = torch.randn(3, 4, 6)
        b2 = torch.randn(3, 6, 5)

        def fn(i, x, y):
            return torch.baddbmm(i, x, y, beta=beta, alpha=alpha)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    def test_baddbmm_method_form(self):
        """`Tensor.baddbmm` (method) must lower the same as the functional form."""
        torch.manual_seed(42)
        inp = torch.randn(2, 3, 4)
        b1 = torch.randn(2, 3, 5)
        b2 = torch.randn(2, 5, 4)

        def fn(i, x, y):
            return i.baddbmm(x, y, beta=0.5, alpha=2.0)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "input_shape", [(1, 4, 5), (3, 1, 5), (4, 5), (5,)], ids=["batch1", "rows1", "2d", "1d"]
    )
    def test_baddbmm_broadcast_input(self, input_shape):
        torch.manual_seed(42)
        inp = torch.randn(*input_shape)
        b1 = torch.randn(3, 4, 6)
        b2 = torch.randn(3, 6, 5)

        def fn(i, x, y):
            return torch.baddbmm(i, x, y, beta=0.5, alpha=2.0)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        assert neuron_out.shape == cpu_out.shape
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    def test_baddbmm_attention_pattern(self):
        """Attention scores: bias + scale * (q @ k^T) — the shape that motivates
        the op in transformer models."""
        torch.manual_seed(42)
        batch, seq, head_dim = 4, 16, 8
        q = torch.randn(batch, seq, head_dim)
        k = torch.randn(batch, seq, head_dim)
        bias = torch.randn(1, seq, seq)
        scale = 1.0 / (head_dim**0.5)

        def fn(bias_t, q_t, k_t):
            return torch.baddbmm(bias_t, q_t, k_t.transpose(-2, -1), alpha=scale)

        with torch.no_grad():
            cpu_out = fn(bias, q, k)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(bias.to("neuron"), q.to("neuron"), k.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("beta", [1.0, 0.5])
    def test_baddbmm_zero_contraction_dim(self, beta):
        """A zero contraction dim leaves the output non-empty: beta * input.

        The compile path must not zero-fill it either.
        """
        torch.manual_seed(42)
        inp = torch.randn(2, 3, 4)
        b1 = torch.randn(2, 3, 0)
        b2 = torch.randn(2, 0, 4)

        def fn(i, x, y):
            return torch.baddbmm(i, x, y, beta=beta)

        with torch.no_grad():
            cpu_out = fn(inp, b1, b2)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(inp.to("neuron"), b1.to("neuron"), b2.to("neuron"))

        assert neuron_out.shape == (2, 3, 4)
        assert torch.count_nonzero(neuron_out.cpu()) > 0, "beta * input term was dropped"
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3)

    def test_baddbmm_backward(self):
        """Gradients through the compiled graph match CPU."""
        torch.manual_seed(42)
        inp = torch.randn(2, 3, 4, requires_grad=True)
        b1 = torch.randn(2, 3, 5, requires_grad=True)
        b2 = torch.randn(2, 5, 4, requires_grad=True)
        n_inp = inp.detach().to("neuron").requires_grad_()
        n_b1 = b1.detach().to("neuron").requires_grad_()
        n_b2 = b2.detach().to("neuron").requires_grad_()

        def fn(i, x, y):
            return torch.baddbmm(i, x, y, beta=0.5, alpha=2.0).sum()

        fn(inp, b1, b2).backward()
        torch.compile(fn, backend="neuron", fullgraph=True)(n_inp, n_b1, n_b2).backward()

        for name, got, expected in (
            ("input", n_inp.grad, inp.grad),
            ("batch1", n_b1.grad, b1.grad),
            ("batch2", n_b2.grad, b2.grad),
        ):
            torch.testing.assert_close(
                got.cpu(), expected, rtol=1e-3, atol=1e-3, msg=f"grad {name} mismatch"
            )
