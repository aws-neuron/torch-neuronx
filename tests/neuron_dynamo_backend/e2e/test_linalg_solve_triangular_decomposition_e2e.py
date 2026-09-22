"""E2E test for linalg_solve_triangular forward + backward under backend="neuron"."""

import pytest
import torch


def _make_triangular(shape, upper, unitriangular, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    n = shape[-1]
    a = torch.randn(*shape, generator=g, dtype=dtype)
    a = torch.triu(a) if upper else torch.tril(a)
    if unitriangular:
        # Scale so ||I - U|| stays bounded; put garbage on the diagonal.
        a = a / (2.0 * n**0.5)
        garbage = torch.full(shape[:-1], 7.0, dtype=dtype)
        a = a - torch.diag_embed(torch.diagonal(a, dim1=-2, dim2=-1))
        a = a + torch.diag_embed(garbage)
    else:
        a = a + torch.eye(n, dtype=dtype) * 3.0
    return a


def _cpu_reference(a, b, *, upper, left, unitriangular):
    """CPU reference. Upcasts bf16 -> fp32 (no CPU bf16 kernel)."""
    if a.dtype == torch.bfloat16:
        out_f32 = torch.linalg.solve_triangular(
            a.float(), b.float(), upper=upper, left=left, unitriangular=unitriangular
        )
        return out_f32.to(torch.bfloat16)
    return torch.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unitriangular)


# (a_shape, b_shape, upper, left, unitriangular, dtype, tol, check_backward)
SOLVE_TRIANGULAR_CONFIGS = [
    # Motivating 4D shape: batch=2, heads=4, chunk_size=64, k=8.
    pytest.param(
        (2, 4, 64, 64),
        (2, 4, 64, 8),
        False,
        True,
        True,
        torch.float32,
        1e-4,
        True,
        id="4d-64-lower-unit-f32",
    ),
    pytest.param(
        (2, 4, 64, 64),
        (2, 4, 64, 8),
        False,
        True,
        True,
        torch.bfloat16,
        5e-2,
        False,
        id="4d-64-lower-unit-bf16",
    ),
    pytest.param(
        (2, 4, 64, 64),
        (2, 4, 64, 8),
        False,
        True,
        False,
        torch.float32,
        1e-4,
        True,
        id="4d-64-lower-nonunit-f32",
    ),
    # 2D, all 4 flag combos.
    pytest.param(
        (4, 4),
        (4, 3),
        False,
        True,
        False,
        torch.float32,
        1e-4,
        True,
        id="lower-nonunit-2d-f32",
    ),
    pytest.param(
        (4, 4),
        (4, 3),
        False,
        True,
        True,
        torch.float32,
        1e-4,
        True,
        id="lower-unit-2d-f32",
    ),
    pytest.param(
        (4, 4),
        (4, 3),
        True,
        True,
        False,
        torch.float32,
        1e-4,
        True,
        id="upper-nonunit-2d-f32",
    ),
    pytest.param(
        (4, 4),
        (4, 3),
        True,
        True,
        True,
        torch.float32,
        1e-4,
        True,
        id="upper-unit-2d-f32",
    ),
    # Batched.
    pytest.param(
        (3, 8, 8),
        (3, 8, 2),
        False,
        True,
        False,
        torch.float32,
        1e-4,
        True,
        id="lower-nonunit-3d-f32",
    ),
    pytest.param(
        (3, 8, 8),
        (3, 8, 2),
        True,
        True,
        True,
        torch.bfloat16,
        5e-2,
        False,
        id="upper-unit-3d-bf16",
    ),
    # left=False: X @ A = B, B is (..., k, n).
    pytest.param(
        (2, 4, 8, 8),
        (2, 4, 3, 8),
        False,
        False,
        False,
        torch.float32,
        1e-4,
        True,
        id="lower-nonunit-4d-left-false-f32",
    ),
    pytest.param(
        (2, 4, 8, 8),
        (2, 4, 3, 8),
        True,
        False,
        True,
        torch.float32,
        1e-4,
        True,
        id="upper-unit-4d-left-false-f32",
    ),
    # n=1 exercises the log2 guard.
    pytest.param(
        (1, 1),
        (1, 5),
        False,
        True,
        False,
        torch.float32,
        1e-4,
        True,
        id="n1-lower-nonunit-f32",
    ),
    pytest.param(
        (1, 1),
        (1, 5),
        True,
        True,
        True,
        torch.float32,
        1e-4,
        True,
        id="n1-upper-unit-f32",
    ),
    # Single-column RHS.
    pytest.param(
        (2, 16, 16),
        (2, 16, 1),
        False,
        True,
        True,
        torch.float32,
        1e-4,
        True,
        id="single-col-rhs-f32",
    ),
]


class TestLinalgSolveTriangularDecompositionE2E:
    @pytest.mark.parametrize(
        "a_shape,b_shape,upper,left,unitriangular,dtype,tol,check_backward",
        SOLVE_TRIANGULAR_CONFIGS,
    )
    def test_solve_triangular(
        self, a_shape, b_shape, upper, left, unitriangular, dtype, tol, check_backward
    ):
        torch.manual_seed(0)
        base_a = _make_triangular(a_shape, upper, unitriangular, dtype, seed=123)
        base_b = torch.randn(*b_shape, dtype=dtype)
        grad_out = torch.randn(*b_shape, dtype=dtype)

        def fn(a, b):
            return torch.linalg.solve_triangular(
                a, b, upper=upper, left=left, unitriangular=unitriangular
            )

        compiled = torch.compile(fn, backend="neuron", fullgraph=True)

        with torch.no_grad():
            cpu_out = _cpu_reference(
                base_a, base_b, upper=upper, left=left, unitriangular=unitriangular
            )
            neuron_out = compiled(base_a.to("neuron"), base_b.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

        # bf16 backward drifts past the CPU tolerance.
        if not check_backward:
            return

        def _neuron_fwd_bwd(f, a, b, g):
            a_g = a.detach().clone().to("neuron").requires_grad_(True)
            b_g = b.detach().clone().to("neuron").requires_grad_(True)
            out = f(a_g, b_g)
            out.backward(g.to("neuron"))
            return out, a_g.grad, b_g.grad

        def _cpu_fwd_bwd(a, b, g):
            src_dtype = a.dtype
            up = src_dtype == torch.bfloat16
            a_g = a.detach().clone().float() if up else a.detach().clone()
            a_g = a_g.requires_grad_(True)
            b_g = b.detach().clone().float() if up else b.detach().clone()
            b_g = b_g.requires_grad_(True)
            out = torch.linalg.solve_triangular(
                a_g, b_g, upper=upper, left=left, unitriangular=unitriangular
            )
            out.backward(g.float() if up else g)
            if up:
                return out.to(src_dtype), a_g.grad.to(src_dtype), b_g.grad.to(src_dtype)
            return out, a_g.grad, b_g.grad

        cpu_out, cpu_ga, cpu_gb = _cpu_fwd_bwd(base_a, base_b, grad_out)
        neuron_out, neuron_ga, neuron_gb = _neuron_fwd_bwd(compiled, base_a, base_b, grad_out)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)
        torch.testing.assert_close(neuron_ga.cpu(), cpu_ga, rtol=tol, atol=tol)
        torch.testing.assert_close(neuron_gb.cpu(), cpu_gb, rtol=tol, atol=tol)
