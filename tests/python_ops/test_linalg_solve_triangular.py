"""Eager tests for aten::linalg_solve_triangular on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


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


# (a_shape, b_shape, upper, left, unitriangular, dtype, tol)
SOLVE_TRIANGULAR_CONFIGS = [
    pytest.param(
        (2, 4, 64, 64),
        (2, 4, 64, 8),
        False,
        True,
        True,
        torch.float32,
        1e-4,
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
        id="4d-64-lower-nonunit-f32",
    ),
    pytest.param(
        (4, 4),
        (4, 3),
        False,
        True,
        False,
        torch.float32,
        1e-4,
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
        id="upper-unit-2d-f32",
    ),
    pytest.param(
        (3, 8, 8),
        (3, 8, 2),
        False,
        True,
        False,
        torch.float32,
        1e-4,
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
        id="upper-unit-3d-bf16",
    ),
    pytest.param(
        (2, 4, 8, 8),
        (2, 4, 3, 8),
        False,
        False,
        False,
        torch.float32,
        1e-4,
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
        id="upper-unit-4d-left-false-f32",
    ),
    pytest.param(
        (1, 1),
        (1, 5),
        False,
        True,
        False,
        torch.float32,
        1e-4,
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
        id="n1-upper-unit-f32",
    ),
    pytest.param(
        (2, 16, 16),
        (2, 16, 1),
        False,
        True,
        True,
        torch.float32,
        1e-4,
        id="single-col-rhs-f32",
    ),
]


class TestLinalgSolveTriangular:
    @pytest.mark.parametrize(
        "a_shape,b_shape,upper,left,unitriangular,dtype,tol",
        SOLVE_TRIANGULAR_CONFIGS,
    )
    def test_solve_triangular(self, a_shape, b_shape, upper, left, unitriangular, dtype, tol):
        a_cpu = _make_triangular(a_shape, upper, unitriangular, dtype, seed=123)
        b_cpu = torch.randn(*b_shape, dtype=dtype)

        with track_neuron_ops():
            a_n = a_cpu.to("neuron")
            b_n = b_cpu.to("neuron")

            out_neuron = torch.linalg.solve_triangular(
                a_n, b_n, upper=upper, left=left, unitriangular=unitriangular
            )
            out_cpu = _cpu_reference(
                a_cpu, b_cpu, upper=upper, left=left, unitriangular=unitriangular
            )

            torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=tol, atol=tol)
            assert_op_runs_on_neuron("aten::linalg_solve_triangular")

    # Autograd re-invokes the op on transposed inputs; forward-only misses that path.
    @pytest.mark.parametrize(
        "a_shape,b_shape,upper,left,unitriangular,dtype,tol",
        [c for c in SOLVE_TRIANGULAR_CONFIGS if c.values[5] == torch.float32],
    )
    def test_solve_triangular_backward(
        self, a_shape, b_shape, upper, left, unitriangular, dtype, tol
    ):
        base_a = _make_triangular(a_shape, upper, unitriangular, dtype, seed=123)
        base_b = torch.randn(*b_shape, dtype=dtype)
        grad_out = torch.randn(*b_shape, dtype=dtype)

        def fwd_bwd(a, b, g):
            a_g = a.detach().clone().requires_grad_(True)
            b_g = b.detach().clone().requires_grad_(True)
            out = torch.linalg.solve_triangular(
                a_g, b_g, upper=upper, left=left, unitriangular=unitriangular
            )
            out.backward(g)
            return out, a_g.grad, b_g.grad

        with track_neuron_ops():
            n_out, n_ga, n_gb = fwd_bwd(
                base_a.to("neuron"), base_b.to("neuron"), grad_out.to("neuron")
            )
            assert_op_runs_on_neuron("aten::linalg_solve_triangular")

        c_out, c_ga, c_gb = fwd_bwd(base_a, base_b, grad_out)
        torch.testing.assert_close(n_out.cpu(), c_out, rtol=tol, atol=tol)
        torch.testing.assert_close(n_ga.cpu(), c_ga, rtol=tol, atol=tol)
        torch.testing.assert_close(n_gb.cpu(), c_gb, rtol=tol, atol=tol)
