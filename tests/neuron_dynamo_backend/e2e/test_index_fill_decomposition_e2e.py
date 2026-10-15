"""E2E tests for the aten::index_fill torch.compile decomposition.

index_fill decomposes to expand + index_copy (a functional scatter). These tests
assert the compile-path routing and the numerics for the forward and the
backward. The backward w.r.t. self is grad.index_fill(dim, index, 0), so it runs
through the same decomposition.
"""

import pytest
import torch
from torch._C import DispatchKey

import torch_neuronx

# (shape, dim, index, dtype, tol)
CONFIGS = [
    pytest.param((64,), 0, [0, 5, 9], torch.float32, 1e-5, id="1d-f32"),
    pytest.param((8, 16), 1, [0, 3, 7], torch.float32, 1e-5, id="2d-f32"),
    pytest.param((2, 3, 8), 0, [0, 1], torch.float32, 1e-5, id="3d-f32"),
    pytest.param((64,), 0, [0, 5], torch.bfloat16, 5e-2, id="1d-bf16"),
    pytest.param((4, 6), 1, [0, 2], torch.int32, 1e-5, id="2d-i32"),
]


class TestIndexFillDecompositionE2E:
    """Verify the index_fill decomposition routes correctly and matches CPU."""

    def test_index_fill_decomposition_routing(self):
        """Test no Autograd/CompositeImplicitAutograd shadows the decomposition.

        A decomposition table is only read below __torch_dispatch__. If PyTorch
        published index_fill as py_impl(Autograd) or py_impl(CompositeImplicitAutograd),
        the Neuron decomposition would be dead code (gotcha #5).
        """
        import torch_neuronx.neuron_dynamo_backend.decompositions as d

        assert torch.ops.aten.index_fill.int_Scalar in d.neuron_decompositions
        assert torch.ops.aten.index_fill.int_Tensor in d.neuron_decompositions

        for op in (torch.ops.aten.index_fill.int_Scalar, torch.ops.aten.index_fill.int_Tensor):
            assert DispatchKey.Autograd not in op.py_kernels
            assert DispatchKey.CompositeImplicitAutograd not in op.py_kernels

    @pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
    @pytest.mark.parametrize("shape,dim,index,dtype,tol", CONFIGS)
    def test_index_fill_compile_matches_cpu(self, shape, dim, index, dtype, tol):
        """Test the compiled index_fill forward matches CPU."""
        torch.manual_seed(0)
        if dtype == torch.int32:
            x = torch.randint(-8, 9, shape, dtype=dtype)
        else:
            x = torch.randn(shape).to(dtype)
        idx = torch.tensor(index, dtype=torch.int64)

        def fn(t, i):
            return t.index_fill(dim, i, 3.0)

        torch._dynamo.reset()
        with torch.no_grad():
            cpu_out = fn(x, idx)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), idx.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

    @pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
    def test_index_fill_compile_backward(self):
        """Test the compiled index_fill backward matches CPU.

        The self gradient is grad.index_fill(dim, index, 0), so it routes through
        the same decomposition. int_Scalar has no value gradient, so this avoids
        the aten._unique value gradient (dynamic output, not supported on compile).
        """
        idx = torch.tensor([0, 2, 4], dtype=torch.int64)

        def fn(t, i):
            return (t.index_fill(0, i, 3.0) * 2).sum()

        def run(compiled, device):
            torch.manual_seed(0)
            x = torch.randn(6, 5, device=device).detach().requires_grad_(True)
            out = compiled(x, idx.to(device))
            out.backward()
            return x.grad

        torch._dynamo.reset()
        cpu_grad = run(fn, "cpu")
        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_grad = run(compiled_fn, "neuron")

        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad, rtol=1e-5, atol=1e-5)
