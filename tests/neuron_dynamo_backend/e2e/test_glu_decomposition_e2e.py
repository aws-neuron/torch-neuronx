"""E2E tests for the aten::glu torch.compile decomposition.

aten.glu is added to the Neuron decomposition table (decompositions.py) so the
compile path decomposes it to primitives (a chunk, a sigmoid, a multiply) that
lower cleanly to StableHLO. The backward reaches aten.glu_backward, which is
also in the table. These tests assert the routing and the forward/backward
numerics.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch._C import DispatchKey

import torch_neuronx

# (shape, dim, dtype, tol)
CONFIGS = [
    pytest.param((64, 8), 1, torch.float32, 1e-5, id="2d-f32"),
    pytest.param((2, 3, 8), -1, torch.float32, 1e-5, id="3d-f32"),
    pytest.param((64, 8), 1, torch.float16, 5e-3, id="2d-f16"),
    pytest.param((64, 8), 1, torch.bfloat16, 5e-2, id="2d-bf16"),
]


class TestGluDecompositionE2E:
    """Verify the glu decomposition routes correctly and matches CPU."""

    def test_glu_decomposition_routing(self):
        """Test no Autograd/CompositeImplicitAutograd shadows the decomposition.

        A decomposition table is only read below __torch_dispatch__. A published
        py_impl(Autograd)/py_impl(CompositeImplicitAutograd) would make the Neuron
        decomposition dead code (gotcha #5).
        """
        import torch_neuronx.neuron_dynamo_backend.decompositions as d

        assert torch.ops.aten.glu.default in d.neuron_decompositions
        assert torch.ops.aten.glu.out in d.neuron_decompositions

        op = torch.ops.aten.glu.default
        assert DispatchKey.Autograd not in op.py_kernels
        assert DispatchKey.CompositeImplicitAutograd not in op.py_kernels

    @pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
    @pytest.mark.parametrize("shape,dim,dtype,tol", CONFIGS)
    def test_glu_compile_matches_cpu(self, shape, dim, dtype, tol):
        """Test the compiled glu forward matches CPU across dtypes and shapes."""
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=dtype)

        def fn(t):
            return F.glu(t, dim=dim)

        torch._dynamo.reset()
        with torch.no_grad():
            cpu_out = fn(x)
            neuron_out = torch.compile(fn, backend="neuron", fullgraph=True)(x.to("neuron"))
        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

    @pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
    @pytest.mark.parametrize(
        "dtype,tol",
        [(torch.float32, 1e-3), (torch.bfloat16, 1e-2)],
    )
    def test_glu_compile_backward_matches_cpu(self, dtype, tol):
        """The compiled autograd backward reaches glu_backward and matches CPU.

        fullgraph=True fails if the backward graph breaks.
        """
        gen = torch.Generator().manual_seed(0)
        xc = torch.randn(64, 8, dtype=dtype, generator=gen).requires_grad_()
        x = xc.detach().to("neuron").requires_grad_()

        def fn(t):
            return F.glu(t, dim=1)

        torch._dynamo.reset()
        torch.compile(fn, backend="neuron", fullgraph=True)(x).pow(2).sum().backward()
        fn(xc).pow(2).sum().backward()

        assert x.grad is not None, "no gradient propagated to the input"
        torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=tol, atol=tol)
