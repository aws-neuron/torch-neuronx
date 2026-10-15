"""E2E tests for the aten::logaddexp2 torch.compile decomposition.

Assert the routing (the neuron table entry runs, not a shadowing built-in) and
that the compiled forward and backward match CPU.
"""

import pytest
import torch
from torch._C import DispatchKey

from torch_neuronx.neuron_dynamo_backend.decompositions import neuron_decompositions

aten = torch.ops.aten


class TestLogaddexp2DecompositionE2E:
    def test_routing(self):
        # The neuron decomposition table must hold both overloads.
        assert aten.logaddexp2.default in neuron_decompositions
        assert aten.logaddexp2.out in neuron_decompositions
        # Nothing above __torch_dispatch__ may shadow the table entry.
        for ov in (aten.logaddexp2.default, aten.logaddexp2.out):
            assert DispatchKey.Autograd not in ov.py_kernels
            assert DispatchKey.CompositeImplicitAutograd not in ov.py_kernels

    @pytest.mark.skipif(torch.neuron.device_count() == 0, reason="No Neuron devices available")
    @pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
    def test_compiled_forward(self, dtype, tol):
        torch.manual_seed(0)
        a = (torch.randn(4, 32) * 20).to(dtype)
        b = (torch.randn(4, 32) * 20).to(dtype)

        def fn(x, y):
            return torch.logaddexp2(x, y)

        with torch.no_grad():
            cpu_out = fn(a, b)
            torch._dynamo.reset()
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(a.to("neuron"), b.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

    @pytest.mark.skipif(torch.neuron.device_count() == 0, reason="No Neuron devices available")
    def test_compiled_backward(self):
        torch.manual_seed(0)
        a = (torch.randn(4, 16) * 10).float()
        b = (torch.randn(4, 16) * 10).float()

        def fn(x, y):
            return torch.logaddexp2(x, y).sum()

        a_cpu = a.clone().requires_grad_(True)
        b_cpu = b.clone().requires_grad_(True)
        fn(a_cpu, b_cpu).backward()

        a_n = a.clone().to("neuron").requires_grad_(True)
        b_n = b.clone().to("neuron").requires_grad_(True)
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        compiled(a_n, b_n).backward()

        torch.testing.assert_close(a_n.grad.cpu(), a_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b_n.grad.cpu(), b_cpu.grad, rtol=1e-4, atol=1e-4)
