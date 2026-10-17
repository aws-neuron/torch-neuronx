"""E2E tests for the reflection_pad backward decompositions on the compile path."""

import pytest
import torch
from torch._C import DispatchKey

from torch_neuronx.neuron_dynamo_backend.decompositions import neuron_decompositions

aten = torch.ops.aten

# (input shape, padding, forward callable name)
REFLECTION_PAD_CONFIGS = [
    pytest.param((1, 3, 8), (2, 2), 1, id="1d"),
    pytest.param((2, 4, 16), (3, 1), 1, id="1d-asym"),
    pytest.param((1, 3, 8, 8), (1, 1, 1, 1), 2, id="2d"),
    pytest.param((2, 4, 16, 16), (2, 2, 3, 3), 2, id="2d-asym"),
    pytest.param((1, 2, 4, 4, 4), (1, 1, 1, 1, 1, 1), 3, id="3d"),
]

_BWD_OPS = [
    aten.reflection_pad1d_backward.default,
    aten.reflection_pad2d_backward.default,
    aten.reflection_pad3d_backward.default,
]


class TestReflectionPadDecompositionE2E:
    def test_routing(self):
        """The decomposition table owns the backward, so no Autograd/composite kernel shadows it."""
        for op in _BWD_OPS:
            assert op in neuron_decompositions
            assert DispatchKey.Autograd not in op.py_kernels
            assert DispatchKey.CompositeImplicitAutograd not in op.py_kernels

    @pytest.mark.parametrize("shape,padding,dim", REFLECTION_PAD_CONFIGS)
    def test_reflection_pad_backward_via_compile(self, shape, padding, dim):
        """AOTAutograd derives the backward from the forward decomposition; it must not
        graph-break and must match the CPU gradient."""
        torch._dynamo.reset()

        def fn(t):
            y = torch.nn.functional.pad(t, padding, mode="reflect")
            return (y * y).sum()

        x_cpu = torch.randn(shape, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        compiled = torch.compile(fn, backend="neuron")
        compiled(x_neuron).backward()
        fn(x_cpu).backward()
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
