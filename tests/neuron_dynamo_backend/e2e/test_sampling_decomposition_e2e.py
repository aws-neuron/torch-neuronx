"""End-to-end tests for the grid_sampler decompositions under torch.compile.

Checks numerics (the compiled graph agrees with CPU, forward and backward) and
routing (the grid_sampler decompositions are in the Neuron table). AOTAutograd
derives the backward from the traced forward decomposition, so a differentiable
grid_sample compiles end-to-end once the forward decomposition uses lowered
primitives.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx

aten = torch.ops.aten

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)

MODES_2D = ["bilinear", "nearest", "bicubic"]
MODES_3D = ["bilinear", "nearest"]  # bicubic is not defined for 3D by aten
PADDING = ["zeros", "border", "reflection"]


def _compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="neuron", fullgraph=True)


# --- routing ---
class TestDecompositionRouting:
    @pytest.mark.parametrize(
        "op",
        [
            aten.grid_sampler_2d.default,
            aten.grid_sampler_2d_backward.default,
            aten.grid_sampler_3d.default,
            aten.grid_sampler_3d_backward.default,
        ],
        ids=str,
    )
    def test_registered_in_the_neuron_table(self, op):
        from torch_neuronx.compiler.api import get_neuron_decompositions

        assert op in get_neuron_decompositions(), f"{op} missing from the Neuron table"


# --- forward numerics ---
class TestForward:
    @pytest.mark.parametrize("mode,padding", list(itertools.product(MODES_2D, PADDING)))
    @pytest.mark.parametrize("align", [False, True])
    def test_2d(self, mode, padding, align):
        def fn(x, g):
            return F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=align)

        x = torch.randn(2, 3, 5, 6)
        g = torch.rand(2, 7, 8, 2) * 4 - 2
        expected = fn(x, g)
        got = _compiled(fn)(x.to("neuron"), g.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("mode,padding", list(itertools.product(MODES_3D, PADDING)))
    @pytest.mark.parametrize("align", [False, True])
    def test_3d(self, mode, padding, align):
        def fn(x, g):
            return F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=align)

        x = torch.randn(2, 3, 4, 5, 6)
        g = torch.rand(2, 3, 4, 5, 3) * 4 - 2
        expected = fn(x, g)
        got = _compiled(fn)(x.to("neuron"), g.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)


# --- backward numerics ---
class TestBackward:
    """AOTAutograd generates the backward from the traced forward."""

    @pytest.mark.parametrize("mode", MODES_2D)
    @pytest.mark.parametrize("padding", PADDING)
    def test_2d_backward(self, mode, padding):
        def fn(x, g):
            return F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=False)

        x_cpu = torch.randn(2, 3, 5, 6, requires_grad=True)
        g_cpu = (torch.rand(2, 7, 8, 2) * 4 - 2).requires_grad_(True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        g_n = g_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = fn(x_cpu, g_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(fn)(x_n, g_n)
        out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(g_n.grad.cpu(), g_cpu.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("mode", MODES_3D)
    @pytest.mark.parametrize("padding", PADDING)
    def test_3d_backward(self, mode, padding):
        def fn(x, g):
            return F.grid_sample(x, g, mode=mode, padding_mode=padding, align_corners=False)

        x_cpu = torch.randn(2, 3, 4, 5, 6, requires_grad=True)
        g_cpu = (torch.rand(2, 3, 4, 5, 3) * 4 - 2).requires_grad_(True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        g_n = g_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = fn(x_cpu, g_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(fn)(x_n, g_n)
        out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(g_n.grad.cpu(), g_cpu.grad, atol=1e-3, rtol=1e-3)
