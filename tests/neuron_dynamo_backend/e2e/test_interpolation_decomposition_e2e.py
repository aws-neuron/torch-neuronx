"""End-to-end tests for the upsample/interpolate decompositions under torch.compile.

Checks numerics (the compiled graph agrees with CPU, forward and backward) and
routing (the measured graph is ours). PyTorch publishes its own upsample
decompositions as ``py_impl`` kernels that resolve above ``__torch_dispatch__``,
so without ``_release_upsample_builtin_python_decompositions()`` our registrations
are dead and the numeric tests pass while measuring PyTorch's built-ins.
``make_fx`` does not expose this, so routing is asserted directly.
"""

import pytest
import torch
from torch._C import DispatchKey

import torch_neuronx

aten = torch.ops.aten

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)

FORWARD_PACKETS = [
    aten.upsample_nearest1d,
    aten.upsample_nearest2d,
    aten._upsample_nearest_exact1d,
    aten._upsample_nearest_exact2d,
    aten.upsample_linear1d,
    aten.upsample_bilinear2d,
    aten.upsample_bicubic2d,
    aten.upsample_trilinear3d,
]

MODES_1D = [("nearest", {}), ("nearest-exact", {}), ("linear", {"align_corners": False})]
MODES_2D = [
    ("nearest", {}),
    ("nearest-exact", {}),
    ("bilinear", {"align_corners": False}),
    ("bilinear", {"align_corners": True}),
    ("bicubic", {"align_corners": False}),
    ("bicubic", {"align_corners": True}),
]


def _compiled(fn):
    """A fresh compiled callable.

    Each config needs its own function object: reusing one lambda's code object
    across configs trips ``FailOnRecompileLimitHit``.
    """
    torch._dynamo.reset()
    return torch.compile(fn, backend="neuron", fullgraph=True)


# --- routing ---
class TestDecompositionRouting:
    """PyTorch's python kernels must not shadow our decompositions."""

    @pytest.mark.parametrize("packet", FORWARD_PACKETS, ids=lambda p: str(p).split(".")[-1])
    def test_no_shadowing_python_kernels(self, packet):
        for overload_name in packet.overloads():
            overload = getattr(packet, overload_name)
            keys = set(overload.py_kernels)
            assert DispatchKey.Autograd not in keys, (
                f"{packet}.{overload_name} still has an Autograd python kernel, which "
                "resolves above __torch_dispatch__ and makes our decomposition dead code"
            )
            assert DispatchKey.CompositeImplicitAutograd not in keys, (
                f"{packet}.{overload_name} still has a CompositeImplicitAutograd python "
                "kernel shadowing our decomposition"
            )

    @pytest.mark.parametrize("packet", FORWARD_PACKETS, ids=lambda p: str(p).split(".")[-1])
    def test_meta_kernel_is_preserved(self, packet):
        """Meta must survive the pop; fake-tensor tracing needs it for shapes."""
        assert DispatchKey.Meta in packet.default.py_kernels

    @pytest.mark.parametrize("packet", FORWARD_PACKETS, ids=lambda p: str(p).split(".")[-1])
    def test_registered_in_the_neuron_table(self, packet):
        from torch_neuronx.compiler.api import get_neuron_decompositions

        table = get_neuron_decompositions()
        assert packet.default in table, f"{packet}.default missing from the table"


# --- forward numerics ---
class TestForward:
    @pytest.mark.parametrize("mode,kwargs", MODES_1D)
    def test_1d(self, mode, kwargs):
        def fn(t):
            return torch.nn.functional.interpolate(t, size=9, mode=mode, **kwargs)

        x = torch.randn(2, 3, 5)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode,kwargs", MODES_2D)
    def test_2d(self, mode, kwargs):
        def fn(t):
            return torch.nn.functional.interpolate(t, size=(7, 9), mode=mode, **kwargs)

        x = torch.randn(2, 3, 4, 5)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode,kwargs", MODES_2D)
    def test_2d_scale_factor(self, mode, kwargs):
        """scale_factor routes through the ``.vec`` overload."""

        def fn(t):
            return torch.nn.functional.interpolate(t, scale_factor=2.0, mode=mode, **kwargs)

        x = torch.randn(2, 3, 4, 5)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    def test_downsample_and_size_preserving(self):
        """The short-circuit paths, which is where the aten quirks live."""

        def fn(t):
            down = torch.nn.functional.interpolate(t, size=(2, 3), mode="bilinear")
            same = torch.nn.functional.interpolate(t, size=(6, 6), mode="bilinear")
            return down.sum() + same.sum()

        x = torch.randn(2, 3, 6, 6)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)


# --- backward numerics ---
class TestBackward:
    """AOTAutograd generates the backward from the traced forward."""

    @pytest.mark.parametrize("mode,kwargs", MODES_2D)
    def test_2d_backward(self, mode, kwargs):
        def fn(t):
            return torch.nn.functional.interpolate(t, size=(7, 9), mode=mode, **kwargs)

        x_cpu = torch.randn(2, 3, 4, 5, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = fn(x_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(fn)(x_n)
        out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach())
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    @pytest.mark.parametrize("mode,kwargs", MODES_1D)
    def test_1d_backward(self, mode, kwargs):
        def fn(t):
            return torch.nn.functional.interpolate(t, size=9, mode=mode, **kwargs)

        x_cpu = torch.randn(2, 3, 5, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = fn(x_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(fn)(x_n)
        out_n.backward(grad.to("neuron"))
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)


# --- the family in one graph ---
class TestFullGraph:
    def test_all_modes_in_one_capture(self):
        """Every mode inside a single ``fullgraph=True`` capture."""

        def fn(t):
            near = torch.nn.functional.interpolate(t, size=(8, 8), mode="nearest")
            exact = torch.nn.functional.interpolate(t, size=(8, 8), mode="nearest-exact")
            lin = torch.nn.functional.interpolate(
                t, size=(8, 8), mode="bilinear", align_corners=False
            )
            cub = torch.nn.functional.interpolate(
                t, size=(8, 8), mode="bicubic", align_corners=True
            )
            return near + exact + lin + cub

        x = torch.randn(2, 3, 4, 5)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    def test_upsample_module_in_a_model(self):
        """The shape a real decoder takes: conv-free upsample stack with a backward."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.up1 = torch.nn.Upsample(scale_factor=2, mode="nearest")
                self.up2 = torch.nn.Upsample(size=(9, 11), mode="bilinear", align_corners=False)

            def forward(self, t):
                return self.up2(self.up1(t))

        model = Model()
        x_cpu = torch.randn(2, 3, 3, 4, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = model(x_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(model)(x_n)
        out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach())
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)


# --- trilinear 3d ---
class TestTrilinear3d:
    """The volumetric mode through the compile path, forward and backward.

    The backward is auto-derived by AOTAutograd from the traced forward
    decomposition (the three separable matmuls), so no dedicated backward
    decomposition is exercised here.
    """

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_forward_size(self, align_corners):
        def fn(t):
            return torch.nn.functional.interpolate(
                t, size=(7, 8, 9), mode="trilinear", align_corners=align_corners
            )

        x = torch.randn(2, 3, 4, 4, 4)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    def test_forward_scale_factor(self):
        """scale_factor routes through the ``.vec`` overload."""

        def fn(t):
            return torch.nn.functional.interpolate(
                t, scale_factor=2.0, mode="trilinear", align_corners=False
            )

        x = torch.randn(2, 3, 4, 4, 4)
        expected = fn(x)
        got = _compiled(fn)(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("align_corners", [False, True])
    def test_backward(self, align_corners):
        def fn(t):
            return torch.nn.functional.interpolate(
                t, size=(7, 8, 9), mode="trilinear", align_corners=align_corners
            )

        x_cpu = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = fn(x_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        out_n = _compiled(fn)(x_n)
        out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach())
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)
