"""Test aten::_weight_norm_interface lowering on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

OP = "aten::_weight_norm_interface"


def _g_for(vshape, dim):
    """Keepdim magnitude tensor matching torch.nn.utils.weight_norm's ``g``."""
    return tuple(vshape[i] if i == dim else 1 for i in range(len(vshape)))


# (vshape, dim)
CASES = [
    ((8, 16), 0),
    ((8, 16), 1),
    ((4, 3, 5), 0),
    ((4, 3, 5), 2),
    ((2, 3, 4, 5), 0),
    ((2, 3, 4, 5), 3),
    ((1, 7), 0),
    ((7, 1), 0),
]


class TestWeightNormInterface:
    @pytest.mark.parametrize("vshape,dim", CASES)
    def test_forward_matches_cpu(self, vshape, dim):
        torch.manual_seed(0)
        v = torch.randn(vshape)
        g = torch.randn(_g_for(vshape, dim))

        exp_w, exp_n = torch.ops.aten._weight_norm_interface(v, g, dim)

        with track_neuron_ops():
            w, n = torch.ops.aten._weight_norm_interface(v.to("neuron"), g.to("neuron"), dim)
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)

        assert w.device.type == "neuron"
        assert n.device.type == "neuron"
        torch.testing.assert_close(w.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(n.cpu(), exp_n, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.bfloat16, 3e-2)])
    def test_low_precision_dtypes(self, dtype, tol):
        torch.manual_seed(0)
        vshape, dim = (8, 16), 0
        v = torch.randn(vshape, dtype=dtype)
        g = torch.randn(_g_for(vshape, dim), dtype=dtype)

        exp_w, exp_n = torch.ops.aten._weight_norm_interface(v, g, dim)

        with track_neuron_ops():
            w, n = torch.ops.aten._weight_norm_interface(v.to("neuron"), g.to("neuron"), dim)
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)

        assert w.dtype == dtype
        torch.testing.assert_close(w.cpu().float(), exp_w.float(), rtol=tol, atol=tol)
        torch.testing.assert_close(n.cpu().float(), exp_n.float(), rtol=tol, atol=tol)

    def test_out_variant_writes_and_leaves_inputs_untouched(self):
        torch.manual_seed(0)
        vshape, dim = (6, 10), 0
        v = torch.randn(vshape).to("neuron")
        g = torch.randn(_g_for(vshape, dim)).to("neuron")
        v_before = v.cpu().clone()
        g_before = g.cpu().clone()

        exp_w, exp_n = torch.ops.aten._weight_norm_interface(v_before, g_before, dim)

        out0 = torch.empty(vshape, device="neuron")
        out1 = torch.empty(_g_for(vshape, dim), device="neuron")

        with track_neuron_ops():
            r0, r1 = torch.ops.aten._weight_norm_interface.out(v, g, dim, out0=out0, out1=out1)
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)

        # out tensors received the values
        torch.testing.assert_close(out0.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out1.cpu(), exp_n, rtol=1e-4, atol=1e-5)
        # returned aliases point at the out tensors
        torch.testing.assert_close(r0.cpu(), out0.cpu())
        torch.testing.assert_close(r1.cpu(), out1.cpu())
        # inputs left untouched
        torch.testing.assert_close(v.cpu(), v_before)
        torch.testing.assert_close(g.cpu(), g_before)

    def test_out_variant_noncontiguous(self):
        torch.manual_seed(0)
        vshape, dim = (6, 10), 0
        v = torch.randn(vshape).to("neuron")
        g = torch.randn(_g_for(vshape, dim)).to("neuron")
        exp_w, exp_n = torch.ops.aten._weight_norm_interface(v.cpu(), g.cpu(), dim)

        # Non-contiguous out0 via a transposed larger buffer.
        out0 = torch.empty((10, 6), device="neuron").t()
        out1 = torch.empty(_g_for(vshape, dim), device="neuron")
        assert not out0.is_contiguous()

        torch.ops.aten._weight_norm_interface.out(v, g, dim, out0=out0, out1=out1)
        torch.testing.assert_close(out0.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out1.cpu(), exp_n, rtol=1e-4, atol=1e-5)

    def test_single_element(self):
        v = torch.randn(1, 1).to("neuron")
        g = torch.randn(1, 1).to("neuron")
        exp_w, exp_n = torch.ops.aten._weight_norm_interface(v.cpu(), g.cpu(), 0)
        with track_neuron_ops():
            w, n = torch.ops.aten._weight_norm_interface(v, g, 0)
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)
        torch.testing.assert_close(w.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(n.cpu(), exp_n, rtol=1e-4, atol=1e-5)

    def test_autograd_through_weight_norm_module(self):
        """End-to-end grad via torch.nn.utils.weight_norm, which uses this op."""
        torch.manual_seed(0)
        lin = torch.nn.Linear(16, 8)
        lin = torch.nn.utils.weight_norm(lin, dim=0)
        x = torch.randn(4, 16)

        exp = lin(x)
        exp.sum().backward()
        exp_gv = lin.weight_v.grad.clone()
        exp_gg = lin.weight_g.grad.clone()

        lin.zero_grad()
        lin_n = lin.to("neuron")
        with track_neuron_ops():
            out = lin_n(x.to("neuron"))
            out.sum().backward()
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)
            # The backward pass must also lower natively, not bounce to CPU.
            assert_op_runs_on_neuron("aten::_weight_norm_interface_backward")

        torch.testing.assert_close(out.cpu(), exp, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lin_n.weight_v.grad.cpu(), exp_gv, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(lin_n.weight_g.grad.cpu(), exp_gg, rtol=1e-3, atol=1e-4)

    def test_autograd_multidim_reduce_conv_weight(self):
        """Backward with a 4D weight reduces over multiple dims ([1,2,3] for
        dim=0). The 2D Linear case above only reduces one dim, so this locks in
        the multi-dim-reduce path (the vertical harness skips CPU correctness
        for the backward, so this unit test is the regression guard)."""
        torch.manual_seed(0)
        # Conv2d weight: (out_ch, in_ch, kH, kW) = (64, 32, 3, 3); dim=0 reduces
        # the trailing [1, 2, 3] dims into g of shape (64, 1, 1, 1).
        conv = torch.nn.Conv2d(32, 64, kernel_size=3)
        conv = torch.nn.utils.weight_norm(conv, dim=0)
        x = torch.randn(2, 32, 8, 8)

        exp = conv(x)
        exp.sum().backward()
        exp_gv = conv.weight_v.grad.clone()
        exp_gg = conv.weight_g.grad.clone()

        conv.zero_grad()
        conv_n = conv.to("neuron")
        with track_neuron_ops():
            out = conv_n(x.to("neuron"))
            out.sum().backward()
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)
            assert_op_runs_on_neuron("aten::_weight_norm_interface_backward")

        assert conv_n.weight_v.grad.dim() == 4
        assert conv_n.weight_g.grad.shape == (64, 1, 1, 1)
        torch.testing.assert_close(out.cpu(), exp, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(conv_n.weight_v.grad.cpu(), exp_gv, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(conv_n.weight_g.grad.cpu(), exp_gg, rtol=1e-3, atol=1e-4)

    def test_backward_middle_dim_matches_cpu(self):
        """Backward at a middle dim (dim=1 on a 3D weight reduces over the
        non-adjacent set [0, 2]). The fused CPU kernel asserts a first or last
        dim, so the reference comes from autograd through PyTorch's own forward
        decomposition. The Neuron backward op must run on device and match it."""
        torch.manual_seed(0)
        vshape, dim = (4, 3, 5), 1
        v = torch.randn(vshape)
        g = torch.randn(_g_for(vshape, dim))
        grad_w = torch.randn(vshape)

        # saved_norms is the true norm, so the 1/norm**3 term stays well conditioned.
        _, saved_norms = torch._decomp.decompositions._weight_norm_interface(v, g, dim)
        vv = v.clone().requires_grad_(True)
        gg = g.clone().requires_grad_(True)
        w_ref, _ = torch._decomp.decompositions._weight_norm_interface(vv, gg, dim)
        exp_gv, exp_gg = torch.autograd.grad(w_ref, [vv, gg], grad_outputs=grad_w)

        with track_neuron_ops():
            gv, gg_out = torch.ops.aten._weight_norm_interface_backward(
                grad_w.to("neuron"), v.to("neuron"), g.to("neuron"), saved_norms.to("neuron"), dim
            )
            torch.neuron.synchronize()
            assert_op_runs_on_neuron("aten::_weight_norm_interface_backward")

        assert gg_out.shape == _g_for(vshape, dim)
        torch.testing.assert_close(gv.cpu(), exp_gv, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(gg_out.cpu(), exp_gg, rtol=1e-4, atol=1e-5)

    def test_middle_dim_matches_decomposition(self):
        """The CPU fused kernel rejects a middle dim, but the decomposition Neuron
        uses handles it; validate against PyTorch's reference decomposition."""
        torch.manual_seed(0)
        vshape, dim = (4, 3, 5), 1
        v = torch.randn(vshape)
        g = torch.randn(_g_for(vshape, dim))
        exp_w, exp_n = torch._decomp.decompositions._weight_norm_interface(v, g, dim)
        with track_neuron_ops():
            w, n = torch.ops.aten._weight_norm_interface(v.to("neuron"), g.to("neuron"), dim)
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)
        torch.testing.assert_close(w.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(n.cpu(), exp_n, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "vshape,neg_dim", [((4, 3, 5), -1), ((4, 3, 5), -2), ((2, 3, 4, 5), -1)]
    )
    def test_negative_dim_matches_positive(self, vshape, neg_dim):
        """A negative dim must normalize to ``v.dim() + dim`` in the forward and
        the backward alike. aten's fused CPU kernel rejects a negative dim, so the
        reference comes from PyTorch's forward decomposition at the positive dim."""
        torch.manual_seed(0)
        pos_dim = neg_dim + len(vshape)
        v = torch.randn(vshape)
        g = torch.randn(_g_for(vshape, pos_dim))
        grad_w = torch.randn(vshape)

        exp_w, saved_norms = torch._decomp.decompositions._weight_norm_interface(v, g, pos_dim)
        vv = v.clone().requires_grad_(True)
        gg = g.clone().requires_grad_(True)
        w_ref, _ = torch._decomp.decompositions._weight_norm_interface(vv, gg, pos_dim)
        exp_gv, exp_gg = torch.autograd.grad(w_ref, [vv, gg], grad_outputs=grad_w)

        with track_neuron_ops():
            w, n = torch.ops.aten._weight_norm_interface(v.to("neuron"), g.to("neuron"), neg_dim)
            gv, gg_out = torch.ops.aten._weight_norm_interface_backward(
                grad_w.to("neuron"),
                v.to("neuron"),
                g.to("neuron"),
                saved_norms.to("neuron"),
                neg_dim,
            )
            torch.neuron.synchronize()
            assert_op_runs_on_neuron(OP)
            assert_op_runs_on_neuron("aten::_weight_norm_interface_backward")

        torch.testing.assert_close(w.cpu(), exp_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(n.cpu(), saved_norms, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(gv.cpu(), exp_gv, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(gg_out.cpu(), exp_gg, rtol=1e-4, atol=1e-5)

    def test_decomposition_not_shadowed(self):
        """The compile-path decomposition only runs if no Autograd/Composite
        py_kernel resolves above __torch_dispatch__ and shadows it (gotcha #5)."""
        from torch._C import DispatchKey

        from torch_neuronx.neuron_dynamo_backend.decompositions import neuron_decompositions

        for ov in ("default", "out"):
            op = getattr(torch.ops.aten._weight_norm_interface, ov)
            keys = op.py_kernels
            assert DispatchKey.Autograd not in keys, keys
            assert DispatchKey.CompositeImplicitAutograd not in keys, keys
        # Only the .default overload has a valid (out-wrapper-free) decomposition;
        # the .out overload is intentionally excluded from the compile path.
        assert torch.ops.aten._weight_norm_interface.default in neuron_decompositions
        assert torch.ops.aten._weight_norm_interface.out not in neuron_decompositions
        # AOTAutograd emits the fused backward op in the compiled backward graph,
        # so its .default decomposition must also be registered (torch-mlir has
        # no native lowering for it).
        assert torch.ops.aten._weight_norm_interface_backward.default in neuron_decompositions
        assert torch.ops.aten._weight_norm_interface_backward.out not in neuron_decompositions
