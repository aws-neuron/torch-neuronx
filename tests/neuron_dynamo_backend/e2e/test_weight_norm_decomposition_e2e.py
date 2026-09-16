"""
E2E tests for aten::_weight_norm_interface decomposition through torch.compile.

Verifies that a weight-normalized Linear compiles on the neuron backend and
matches CPU output, exercising the decomposition (linalg_vector_norm + mul/div)
registered in ``neuron_decompositions``.
"""

import pytest
import torch
import torch.nn as nn

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."


class WeightNormLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.utils.weight_norm(nn.Linear(16, 8), dim=dim)

    def forward(self, x):
        return self.lin(x)


class TestWeightNormDecompositionE2E:
    @pytest.mark.parametrize("dim", [0, 1])
    def test_weight_norm_linear_compile(self, dim):
        torch._dynamo.reset()
        torch.manual_seed(0)
        x = torch.randn(4, 16)

        cpu_model = WeightNormLinear(dim)
        cpu_out = cpu_model(x)

        cpu_out.sum().backward()

        neuron_model = WeightNormLinear(dim).to("neuron")
        neuron_model.load_state_dict(cpu_model.state_dict())
        compiled = torch.compile(neuron_model, backend="neuron", dynamic=False)
        neuron_out = compiled(x.to("neuron"))
        # The compiled backward graph carries the fused
        # ``_weight_norm_interface_backward`` op (autograd formula for the
        # still-fused forward); its decomposition must let it lower on Neuron.
        neuron_out.sum().backward()

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            neuron_model.lin.weight_v.grad.cpu(),
            cpu_model.lin.weight_v.grad,
            rtol=1e-3,
            atol=1e-4,
        )
        torch.testing.assert_close(
            neuron_model.lin.weight_g.grad.cpu(),
            cpu_model.lin.weight_g.grad,
            rtol=1e-3,
            atol=1e-4,
        )

    def test_weight_norm_backward_middle_dim_compile(self):
        """The compiled backward at a middle dim (dim=1 on a 3D weight reduces
        over the non-adjacent set [0, 2]) routes through the registered
        _weight_norm_interface_backward decomposition. The fused CPU kernel
        rejects a middle dim, so compare grad_v/grad_g against autograd through
        PyTorch's own forward decomposition."""
        torch._dynamo.reset()
        torch.manual_seed(0)
        vshape, dim = (4, 3, 5), 1
        v = torch.randn(vshape)
        g = torch.randn(tuple(vshape[i] if i == dim else 1 for i in range(len(vshape))))
        grad_w = torch.randn(vshape)

        _, saved_norms = torch._decomp.decompositions._weight_norm_interface(v, g, dim)
        vv = v.clone().requires_grad_(True)
        gg = g.clone().requires_grad_(True)
        w_ref, _ = torch._decomp.decompositions._weight_norm_interface(vv, gg, dim)
        exp_gv, exp_gg = torch.autograd.grad(w_ref, [vv, gg], grad_outputs=grad_w)

        def backward(grad_w, v, g, saved_norms):
            return torch.ops.aten._weight_norm_interface_backward(grad_w, v, g, saved_norms, dim)

        compiled = torch.compile(backward, backend="neuron", dynamic=False)
        gv, gg_out = compiled(
            grad_w.to("neuron"), v.to("neuron"), g.to("neuron"), saved_norms.to("neuron")
        )

        torch.testing.assert_close(gv.cpu(), exp_gv, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(gg_out.cpu(), exp_gg, rtol=1e-3, atol=1e-4)
