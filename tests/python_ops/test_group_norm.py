"""Test that native_group_norm forward and backward are properly registered and run on Neuron."""

import pytest
import torch
import torch.nn as nn

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNativeGroupNorm:
    @pytest.mark.parametrize("num_groups", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_group_norm_forward(self, num_groups, dtype):
        """Test GroupNorm forward produces correct output on Neuron."""
        torch.manual_seed(42)
        n, c, h, w = 4, 16, 8, 8

        gn_cpu = nn.GroupNorm(num_groups, c).to(dtype)
        gn_neuron = nn.GroupNorm(num_groups, c).to(dtype)
        gn_neuron.load_state_dict(gn_cpu.state_dict())
        gn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w, dtype=dtype)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = gn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_group_norm")
        out_cpu = gn_cpu(x_cpu)

        rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.3e-6
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=rtol, atol=atol)

    def test_group_norm_forward_1d(self):
        """Test GroupNorm forward with 1D spatial input (N, C, L)."""
        torch.manual_seed(42)
        n, c, seq_len = 4, 12, 32
        num_groups = 3

        gn_cpu = nn.GroupNorm(num_groups, c)
        gn_neuron = nn.GroupNorm(num_groups, c)
        gn_neuron.load_state_dict(gn_cpu.state_dict())
        gn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, seq_len)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = gn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_group_norm")
        out_cpu = gn_cpu(x_cpu)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    @pytest.mark.parametrize("num_groups", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_group_norm_backward(self, num_groups, dtype):
        """Test GroupNorm backward produces correct gradients on Neuron."""
        torch.manual_seed(42)
        n, c, h, w = 4, 16, 8, 8

        gn_cpu = nn.GroupNorm(num_groups, c).to(dtype)
        gn_neuron = nn.GroupNorm(num_groups, c).to(dtype)
        gn_neuron.load_state_dict(gn_cpu.state_dict())
        gn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = gn_cpu(x_cpu).sum()
        with track_neuron_ops():
            loss_neuron = gn_neuron(x_neuron).sum()
            loss_neuron.backward()
        assert_op_runs_on_neuron("aten::native_group_norm_backward")

        loss_cpu.backward()

        rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.3e-6
        atol = 2e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)

    def test_group_norm_backward_1d(self):
        """Test GroupNorm backward with 1D spatial input (N, C, L)."""
        torch.manual_seed(42)
        n, c, seq_len = 4, 12, 32
        num_groups = 3

        gn_cpu = nn.GroupNorm(num_groups, c)
        gn_neuron = nn.GroupNorm(num_groups, c)
        gn_neuron.load_state_dict(gn_cpu.state_dict())
        gn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, seq_len, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = gn_cpu(x_cpu).sum()
        with track_neuron_ops():
            loss_neuron = gn_neuron(x_neuron).sum()
            loss_neuron.backward()
        assert_op_runs_on_neuron("aten::native_group_norm_backward")

        loss_cpu.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    def test_group_norm_forward_out_variant(self):
        """Test aten::native_group_norm.out dispatches to Neuron."""
        torch.manual_seed(42)
        n, c, h_x_w, group = 4, 8, 64, 4
        x = torch.randn(n, c, h_x_w, device="neuron")
        weight = torch.randn(c, device="neuron")
        bias = torch.randn(c, device="neuron")
        out0 = torch.empty(n, c, h_x_w, device="neuron")
        out1 = torch.empty(n, group, device="neuron")
        out2 = torch.empty(n, group, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.native_group_norm.out(
                x,
                weight,
                bias,
                n,
                c,
                h_x_w,
                group,
                1e-5,
                out0=out0,
                out1=out1,
                out2=out2,
            )
        assert_op_runs_on_neuron("aten::native_group_norm")
        expected = torch.ops.aten.native_group_norm(
            x.cpu(), weight.cpu(), bias.cpu(), n, c, h_x_w, group, 1e-5
        )
        torch.testing.assert_close(result[0].cpu(), expected[0])
        torch.testing.assert_close(out0.cpu(), expected[0])

    def test_group_norm_backward_out_variant(self):
        """Test aten::native_group_norm_backward.out dispatches to Neuron."""
        torch.manual_seed(42)
        n, c, h_x_w, group = 4, 8, 64, 4
        x = torch.randn(n, c, h_x_w, device="neuron")
        weight = torch.randn(c, device="neuron")
        fwd = torch.ops.aten.native_group_norm(x, weight, None, n, c, h_x_w, group, 1e-5)
        mean, rstd = fwd[1], fwd[2]
        grad_out = torch.ones_like(x)
        out0 = torch.empty_like(x)
        out1 = torch.empty(c, device="neuron")
        out2 = torch.empty(c, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.native_group_norm_backward.out(
                grad_out,
                x,
                mean,
                rstd,
                weight,
                n,
                c,
                h_x_w,
                group,
                [True, True, True],
                out0=out0,
                out1=out1,
                out2=out2,
            )
        assert_op_runs_on_neuron("aten::native_group_norm_backward")
        expected = torch.ops.aten.native_group_norm_backward(
            grad_out.cpu(),
            x.cpu(),
            mean.cpu(),
            rstd.cpu(),
            weight.cpu(),
            n,
            c,
            h_x_w,
            group,
            [True, True, True],
        )
        torch.testing.assert_close(result[0].cpu(), expected[0])
        torch.testing.assert_close(out0.cpu(), expected[0])
