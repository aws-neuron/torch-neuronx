"""Test that native_batch_norm forward and backward are properly registered and run on Neuron."""

import pytest
import torch
import torch.nn as nn

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

_TOLERANCES = {
    torch.float32: (1.3e-6, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-3),
}


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNativeBatchNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_batch_norm_forward_train(self, dtype):
        """Test BatchNorm2d forward in training mode produces correct output on Neuron."""
        torch.manual_seed(42)
        n, c, h, w = 4, 16, 8, 8

        bn_cpu = nn.BatchNorm2d(c).to(dtype).train()
        bn_neuron = nn.BatchNorm2d(c).to(dtype).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w, dtype=dtype)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = bn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_batch_norm")
        out_cpu = bn_cpu(x_cpu)

        rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.3e-6
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_batch_norm_forward_eval(self, dtype):
        """Test BatchNorm2d forward in eval mode produces correct output on Neuron."""
        torch.manual_seed(42)
        n, c, h, w = 4, 16, 8, 8

        bn_cpu = nn.BatchNorm2d(c).to(dtype).eval()
        bn_neuron = nn.BatchNorm2d(c).to(dtype).eval()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w, dtype=dtype)
        x_neuron = x_cpu.clone().to("neuron")

        with torch.no_grad():
            with track_neuron_ops():
                out_neuron = bn_neuron(x_neuron)
            assert_op_runs_on_neuron("aten::native_batch_norm")
            out_cpu = bn_cpu(x_cpu)

        rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.3e-6
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "memory_format,input_shape",
        [
            (torch.channels_last, (2, 4, 6, 7)),
            (torch.channels_last_3d, (2, 4, 3, 6, 7)),
        ],
    )
    def test_native_batch_norm_preserves_channels_last(self, memory_format, input_shape):
        torch.manual_seed(42)
        channels = input_shape[1]
        input_cpu = torch.randn(input_shape).contiguous(memory_format=memory_format)
        weight_cpu = torch.randn(channels)
        bias_cpu = torch.randn(channels)
        mean_cpu = torch.randn(channels)
        var_cpu = torch.rand(channels) + 0.5
        cpu_args = (input_cpu, weight_cpu, bias_cpu, mean_cpu, var_cpu)
        neuron_args = tuple(value.to("neuron") for value in cpu_args)

        expected = torch.ops.aten.native_batch_norm(*cpu_args, False, 0.1, 1e-5)[0]
        with track_neuron_ops():
            actual = torch.ops.aten.native_batch_norm(*neuron_args, False, 0.1, 1e-5)[0]
        assert_op_runs_on_neuron("aten::native_batch_norm")

        assert actual.is_contiguous(memory_format=memory_format)
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.parametrize(
        "memory_format,input_shape",
        [
            (torch.channels_last, (0, 4, 6, 7)),
            (torch.channels_last_3d, (0, 4, 3, 6, 7)),
        ],
    )
    def test_empty_native_batch_norm_preserves_channels_last(self, memory_format, input_shape):
        channels = input_shape[1]
        input = torch.empty(input_shape, device="neuron", memory_format=memory_format)
        weight = torch.ones(channels, device="neuron")
        bias = torch.zeros(channels, device="neuron")
        running_mean = torch.zeros(channels, device="neuron")
        running_var = torch.ones(channels, device="neuron")

        actual = torch.ops.aten.native_batch_norm(
            input, weight, bias, running_mean, running_var, False, 0.1, 1e-5
        )[0]

        assert actual.numel() == 0
        assert actual.is_contiguous(memory_format=memory_format)

    def test_batch_norm_forward_1d(self):
        """Test BatchNorm1d forward with 1D spatial input (N, C, L)."""
        torch.manual_seed(42)
        n, c, seq_len = 4, 12, 32

        bn_cpu = nn.BatchNorm1d(c).train()
        bn_neuron = nn.BatchNorm1d(c).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, seq_len)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = bn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_batch_norm")
        out_cpu = bn_cpu(x_cpu)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_batch_norm_backward(self, dtype):
        """Test BatchNorm2d backward produces correct gradients on Neuron."""
        torch.manual_seed(42)
        n, c, h, w = 4, 16, 8, 8

        bn_cpu = nn.BatchNorm2d(c).to(dtype).train()
        bn_neuron = nn.BatchNorm2d(c).to(dtype).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w, dtype=dtype, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = bn_cpu(x_cpu).sum()
        with track_neuron_ops():
            loss_neuron = bn_neuron(x_neuron).sum()
            loss_neuron.backward()
        assert_op_runs_on_neuron("aten::native_batch_norm_backward")

        loss_cpu.backward()

        rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.3e-6
        atol = 4e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)

    def test_batch_norm_backward_1d(self):
        """Test BatchNorm1d backward with 1D spatial input (N, C, L)."""
        torch.manual_seed(42)
        n, c, seq_len = 4, 12, 32

        bn_cpu = nn.BatchNorm1d(c).train()
        bn_neuron = nn.BatchNorm1d(c).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, seq_len, requires_grad=True)
        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        loss_cpu = bn_cpu(x_cpu).sum()
        with track_neuron_ops():
            loss_neuron = bn_neuron(x_neuron).sum()
            loss_neuron.backward()
        assert_op_runs_on_neuron("aten::native_batch_norm_backward")

        loss_cpu.backward()

        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_native_batch_norm_out_variant(self, dtype):
        """aten::native_batch_norm.out dispatches to Neuron and writes pre-allocated outputs."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype, device="neuron")
        weight = torch.ones(c, dtype=dtype, device="neuron")
        bias = torch.zeros(c, dtype=dtype, device="neuron")
        running_mean = torch.zeros(c, dtype=dtype, device="neuron")
        running_var = torch.ones(c, dtype=dtype, device="neuron")

        expected = torch.ops.aten.native_batch_norm(
            input.cpu(),
            weight.cpu(),
            bias.cpu(),
            running_mean.cpu(),
            running_var.cpu(),
            True,
            0.1,
            1e-5,
        )
        out = torch.empty_like(expected[0], device="neuron")
        save_mean = torch.empty_like(expected[1], device="neuron")
        save_invstd = torch.empty_like(expected[2], device="neuron")
        output_impls = tuple(tensor._cdata for tensor in (out, save_mean, save_invstd))
        output_storage_impls = tuple(
            tensor.untyped_storage()._cdata for tensor in (out, save_mean, save_invstd)
        )

        with track_neuron_ops():
            result = torch.ops.aten.native_batch_norm.out(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                True,
                0.1,
                1e-5,
                out=out,
                save_mean=save_mean,
                save_invstd=save_invstd,
            )
        assert_op_runs_on_neuron("aten::native_batch_norm")

        assert tuple(tensor._cdata for tensor in result) == output_impls
        assert tuple(tensor.untyped_storage()._cdata for tensor in result) == output_storage_impls
        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(result[0].cpu(), expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(out.cpu(), expected[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_native_batch_norm_backward_out_variant(self, dtype):
        """native_batch_norm_backward.out dispatches to Neuron and writes outputs."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype, device="neuron")
        weight = torch.randn(c, dtype=dtype, device="neuron")
        running_mean = torch.zeros(c, dtype=dtype, device="neuron")
        running_var = torch.ones(c, dtype=dtype, device="neuron")
        fwd = torch.ops.aten.native_batch_norm(
            input,
            weight,
            None,
            running_mean,
            running_var,
            True,
            0.1,
            1e-5,
        )
        save_mean, save_invstd = fwd[1], fwd[2]
        grad_out = torch.ones_like(input)

        expected = torch.ops.aten.native_batch_norm_backward(
            grad_out.cpu(),
            input.cpu(),
            weight.cpu(),
            running_mean.cpu(),
            running_var.cpu(),
            save_mean.cpu(),
            save_invstd.cpu(),
            True,
            1e-5,
            [True, True, True],
        )
        out0 = torch.empty_like(expected[0], device="neuron")
        out1 = torch.empty_like(expected[1], device="neuron")
        out2 = torch.empty_like(expected[2], device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.native_batch_norm_backward.out(
                grad_out,
                input,
                weight,
                running_mean,
                running_var,
                save_mean,
                save_invstd,
                True,
                1e-5,
                [True, True, True],
                out0=out0,
                out1=out1,
                out2=out2,
            )
        assert_op_runs_on_neuron("aten::native_batch_norm_backward")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(result[0].cpu(), expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(out0.cpu(), expected[0], rtol=rtol, atol=atol)

    def test_native_batch_norm_backward_default_overload(self):
        """Direct call to aten::native_batch_norm_backward (default overload, no .out)."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, device="neuron")
        weight = torch.randn(c, device="neuron")
        running_mean = torch.zeros(c, device="neuron")
        running_var = torch.ones(c, device="neuron")
        fwd = torch.ops.aten.native_batch_norm(
            input, weight, None, running_mean, running_var, True, 0.1, 1e-5
        )
        save_mean, save_invstd = fwd[1], fwd[2]
        grad_out = torch.ones_like(input)

        expected = torch.ops.aten.native_batch_norm_backward(
            grad_out.cpu(),
            input.cpu(),
            weight.cpu(),
            running_mean.cpu(),
            running_var.cpu(),
            save_mean.cpu(),
            save_invstd.cpu(),
            True,
            1e-5,
            [True, True, True],
        )

        with track_neuron_ops():
            result = torch.ops.aten.native_batch_norm_backward(
                grad_out,
                input,
                weight,
                running_mean,
                running_var,
                save_mean,
                save_invstd,
                True,
                1e-5,
                [True, True, True],
            )
        assert_op_runs_on_neuron("aten::native_batch_norm_backward")

        rtol, atol = _TOLERANCES[torch.float32]
        torch.testing.assert_close(result[0].cpu(), expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(result[1].cpu(), expected[1], rtol=rtol, atol=atol)
        torch.testing.assert_close(result[2].cpu(), expected[2], rtol=rtol, atol=atol)

    def test_batch_norm_empty_batch(self):
        """N=0 input through aten::native_batch_norm directly: shape preserved, no crash."""
        torch.manual_seed(42)
        n, c, h, w = 0, 8, 4, 4
        weight = torch.randn(c)
        bias = torch.randn(c)
        running_mean = torch.zeros(c)
        running_var = torch.ones(c)
        x_cpu = torch.randn(n, c, h, w)

        cpu_out = torch.ops.aten.native_batch_norm(
            x_cpu, weight, bias, running_mean, running_var, False, 0.1, 1e-5
        )

        with track_neuron_ops():
            neuron_out = torch.ops.aten.native_batch_norm(
                x_cpu.to("neuron"),
                weight.to("neuron"),
                bias.to("neuron"),
                running_mean.to("neuron"),
                running_var.to("neuron"),
                False,
                0.1,
                1e-5,
            )
        assert_op_runs_on_neuron("aten::native_batch_norm")

        assert neuron_out[0].shape == (n, c, h, w)
        torch.testing.assert_close(neuron_out[0].cpu(), cpu_out[0])

    def test_batch_norm_single_element_batch(self):
        """N=1 in training mode: per-channel variance is zero, eps prevents divide-by-zero."""
        torch.manual_seed(42)
        n, c, h, w = 1, 8, 4, 4

        bn_cpu = nn.BatchNorm2d(c).train()
        bn_neuron = nn.BatchNorm2d(c).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = bn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_batch_norm")
        out_cpu = bn_cpu(x_cpu)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert torch.isfinite(out_neuron.cpu()).all()

    def test_batch_norm_large_channel_count(self):
        """Large channel count (ResNet-50 final block scale) — verifies no shape constraints."""
        torch.manual_seed(42)
        n, c, h, w = 2, 2048, 4, 4

        bn_cpu = nn.BatchNorm2d(c).train()
        bn_neuron = nn.BatchNorm2d(c).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = bn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_batch_norm")
        out_cpu = bn_cpu(x_cpu)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_batch_norm_no_affine(self):
        """weight=None and bias=None — batch norm without affine parameters."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4

        bn_cpu = nn.BatchNorm2d(c, affine=False).train()
        bn_neuron = nn.BatchNorm2d(c, affine=False).train()
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron.to("neuron")

        x_cpu = torch.randn(n, c, h, w)
        x_neuron = x_cpu.clone().to("neuron")

        with track_neuron_ops():
            out_neuron = bn_neuron(x_neuron)
        assert_op_runs_on_neuron("aten::native_batch_norm")
        out_cpu = bn_cpu(x_cpu)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
