"""Tests that _native_batch_norm_legit* ops are registered and run on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

_TOLERANCES = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.6e-2, 5e-2),
}


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNativeBatchNormLegit:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit_functional(self, dtype):
        """_native_batch_norm_legit_functional: training, returns running stats as outputs."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype)
        weight = torch.ones(c, dtype=dtype)
        bias = torch.zeros(c, dtype=dtype)
        running_mean = torch.zeros(c, dtype=dtype)
        running_var = torch.ones(c, dtype=dtype)

        cpu_out = torch.ops.aten._native_batch_norm_legit_functional(
            input, weight, bias, running_mean, running_var, True, 0.1, 1e-5
        )

        with track_neuron_ops():
            neuron_out = torch.ops.aten._native_batch_norm_legit_functional(
                input.to("neuron"),
                weight.to("neuron"),
                bias.to("neuron"),
                running_mean.to("neuron"),
                running_var.to("neuron"),
                True,
                0.1,
                1e-5,
            )
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit_functional")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(neuron_out[0].cpu(), cpu_out[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit(self, dtype):
        """_native_batch_norm_legit: training, updates running stats in-place."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype)
        weight = torch.ones(c, dtype=dtype)
        bias = torch.zeros(c, dtype=dtype)
        running_mean_cpu = torch.zeros(c, dtype=dtype)
        running_var_cpu = torch.ones(c, dtype=dtype)
        running_mean_neuron = running_mean_cpu.clone().to("neuron")
        running_var_neuron = running_var_cpu.clone().to("neuron")

        cpu_out = torch.ops.aten._native_batch_norm_legit(
            input, weight, bias, running_mean_cpu, running_var_cpu, True, 0.1, 1e-5
        )

        with track_neuron_ops():
            neuron_out = torch.ops.aten._native_batch_norm_legit(
                input.to("neuron"),
                weight.to("neuron"),
                bias.to("neuron"),
                running_mean_neuron,
                running_var_neuron,
                True,
                0.1,
                1e-5,
            )
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(neuron_out[0].cpu(), cpu_out[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit_no_training(self, dtype):
        """_native_batch_norm_legit_no_training: eval mode, uses running stats directly."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype)
        weight = torch.ones(c, dtype=dtype)
        bias = torch.zeros(c, dtype=dtype)
        running_mean = torch.randn(c, dtype=dtype)
        running_var = torch.rand(c, dtype=dtype).abs() + 0.1

        cpu_out = torch.ops.aten._native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, 0.1, 1e-5
        )

        with track_neuron_ops():
            neuron_out = torch.ops.aten._native_batch_norm_legit_no_training(
                input.to("neuron"),
                weight.to("neuron"),
                bias.to("neuron"),
                running_mean.to("neuron"),
                running_var.to("neuron"),
                0.1,
                1e-5,
            )
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit_no_training")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(neuron_out[0].cpu(), cpu_out[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit_out_variant(self, dtype):
        """_native_batch_norm_legit.out dispatches to Neuron and writes outputs."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype, device="neuron")
        weight = torch.ones(c, dtype=dtype, device="neuron")
        bias = torch.zeros(c, dtype=dtype, device="neuron")
        running_mean = torch.zeros(c, dtype=dtype, device="neuron")
        running_var = torch.ones(c, dtype=dtype, device="neuron")

        expected = torch.ops.aten._native_batch_norm_legit(
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

        with track_neuron_ops():
            result = torch.ops.aten._native_batch_norm_legit.out(
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
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(result[0].cpu(), expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(out.cpu(), expected[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit_no_training_out_variant(self, dtype):
        """_native_batch_norm_legit_no_training.out dispatches to Neuron and writes outputs."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w, dtype=dtype, device="neuron")
        weight = torch.ones(c, dtype=dtype, device="neuron")
        bias = torch.zeros(c, dtype=dtype, device="neuron")
        running_mean = torch.randn(c, dtype=dtype, device="neuron")
        running_var = torch.rand(c, dtype=dtype, device="neuron").abs() + 0.1

        expected = torch.ops.aten._native_batch_norm_legit_no_training(
            input.cpu(),
            weight.cpu(),
            bias.cpu(),
            running_mean.cpu(),
            running_var.cpu(),
            0.1,
            1e-5,
        )
        out0 = torch.empty_like(expected[0], device="neuron")
        out1 = torch.empty_like(expected[1], device="neuron")
        out2 = torch.empty_like(expected[2], device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten._native_batch_norm_legit_no_training.out(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                0.1,
                1e-5,
                out0=out0,
                out1=out1,
                out2=out2,
            )
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit_no_training")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(result[0].cpu(), expected[0], rtol=rtol, atol=atol)
        torch.testing.assert_close(out0.cpu(), expected[0], rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_batch_norm_legit_functional_backward(self, dtype):
        """Autograd through _native_batch_norm_legit_functional: gradients should match CPU."""
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4

        x_cpu = torch.randn(n, c, h, w, dtype=dtype, requires_grad=True)
        weight_cpu = torch.randn(c, dtype=dtype, requires_grad=True)
        bias_cpu = torch.randn(c, dtype=dtype, requires_grad=True)
        running_mean_cpu = torch.zeros(c, dtype=dtype)
        running_var_cpu = torch.ones(c, dtype=dtype)

        x_neuron = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        weight_neuron = weight_cpu.detach().clone().to("neuron").requires_grad_(True)
        bias_neuron = bias_cpu.detach().clone().to("neuron").requires_grad_(True)
        running_mean_neuron = running_mean_cpu.clone().to("neuron")
        running_var_neuron = running_var_cpu.clone().to("neuron")

        out_cpu = torch.ops.aten._native_batch_norm_legit_functional(
            x_cpu, weight_cpu, bias_cpu, running_mean_cpu, running_var_cpu, True, 0.1, 1e-5
        )
        out_cpu[0].sum().backward()

        with track_neuron_ops():
            out_neuron = torch.ops.aten._native_batch_norm_legit_functional(
                x_neuron,
                weight_neuron,
                bias_neuron,
                running_mean_neuron,
                running_var_neuron,
                True,
                0.1,
                1e-5,
            )
            out_neuron[0].sum().backward()
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit_functional")
        assert_op_runs_on_neuron("aten::native_batch_norm_backward")

        rtol, atol = _TOLERANCES[dtype]
        torch.testing.assert_close(x_neuron.grad.cpu(), x_cpu.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(weight_neuron.grad.cpu(), weight_cpu.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(bias_neuron.grad.cpu(), bias_cpu.grad, rtol=rtol, atol=atol)

    def test_native_batch_norm_legit_functional_momentum(self):
        """Different momentum values produce correct running stats.

        momentum is in static_argnums (the underlying ATen schema requires a Python
        float, which the framework cannot satisfy if the arg is tracerized into a
        meta tensor), so each value triggers a recompile. This test runs the op with
        two distinct momentum values and verifies the running_mean/running_var
        outputs match the CPU reference for both — catching any regression where
        momentum is incorrectly cached or an incorrect value is propagated.
        """
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input = torch.randn(n, c, h, w)
        weight = torch.randn(c)
        bias = torch.randn(c)
        running_mean = torch.randn(c)
        running_var = torch.rand(c).abs() + 0.1
        rtol, atol = _TOLERANCES[torch.float32]

        for momentum in (0.1, 0.5):
            cpu_out = torch.ops.aten._native_batch_norm_legit_functional(
                input, weight, bias, running_mean, running_var, True, momentum, 1e-5
            )

            with track_neuron_ops():
                neuron_out = torch.ops.aten._native_batch_norm_legit_functional(
                    input.to("neuron"),
                    weight.to("neuron"),
                    bias.to("neuron"),
                    running_mean.to("neuron"),
                    running_var.to("neuron"),
                    True,
                    momentum,
                    1e-5,
                )
            assert_op_runs_on_neuron("aten::_native_batch_norm_legit_functional")

            # _native_batch_norm_legit_functional returns:
            # (output, save_mean, save_invstd, running_mean_new, running_var_new)
            torch.testing.assert_close(neuron_out[3].cpu(), cpu_out[3], rtol=rtol, atol=atol)
            torch.testing.assert_close(neuron_out[4].cpu(), cpu_out[4], rtol=rtol, atol=atol)

    def test_eager_forward_preserves_autograd_chain(self):
        """Eager registration must not detach autograd: out.grad_fn should be non-None.

        native_batch_norm is in BOTH the torch.compile decomposition table and the
        eager @register_aten registration. PrivateUse1 dispatches below autograd, so
        grad_fn should still attach — this test guards against any future regression
        that would silently break training.
        """
        n, c, h, w = 4, 8, 4, 4
        running_mean = torch.zeros(c, device="neuron")
        running_var = torch.ones(c, device="neuron")

        def make_inputs():
            torch.manual_seed(42)
            x = torch.randn(n, c, h, w, device="neuron", requires_grad=True)
            weight = torch.randn(c, device="neuron", requires_grad=True)
            bias = torch.randn(c, device="neuron", requires_grad=True)
            return x, weight, bias

        x, weight, bias = make_inputs()
        out_legit = torch.ops.aten._native_batch_norm_legit(
            x, weight, bias, running_mean.clone(), running_var.clone(), True, 0.1, 1e-5
        )
        assert out_legit[0].grad_fn is not None, "_native_batch_norm_legit broke autograd chain"

        x, weight, bias = make_inputs()
        out_func = torch.ops.aten._native_batch_norm_legit_functional(
            x, weight, bias, running_mean.clone(), running_var.clone(), True, 0.1, 1e-5
        )
        assert (
            out_func[0].grad_fn is not None
        ), "_native_batch_norm_legit_functional broke autograd chain"

        x, weight, bias = make_inputs()
        out_nbn = torch.ops.aten.native_batch_norm(
            x, weight, bias, running_mean.clone(), running_var.clone(), True, 0.1, 1e-5
        )
        assert out_nbn[0].grad_fn is not None, "native_batch_norm broke autograd chain"

    def test_native_batch_norm_legit_mixed_dtype(self):
        """BF16 input/weight/bias + FP32 running_mean/running_var on the eager path.

        This is the exact configuration that triggered the original ticket (P422864134)
        in ResNet-50 with mixed precision: ATen's CPU _native_batch_norm_legit rejects
        the BF16-input + FP32-stats combo, but Neuron's eager path must accept it.
        We compare the Neuron BF16 output against a CPU FP32 reference (input cast up).
        """
        torch.manual_seed(42)
        n, c, h, w = 4, 8, 4, 4
        input_bf16 = torch.randn(n, c, h, w, dtype=torch.bfloat16)
        weight_bf16 = torch.ones(c, dtype=torch.bfloat16)
        bias_bf16 = torch.zeros(c, dtype=torch.bfloat16)
        running_mean = torch.zeros(c, dtype=torch.float32)
        running_var = torch.ones(c, dtype=torch.float32)

        # CPU reference in pure FP32 since CPU op rejects the mixed-dtype combo.
        cpu_out = torch.ops.aten._native_batch_norm_legit(
            input_bf16.float(),
            weight_bf16.float(),
            bias_bf16.float(),
            running_mean.clone(),
            running_var.clone(),
            True,
            0.1,
            1e-5,
        )

        with track_neuron_ops():
            neuron_out = torch.ops.aten._native_batch_norm_legit(
                input_bf16.to("neuron"),
                weight_bf16.to("neuron"),
                bias_bf16.to("neuron"),
                running_mean.to("neuron"),
                running_var.to("neuron"),
                True,
                0.1,
                1e-5,
            )
        assert_op_runs_on_neuron("aten::_native_batch_norm_legit")

        rtol, atol = _TOLERANCES[torch.bfloat16]
        torch.testing.assert_close(neuron_out[0].float().cpu(), cpu_out[0], rtol=rtol, atol=atol)
