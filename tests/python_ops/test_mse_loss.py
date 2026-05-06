"""Test that mse_loss forward and backward are properly registered and run on Neuron."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMSELoss:
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backward", [True, False])
    def test_mse_loss(self, reduction, dtype, backward):
        """Test mse_loss forward and backward with different reductions."""
        torch.manual_seed(42)
        input_cpu = torch.randn(8, 16, dtype=dtype, requires_grad=backward)
        target_cpu = torch.randn(8, 16, dtype=dtype)

        input_neuron = input_cpu.detach().clone().to("neuron")
        if backward:
            input_neuron.requires_grad = True
        target_neuron = target_cpu.to("neuron")

        with track_neuron_ops():
            result_neuron = F.mse_loss(input_neuron, target_neuron, reduction=reduction)
            if backward:
                loss_neuron = result_neuron.sum() if reduction == "none" else result_neuron
                loss_neuron.backward()
        assert_op_runs_on_neuron("aten::mse_loss")
        if backward:
            assert_op_runs_on_neuron("aten::mse_loss_backward")

        result_cpu = F.mse_loss(input_cpu, target_cpu, reduction=reduction)
        if backward:
            loss_cpu = result_cpu.sum() if reduction == "none" else result_cpu
            loss_cpu.backward()

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)
        if backward:
            torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad)

    @pytest.mark.parametrize(
        "shape",
        [(32,), (8, 16), (4, 6, 8), (4, 6, 8, 10)],
    )
    def test_mse_loss_shapes(self, shape):
        """Test mse_loss with various tensor shapes."""
        torch.manual_seed(42)
        input_cpu = torch.randn(shape, requires_grad=True)
        target_cpu = torch.randn(shape)

        input_neuron = input_cpu.detach().clone().to("neuron").requires_grad_(True)
        target_neuron = target_cpu.to("neuron")

        with track_neuron_ops():
            result_neuron = F.mse_loss(input_neuron, target_neuron)
            result_neuron.backward()
        assert_op_runs_on_neuron("aten::mse_loss")
        assert_op_runs_on_neuron("aten::mse_loss_backward")

        result_cpu = F.mse_loss(input_cpu, target_cpu)
        result_cpu.backward()

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)
        torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad)

    @pytest.mark.parametrize("reduction", [0, 1, 2])  # none, mean, sum
    def test_mse_loss_forward_out_variant(self, reduction):
        """Test aten::mse_loss.out dispatches to Neuron."""
        torch.manual_seed(42)
        input_t = torch.randn(8, 16, device="neuron")
        target = torch.randn(8, 16, device="neuron")
        out = torch.empty(0, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.mse_loss.out(input_t, target, reduction, out=out)
        assert_op_runs_on_neuron("aten::mse_loss")
        expected = torch.ops.aten.mse_loss(input_t.cpu(), target.cpu(), reduction)
        torch.testing.assert_close(result.cpu(), expected)

    def test_mse_loss_backward_grad_input_variant(self):
        """Test aten::mse_loss_backward.grad_input dispatches to Neuron."""
        torch.manual_seed(42)
        input_t = torch.randn(8, 16, device="neuron")
        target = torch.randn(8, 16, device="neuron")
        grad_output = torch.ones((), device="neuron")
        grad_input = torch.empty_like(input_t)

        with track_neuron_ops():
            result = torch.ops.aten.mse_loss_backward.grad_input(
                grad_output, input_t, target, 1, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::mse_loss_backward")
        expected = torch.ops.aten.mse_loss_backward(
            grad_output.cpu(), input_t.cpu(), target.cpu(), 1
        )
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(grad_input.cpu(), expected)
