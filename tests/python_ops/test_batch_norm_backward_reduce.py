"""Test batch_norm_backward_reduce operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestBatchNormBackwardReduce:
    """Test batch_norm_backward_reduce operation."""

    def test_batch_norm_backward_reduce_basic(self):
        """Test basic batch_norm_backward_reduce functionality."""
        device = "neuron"

        # Create input tensors
        grad_out = torch.randn(4, 8, 16, 16, device=device)
        input_tensor = torch.randn(4, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        input_g = True
        weight_g = True
        bias_g = True

        with track_neuron_ops():
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.ops.aten.batch_norm_backward_reduce(
                grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        # Verify outputs
        assert sum_dy.device.type == "neuron"
        assert sum_dy_xmu.device.type == "neuron"
        assert grad_weight.device.type == "neuron"
        assert grad_bias.device.type == "neuron"
        assert sum_dy.shape == (8,)
        assert sum_dy_xmu.shape == (8,)
        assert grad_weight.shape == (8,)
        assert grad_bias.shape == (8,)

    def test_batch_norm_backward_reduce_no_weight_bias_grad(self):
        """Test without weight and bias gradients."""
        device = "neuron"

        grad_out = torch.randn(2, 16, 32, 32, device=device)
        input_tensor = torch.randn(2, 16, 32, 32, device=device)
        mean = torch.randn(16, device=device)
        invstd = torch.randn(16, device=device)
        weight = torch.randn(16, device=device)
        input_g = True
        weight_g = False
        bias_g = False

        with track_neuron_ops():
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.ops.aten.batch_norm_backward_reduce(
                grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        assert sum_dy.shape == (16,)
        assert sum_dy_xmu.shape == (16,)

    def test_batch_norm_backward_reduce_3d_input(self):
        """Test with 3D input tensor."""
        device = "neuron"

        grad_out = torch.randn(8, 32, 64, device=device)
        input_tensor = torch.randn(8, 32, 64, device=device)
        mean = torch.randn(32, device=device)
        invstd = torch.randn(32, device=device)
        weight = torch.randn(32, device=device)
        input_g = True
        weight_g = True
        bias_g = True

        with track_neuron_ops():
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.ops.aten.batch_norm_backward_reduce(
                grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        assert sum_dy.shape == (32,)
        assert sum_dy_xmu.shape == (32,)
        assert grad_weight.shape == (32,)
        assert grad_bias.shape == (32,)

    def test_batch_norm_backward_reduce_multiple_calls(self):
        """Test multiple calls with same inputs."""
        device = "neuron"
        grad_out = torch.randn(2, 4, 8, 8, device=device)
        input_tensor = torch.randn(2, 4, 8, 8, device=device)
        mean = torch.randn(4, device=device)
        invstd = torch.randn(4, device=device)
        weight = torch.randn(4, device=device)
        input_g = True
        weight_g = True
        bias_g = True

        with track_neuron_ops():
            sum_dy1, sum_dy_xmu1, grad_weight1, grad_bias1 = (
                torch.ops.aten.batch_norm_backward_reduce(
                    grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
                )
            )
            sum_dy2, sum_dy_xmu2, grad_weight2, grad_bias2 = (
                torch.ops.aten.batch_norm_backward_reduce(
                    grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
                )
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        # Results should be identical
        torch.testing.assert_close(sum_dy1, sum_dy2)
        torch.testing.assert_close(sum_dy_xmu1, sum_dy_xmu2)
        torch.testing.assert_close(grad_weight1, grad_weight2)
        torch.testing.assert_close(grad_bias1, grad_bias2)

    def test_batch_norm_backward_reduce_no_input_grad(self):
        """Test with input_g=False."""
        device = "neuron"

        grad_out = torch.randn(2, 8, 16, 16, device=device)
        input_tensor = torch.randn(2, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        input_g = False
        weight_g = True
        bias_g = True

        with track_neuron_ops():
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.ops.aten.batch_norm_backward_reduce(
                grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        assert sum_dy.shape == (8,)
        assert sum_dy_xmu.shape == (8,)
        assert grad_weight.shape == (8,)
        assert grad_bias.shape == (8,)

    def test_batch_norm_backward_reduce_weight_grad_only(self):
        """Test with only weight gradients enabled."""
        device = "neuron"

        grad_out = torch.randn(2, 8, 16, 16, device=device)
        input_tensor = torch.randn(2, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        input_g = False
        weight_g = True
        bias_g = False

        with track_neuron_ops():
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.ops.aten.batch_norm_backward_reduce(
                grad_out, input_tensor, mean, invstd, weight, input_g, weight_g, bias_g
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_reduce")

        assert sum_dy.shape == (8,)
        assert sum_dy_xmu.shape == (8,)
        assert grad_weight.shape == (8,)
        assert grad_bias.shape == (8,)
