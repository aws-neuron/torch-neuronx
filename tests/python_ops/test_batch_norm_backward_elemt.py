"""Test batch_norm_backward_elemt operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestBatchNormBackwardElemt:
    """Test batch_norm_backward_elemt operation."""

    def test_batch_norm_backward_elemt_basic(self):
        """Test basic batch_norm_backward_elemt functionality."""
        device = "neuron"

        # Create input tensors
        grad_out = torch.randn(4, 8, 16, 16, device=device)
        input_tensor = torch.randn(4, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        sum_dy = torch.randn(8, device=device)
        sum_dy_xmu = torch.randn(8, device=device)
        count = torch.tensor(1024, device=device)  # N*H*W

        with track_neuron_ops():
            grad_input = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        # Verify output
        assert grad_input.device.type == "neuron"
        assert grad_input.shape == input_tensor.shape

    def test_batch_norm_backward_elemt_no_weight(self):
        """Test batch_norm_backward_elemt without weight."""
        device = "neuron"

        grad_out = torch.randn(2, 16, 32, 32, device=device)
        input_tensor = torch.randn(2, 16, 32, 32, device=device)
        mean = torch.randn(16, device=device)
        invstd = torch.randn(16, device=device)
        sum_dy = torch.randn(16, device=device)
        sum_dy_xmu = torch.randn(16, device=device)
        count = torch.tensor(2048, device=device)

        with track_neuron_ops():
            grad_input = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, None, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        assert grad_input.shape == input_tensor.shape

    def test_batch_norm_backward_elemt_3d_input(self):
        """Test with 3D input tensor."""
        device = "neuron"

        grad_out = torch.randn(8, 32, 64, device=device)
        input_tensor = torch.randn(8, 32, 64, device=device)
        mean = torch.randn(32, device=device)
        invstd = torch.randn(32, device=device)
        weight = torch.randn(32, device=device)
        sum_dy = torch.randn(32, device=device)
        sum_dy_xmu = torch.randn(32, device=device)
        count = torch.tensor(512, device=device)

        with track_neuron_ops():
            grad_input = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        assert grad_input.shape == input_tensor.shape

    def test_batch_norm_backward_elemt_multiple_calls(self):
        """Test multiple calls with same inputs."""
        device = "neuron"
        grad_out = torch.randn(2, 4, 8, 8, device=device)
        input_tensor = torch.randn(2, 4, 8, 8, device=device)
        mean = torch.randn(4, device=device)
        invstd = torch.randn(4, device=device)
        weight = torch.randn(4, device=device)
        sum_dy = torch.randn(4, device=device)
        sum_dy_xmu = torch.randn(4, device=device)
        count = torch.tensor(128, device=device)

        with track_neuron_ops():
            grad_input1 = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            grad_input2 = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        # Results should be identical
        torch.testing.assert_close(grad_input1, grad_input2)

    def test_batch_norm_backward_elemt_different_counts(self):
        """Test with different count values."""
        device = "neuron"
        grad_out = torch.randn(4, 8, 16, 16, device=device)
        input_tensor = torch.randn(4, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        sum_dy = torch.randn(8, device=device)
        sum_dy_xmu = torch.randn(8, device=device)

        count_values = [256, 1024, 4096]

        for count_val in count_values:
            count = torch.tensor(count_val, device=device)

            with track_neuron_ops():
                grad_input = torch.ops.aten.batch_norm_backward_elemt(
                    grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
                )
                assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

            assert grad_input.shape == input_tensor.shape
            assert torch.all(torch.isfinite(grad_input))

    def test_batch_norm_backward_elemt_large_tensor(self):
        """Test with larger tensor sizes."""
        device = "neuron"

        grad_out = torch.randn(8, 64, 32, 32, device=device)
        input_tensor = torch.randn(8, 64, 32, 32, device=device)
        mean = torch.randn(64, device=device)
        invstd = torch.randn(64, device=device)
        weight = torch.randn(64, device=device)
        sum_dy = torch.randn(64, device=device)
        sum_dy_xmu = torch.randn(64, device=device)
        count = torch.tensor(8192, device=device)  # 8*32*32

        with track_neuron_ops():
            grad_input = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        assert grad_input.shape == input_tensor.shape
        assert torch.all(torch.isfinite(grad_input))

    def test_batch_norm_backward_elemt_single_batch(self):
        """Test with single batch dimension."""
        device = "neuron"

        grad_out = torch.randn(1, 8, 16, 16, device=device)
        input_tensor = torch.randn(1, 8, 16, 16, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        weight = torch.randn(8, device=device)
        sum_dy = torch.randn(8, device=device)
        sum_dy_xmu = torch.randn(8, device=device)
        count = torch.tensor(256, device=device)  # 1*16*16

        with track_neuron_ops():
            grad_input = torch.ops.aten.batch_norm_backward_elemt(
                grad_out, input_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
            )
            assert_op_runs_on_neuron("aten::batch_norm_backward_elemt")

        assert grad_input.shape == input_tensor.shape
