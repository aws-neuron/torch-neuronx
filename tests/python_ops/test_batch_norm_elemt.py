"""Test batch_norm_elemt operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestBatchNormElemt:
    """Test batch_norm_elemt operation."""

    def test_batch_norm_elemt_basic(self):
        """Test basic batch_norm_elemt functionality."""
        device = "neuron"

        # Create input tensors
        input_tensor = torch.randn(4, 8, 16, 16, device=device)
        weight = torch.randn(8, device=device)
        bias = torch.randn(8, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        eps = 1e-5

        with track_neuron_ops():
            output = torch.ops.aten.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        # Verify output
        assert output.device.type == "neuron"
        assert output.shape == input_tensor.shape

    def test_batch_norm_elemt_no_weight_bias(self):
        """Test batch_norm_elemt without weight and bias."""
        device = "neuron"

        input_tensor = torch.randn(2, 16, 32, 32, device=device)
        mean = torch.randn(16, device=device)
        invstd = torch.randn(16, device=device)
        eps = 1e-5

        with track_neuron_ops():
            output = torch.ops.aten.batch_norm_elemt(input_tensor, None, None, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        assert output.shape == input_tensor.shape

    def test_batch_norm_elemt_3d_input(self):
        """Test with 3D input tensor."""
        device = "neuron"

        input_tensor = torch.randn(8, 32, 64, device=device)
        weight = torch.randn(32, device=device)
        bias = torch.randn(32, device=device)
        mean = torch.randn(32, device=device)
        invstd = torch.randn(32, device=device)
        eps = 1e-4

        with track_neuron_ops():
            output = torch.ops.aten.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        assert output.shape == input_tensor.shape

    def test_batch_norm_elemt_multiple_calls(self):
        """Test multiple calls with same inputs."""
        device = "neuron"
        input_tensor = torch.randn(2, 4, 8, 8, device=device)
        weight = torch.randn(4, device=device)
        bias = torch.randn(4, device=device)
        mean = torch.randn(4, device=device)
        invstd = torch.randn(4, device=device)
        eps = 1e-5

        with track_neuron_ops():
            output1 = torch.ops.aten.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
            output2 = torch.ops.aten.batch_norm_elemt(input_tensor, weight, bias, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        # Results should be identical
        torch.testing.assert_close(output1, output2)

    def test_batch_norm_elemt_different_eps(self):
        """Test with different epsilon values."""
        device = "neuron"
        input_tensor = torch.randn(4, 8, 16, 16, device=device)
        weight = torch.randn(8, device=device)
        bias = torch.randn(8, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)

        eps_values = [1e-3, 1e-5, 1e-7]

        for eps in eps_values:
            with track_neuron_ops():
                output = torch.ops.aten.batch_norm_elemt(
                    input_tensor, weight, bias, mean, invstd, eps
                )
                assert_op_runs_on_neuron("aten::batch_norm_elemt")

            assert output.shape == input_tensor.shape
            assert torch.all(torch.isfinite(output))

    def test_batch_norm_elemt_weight_only(self):
        """Test with weight but no bias."""
        device = "neuron"

        input_tensor = torch.randn(2, 8, 16, 16, device=device)
        weight = torch.randn(8, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        eps = 1e-5

        with track_neuron_ops():
            output = torch.ops.aten.batch_norm_elemt(input_tensor, weight, None, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        assert output.shape == input_tensor.shape

    def test_batch_norm_elemt_bias_only(self):
        """Test with bias but no weight."""
        device = "neuron"

        input_tensor = torch.randn(2, 8, 16, 16, device=device)
        bias = torch.randn(8, device=device)
        mean = torch.randn(8, device=device)
        invstd = torch.randn(8, device=device)
        eps = 1e-5

        with track_neuron_ops():
            output = torch.ops.aten.batch_norm_elemt(input_tensor, None, bias, mean, invstd, eps)
            assert_op_runs_on_neuron("aten::batch_norm_elemt")

        assert output.shape == input_tensor.shape
