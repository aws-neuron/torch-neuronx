"""Test batch_norm_stats operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestBatchNormStats:
    """Test batch_norm_stats operation."""

    def test_batch_norm_stats_basic(self):
        """Test basic batch_norm_stats functionality."""
        device = "neuron"

        # Create input tensor (N, C, H, W)
        input_tensor = torch.randn(4, 16, 32, 32, device=device)
        eps = 1e-5

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        # Verify outputs
        assert mean_out.device.type == "neuron"
        assert invstd_out.device.type == "neuron"
        assert mean_out.shape == (16,)  # Channel dimension
        assert invstd_out.shape == (16,)

    def test_batch_norm_stats_3d_input(self):
        """Test with 3D input tensor."""
        device = "neuron"

        # Create 3D input tensor (N, C, L)
        input_tensor = torch.randn(8, 32, 128, device=device)
        eps = 1e-4

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        assert mean_out.shape == (32,)
        assert invstd_out.shape == (32,)

    def test_batch_norm_stats_5d_input(self):
        """Test with 5D input tensor."""
        device = "neuron"

        # Create 5D input tensor (N, C, D, H, W)
        input_tensor = torch.randn(2, 8, 4, 16, 16, device=device)
        eps = 1e-6

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        assert mean_out.shape == (8,)
        assert invstd_out.shape == (8,)

    def test_batch_norm_stats_multiple_calls(self):
        """Test multiple calls with same inputs."""
        device = "neuron"
        input_tensor = torch.randn(2, 4, 8, 8, device=device)
        eps = 1e-5

        with track_neuron_ops():
            mean_out1, invstd_out1 = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            mean_out2, invstd_out2 = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        # Results should be identical
        torch.testing.assert_close(mean_out1, mean_out2)
        torch.testing.assert_close(invstd_out1, invstd_out2)

    def test_batch_norm_stats_different_eps(self):
        """Test with different epsilon values."""
        device = "neuron"
        input_tensor = torch.randn(4, 8, 16, 16, device=device)

        eps_values = [1e-3, 1e-5, 1e-7]

        for eps in eps_values:
            with track_neuron_ops():
                mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
                assert_op_runs_on_neuron("aten::batch_norm_stats")

            assert mean_out.shape == (8,)
            assert invstd_out.shape == (8,)
            assert torch.all(torch.isfinite(mean_out))
            assert torch.all(torch.isfinite(invstd_out))

    def test_batch_norm_stats_large_tensor(self):
        """Test with larger tensor sizes."""
        device = "neuron"
        input_tensor = torch.randn(8, 64, 32, 32, device=device)
        eps = 1e-6

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        assert mean_out.shape == (64,)
        assert invstd_out.shape == (64,)
        assert torch.all(torch.isfinite(mean_out))
        assert torch.all(torch.isfinite(invstd_out))

    def test_batch_norm_stats_single_batch(self):
        """Test with single batch dimension."""
        device = "neuron"
        input_tensor = torch.randn(1, 8, 16, 16, device=device)
        eps = 1e-5

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_stats(input_tensor, eps)
            assert_op_runs_on_neuron("aten::batch_norm_stats")

        assert mean_out.shape == (8,)
        assert invstd_out.shape == (8,)
