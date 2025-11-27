"""Test batch_norm_gather_stats_with_counts operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestBatchNormGatherStatsWithCounts:
    """Test batch_norm_gather_stats_with_counts operation."""

    def test_batch_norm_gather_stats_with_counts_basic(self):
        """Test basic batch_norm_gather_stats_with_counts functionality."""
        device = "neuron"

        # Create input tensors - use 2D tensors for means/invstds as per working example
        input_tensor = torch.randn(3, 2, 28, 28, device=device)
        means = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device=device)  # 2D: [devices, features]
        invstds = torch.tensor(
            [[1.0, 0.5], [0.7071, 0.4082]], device=device
        )  # 2D: [devices, features]
        running_mean = torch.zeros(2, device=device)
        running_var = torch.ones(2, device=device)
        momentum = 0.1
        eps = 1e-5
        counts = torch.tensor([10.0, 20.0], dtype=torch.float32, device=device)  # 1D: [devices]

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_gather_stats_with_counts(
                input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
            )
            assert_op_runs_on_neuron("aten::batch_norm_gather_stats_with_counts")

        # Verify outputs
        assert mean_out.device.type == "neuron"
        assert invstd_out.device.type == "neuron"
        assert mean_out.shape == (2,)  # Features dimension
        assert invstd_out.shape == (2,)

    def test_batch_norm_gather_stats_with_counts_different_shapes(self):
        """Test with different input shapes."""
        device = "neuron"

        # Test with different number of features
        input_tensor = torch.randn(2, 4, 16, 16, device=device)
        means = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], device=device
        )  # 2D: [devices, features]
        invstds = torch.tensor(
            [[1.0, 0.5, 0.33, 0.25], [0.8, 0.6, 0.4, 0.2]], device=device
        )  # 2D: [devices, features]
        running_mean = torch.zeros(4, device=device)
        running_var = torch.ones(4, device=device)
        momentum = 0.05
        eps = 1e-4
        counts = torch.tensor([32.0, 64.0], dtype=torch.float32, device=device)  # 1D: [devices]

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_gather_stats_with_counts(
                input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
            )
            assert_op_runs_on_neuron("aten::batch_norm_gather_stats_with_counts")

        assert mean_out.shape == (4,)
        assert invstd_out.shape == (4,)

    def test_batch_norm_gather_stats_with_counts_multiple_calls(self):
        """Test multiple calls with same inputs."""
        device = "neuron"

        input_tensor = torch.randn(3, 2, 28, 28, device=device)
        means = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device=device)
        invstds = torch.tensor([[1.0, 0.5], [0.7071, 0.4082]], device=device)
        running_mean = torch.zeros(2, device=device)
        running_var = torch.ones(2, device=device)
        momentum = 0.1
        eps = 1e-5
        counts = torch.tensor([10.0, 20.0], dtype=torch.float32, device=device)

        with track_neuron_ops():
            mean_out1, invstd_out1 = torch.ops.aten.batch_norm_gather_stats_with_counts(
                input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
            )
            mean_out2, invstd_out2 = torch.ops.aten.batch_norm_gather_stats_with_counts(
                input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
            )
            assert_op_runs_on_neuron("aten::batch_norm_gather_stats_with_counts")

        # Results should be identical
        torch.testing.assert_close(mean_out1, mean_out2)
        torch.testing.assert_close(invstd_out1, invstd_out2)

    def test_batch_norm_gather_stats_with_counts_zero_counts(self):
        """Test with zero counts (edge case)."""
        device = "neuron"

        input_tensor = torch.randn(2, 3, 16, 16, device=device)
        means = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], device=device)
        invstds = torch.tensor([[1.0, 0.5, 0.33], [1.0, 1.0, 1.0]], device=device)
        running_mean = torch.zeros(3, device=device)
        running_var = torch.ones(3, device=device)
        momentum = 0.1
        eps = 1e-5
        counts = torch.tensor(
            [16.0, 0.0], dtype=torch.float32, device=device
        )  # Second device has zero count

        with track_neuron_ops():
            mean_out, invstd_out = torch.ops.aten.batch_norm_gather_stats_with_counts(
                input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
            )
            assert_op_runs_on_neuron("aten::batch_norm_gather_stats_with_counts")

        # Verify basic properties (shape and device)
        assert mean_out.shape == (3,)
        assert invstd_out.shape == (3,)
        assert mean_out.device.type == "neuron"
        assert invstd_out.device.type == "neuron"
        # Note: With zero counts, some values may be NaN which is expected behavior
