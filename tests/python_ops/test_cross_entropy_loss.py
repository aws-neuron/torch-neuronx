"""Test that cross_entropy operation is properly registered with PyTorch dispatcher."""

import re

import pytest
import torch
import torch.nn.functional as functional

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.xfail(
    reason="Enable once we support cross_entropy loss, change was "
    "reverted due to missing backward registration"
)
@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCrossEntropy:
    def setup(self):
        """Set up test environment before each test method."""
        torch.manual_seed(42)

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_cross_entropy_reduction(self, reduction):
        """Test cross_entropy with different reduction modes."""
        logits_cpu = torch.tensor(
            [[2.0, 1.5, 0.3, -1.0, 0.5], [0.5, 0.8, 1.2, 2.0, -0.1], [0.1, -0.3, 0.7, 1.8, -0.2]],
            requires_grad=True,
        )
        targets_cpu = torch.tensor([0, 3, 3], dtype=torch.long)

        # Get CPU result
        loss_cpu = functional.cross_entropy(logits_cpu, targets_cpu, reduction=reduction)

        # Create identical tensors on Neuron device
        logits_neuron = logits_cpu.detach().clone().to("neuron")
        targets_neuron = targets_cpu.detach().clone().to("neuron")

        # Run on Neuron device and track ops
        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(
                logits_neuron, targets_neuron, reduction=reduction
            )
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        # Check results match
        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

        # Verify expected shape based on reduction
        expected_shape = torch.Size([3]) if reduction == "none" else torch.Size([])
        assert loss_cpu.size() == expected_shape
        assert loss_neuron.size() == expected_shape

    def test_cross_entropy_with_weights(self):
        """Test cross_entropy with class weights."""
        logits = torch.tensor(
            [[2.0, 0.5, 1.0], [0.5, 1.0, 2.0], [0.1, 0.7, 3.0]], requires_grad=True
        )
        targets = torch.tensor([0, 2, 1], dtype=torch.long)
        weight = torch.tensor([0.1, 1.0, 2.0])  # Class 2 is 20x more important than class 0

        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets, weight=weight)
        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")
        weight_neuron = weight.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(
                logits_neuron, targets_neuron, weight=weight_neuron
            )
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
    def test_cross_entropy_label_smoothing(self, smoothing):
        """Test cross_entropy with label smoothing."""
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]], requires_grad=True)
        targets = torch.tensor([0, 2], dtype=torch.long)

        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets, label_smoothing=smoothing)

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(
                logits_neuron, targets_neuron, label_smoothing=smoothing
            )
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    def test_cross_entropy_ignore_index(self):
        """Test cross_entropy with ignore_index."""
        logits = torch.tensor(
            [[2.0, 0.5, 1.0], [0.5, 1.0, 2.0], [0.1, 0.7, 3.0]], requires_grad=True
        )
        targets = torch.tensor([0, -100, 1], dtype=torch.long)

        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets, ignore_index=-100)

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron, ignore_index=-100)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    def test_cross_entropy_probability_targets(self):
        """Test cross_entropy with probability targets."""
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]], requires_grad=True)

        # Create soft targets (probability distributions)
        targets = torch.tensor(
            [
                [0.2, 0.7, 0.1],  # First sample is mostly class 1
                [0.05, 0.15, 0.8],  # Second sample is mostly class 2
            ]
        )

        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets)

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3, 2, 2),  # 2D spatial (e.g., image)
            (2, 3, 4, 5, 2),  # 3D spatial (e.g., video)
        ],
    )
    def test_cross_entropy_multidimensional(self, shape):
        """Test cross_entropy with multi-dimensional inputs."""
        # Create multi-dimensional logits
        logits = torch.randn(shape, requires_grad=True)

        # Create corresponding targets (without the channel dimension)
        target_shape = (shape[0],) + shape[2:]
        targets = torch.randint(0, shape[1], target_shape, dtype=torch.long)
        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets)

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    def test_cross_entropy_all_params_with_indices(self):
        """Test cross_entropy with all parameters active (using class indices)."""
        logits = torch.tensor(
            [[2.0, 0.5, 1.0, -0.3], [0.5, 1.0, 2.0, 0.7], [0.1, 0.7, 3.0, 1.2]], requires_grad=True
        )
        targets = torch.tensor([0, -100, 3], dtype=torch.long)  # One target is ignored
        weight = torch.tensor([0.1, 1.0, 2.0, 0.5])  # Custom weights for all classes

        # CPU computation with all parameters active
        loss_cpu = functional.cross_entropy(
            logits, targets, weight=weight, ignore_index=-100, reduction="sum", label_smoothing=0.2
        )

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")
        weight_neuron = weight.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(
                logits_neuron,
                targets_neuron,
                weight=weight_neuron,
                ignore_index=-100,
                reduction="sum",
                label_smoothing=0.2,
            )
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    def test_cross_entropy_all_params_with_probs(self):
        """Test cross_entropy with all parameters active (using probability targets)."""
        logits = torch.tensor(
            [
                [2.0, 0.5, 1.0, -0.3],
                [0.5, 1.0, 2.0, 0.7],
            ],
            requires_grad=True,
        )

        # Probability targets
        targets = torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.05, 0.15, 0.5, 0.3]])

        weight = torch.tensor([0.1, 1.0, 2.0, 0.5])  # Custom weights for all classes

        # CPU computation with all parameters active
        loss_cpu = functional.cross_entropy(
            logits, targets, weight=weight, reduction="sum", label_smoothing=0.1
        )

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")
        weight_neuron = weight.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(
                logits_neuron,
                targets_neuron,
                weight=weight_neuron,
                reduction="sum",
                label_smoothing=0.1,
            )
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=1e-4, atol=1e-4)

    @assert_raises(RuntimeError, match="Expected target size")
    def test_cross_entropy_invalid_target_shape(self):
        """Test cross_entropy with invalid target shape (should fail with same error)."""
        logits = torch.randn((2, 3, 4, 4), requires_grad=True)  # [N, C, H, W]

        # Invalid target shape - missing spatial dimension
        targets = torch.tensor([0, 1], dtype=torch.long)

        # Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron)

    @assert_raises((IndexError, RuntimeError), match="out of range")
    def test_cross_entropy_out_of_range_targets(self):
        """Test cross_entropy with targets outside valid class range."""
        logits = torch.randn((2, 3), requires_grad=True)

        # Target class 5 is invalid (only 0,1,2 are valid)
        targets = torch.tensor([1, 5], dtype=torch.long)

        # Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron)

    @assert_raises(RuntimeError, match="weight tensor should be defined either for all")
    def test_cross_entropy_invalid_weight_size(self):
        """Test cross_entropy with invalid weight size."""
        logits = torch.randn((2, 3), requires_grad=True)
        targets = torch.tensor([1, 2], dtype=torch.long)

        # Weight should have size 3, but has size 2
        weight = torch.tensor([0.5, 1.5])

        # Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")
        weight_neuron = weight.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron, weight=weight_neuron)

    @assert_raises((RuntimeError, Exception), match="Expected target size")
    def test_cross_entropy_probability_shape_mismatch(self):
        """Test cross_entropy when probability targets have wrong shape."""
        logits = torch.randn((2, 3), requires_grad=True)

        # Target has wrong number of classes (4 instead of 3)
        targets = torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])

        # Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron)

    @assert_raises((ValueError, Exception), match="is not a valid value for reduction")
    def test_cross_entropy_invalid_reduction(self):
        """Test cross_entropy with invalid reduction mode."""
        logits = torch.randn((2, 3), requires_grad=True)
        targets = torch.tensor([0, 1], dtype=torch.long)

        # Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron, reduction="invalid")

    def test_cross_entropy_all_ignored(self):
        """Test cross_entropy when all targets are ignored."""
        logits = torch.randn((3, 4), requires_grad=True)
        targets = torch.tensor([-100, -100, -100], dtype=torch.long)  # All targets are ignore_index

        # CPU computation
        loss_cpu = functional.cross_entropy(logits, targets, ignore_index=-100)

        # Neuron computation
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron, ignore_index=-100)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        # Verify both return NaN
        assert torch.isnan(loss_cpu)
        assert torch.isnan(loss_neuron.cpu())

    @assert_raises((RuntimeError, Exception), match="Expected target size")
    def test_cross_entropy_incompatible_spatial_dimensions(self):
        """Test cross_entropy with incompatible spatial dimensions."""
        logits = torch.randn((2, 3, 4, 4), requires_grad=True)  # [N, C, H, W]

        # Target has wrong spatial dimensions (5x5 instead of 4x4)
        targets = torch.randint(0, 3, (2, 5, 5), dtype=torch.long)

        # Now Verify that Neuron raises error
        logits_neuron = logits.detach().clone().to("neuron")
        targets_neuron = targets.detach().clone().to("neuron")

        functional.cross_entropy(logits_neuron, targets_neuron)
