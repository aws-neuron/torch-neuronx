"""Test that one_hot operation is properly registered with PyTorch dispatcher."""

import re

import pytest
import torch
import torch.nn.functional as functional

import torch_neuron
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuron.device_count() == 0, reason="No Neuron devices available")
class TestOneHot:
    def setup(self):
        """Set up test environment before each test method."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)

    def test_one_hot_basic(self):
        """Test basic one_hot functionality."""
        with track_neuron_ops():
            indices = torch.tensor([0, 1, 2], dtype=torch.int64).to("neuron")
            num_classes = 3
            result = functional.one_hot(indices, num_classes)
            assert result.size() == torch.Size([3, 3])
            assert_op_runs_on_neuron("aten::one_hot")

    @pytest.mark.xfail(
        reason="Can't support meta tensor evaluation requiring accessing meta tensor content"
    )
    def test_one_hot_inferred_classes(self):
        """Test one_hot with automatically inferred num_classes (-1)."""
        with track_neuron_ops():
            indices = torch.tensor([0, 2, 1, 3], dtype=torch.int64).to("neuron")
            # Default num_classes is -1 (auto-infer)
            result = functional.one_hot(indices)
            assert result.size() == torch.Size([4, 4])
            assert_op_runs_on_neuron("aten::one_hot")

    @pytest.mark.parametrize(
        "indices_shape, num_classes, expected_shape",
        [
            # 1D tensor case
            ((5,), 3, (5, 3)),
            # 2D tensor case
            ((2, 3), 4, (2, 3, 4)),
            # 3D tensor case
            ((2, 3, 2), 5, (2, 3, 2, 5)),
        ],
    )
    def test_one_hot_different_dims(self, indices_shape, num_classes, expected_shape):
        """Test one_hot on tensors with different numbers of dimensions."""
        with track_neuron_ops():
            x = torch.randint(0, num_classes, indices_shape, dtype=torch.int64).to("neuron")
            result = functional.one_hot(x, num_classes)
            assert result.size() == torch.Size(expected_shape)
            assert_op_runs_on_neuron("aten::one_hot")

    def test_one_hot_data_correctness(self):
        """Test that one_hot correctly produces expected output."""
        # Create and compute one_hot on CPU first
        indices_cpu = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
        expected = functional.one_hot(indices_cpu, 4)

        # Now test on neuron and compare with CPU result
        with track_neuron_ops():
            indices = indices_cpu.to("neuron")
            result = functional.one_hot(indices, 4)
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::one_hot")

    def test_one_hot_type_error(self):
        """Test that one_hot raises an error for non-int64 inputs like PyTorch."""
        # First check CPU behavior to get the expected error message
        indices_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)

        # Verify that CPU PyTorch raises a RuntimeError for non-int64 input
        cpu_error_message = None
        try:
            functional.one_hot(indices_cpu, 3)
        except RuntimeError as e:
            cpu_error_message = str(e)

        # Now verify that Neuron matches the CPU behavior
        with track_neuron_ops():
            indices = indices_cpu.to("neuron")
            with pytest.raises(RuntimeError, match=re.escape(cpu_error_message)):
                functional.one_hot(indices, 3)

    def test_one_hot_zero_classes(self):
        """Test one_hot with num_classes=0 raises expected error."""
        # First check CPU behavior to get expected error message
        indices_cpu = torch.tensor([0, 1, 2], dtype=torch.int64)

        try:
            functional.one_hot(indices_cpu, num_classes=0)
        except RuntimeError as e:
            cpu_error_message = str(e)

        # Now test on neuron
        with track_neuron_ops():
            indices = indices_cpu.to("neuron")

            with pytest.raises(RuntimeError, match=re.escape(cpu_error_message)):
                functional.one_hot(indices, num_classes=0)

    def test_one_hot_out_of_range_error(self):
        """Test that one_hot raises an error for indices larger than num_classes."""
        # First check CPU behavior to get expected error message
        indices_cpu = torch.tensor([3, 4, 1, 0], dtype=torch.int64)

        try:
            functional.one_hot(indices_cpu, 3)
        except RuntimeError as e:
            expected_error = str(e)

        # Now verify the same behavior on neuron
        with track_neuron_ops():
            indices = indices_cpu.to("neuron")

            with pytest.raises(RuntimeError, match=re.escape(expected_error)):
                functional.one_hot(indices, 3)

    def test_one_hot_negative_indices(self):
        """Test one_hot with negative indices raises an error."""
        # First check CPU behavior to get expected error message
        indices_cpu = torch.tensor([-1, 0, 1], dtype=torch.int64)

        try:
            functional.one_hot(indices_cpu, 3)
        except RuntimeError as e:
            expected_error = str(e)

        # Now test on neuron
        with track_neuron_ops():
            indices = indices_cpu.to("neuron")

            with pytest.raises(RuntimeError, match=re.escape(expected_error)):
                functional.one_hot(indices, 3)

    def test_one_hot_scalar_tensor(self):
        """Test one_hot on a scalar tensor."""
        # First get CPU behavior as reference
        index_cpu = torch.tensor(2, dtype=torch.int64)
        expected = functional.one_hot(index_cpu, 5)

        # Now test on neuron
        with track_neuron_ops():
            index = index_cpu.to("neuron")
            result = functional.one_hot(index, 5)

            # Compare with CPU result
            assert result.size() == expected.size()
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::one_hot")
