"""Test that index_add operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.utils import use_mlir_aten_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIndexAddRegistration:
    """Test index_add operation registration and functionality."""

    @pytest.mark.parametrize(
        "input_data,source_data,indices,dim",
        [
            ([5, 3], [2, 3], [0, 2], 0),
            ([3, 5], [3, 2], [1, 3], 1),
            ([4, 3, 2], [2, 3, 2], [0, 3], 0),
            ([3, 4], [3, 2], [0, 2], -1),  # negative dim
        ],
    )
    def test_index_add_runs_on_neuron(self, input_data, source_data, indices, dim):
        """Test that index_add runs on Neuron without CPU fallback"""
        input_cpu = torch.zeros(input_data)
        source_cpu = torch.ones(source_data)
        indices_cpu = torch.tensor(indices)
        expected = input_cpu.index_add(dim, indices_cpu, source_cpu)

        with track_neuron_ops():
            input_neuron = torch.zeros(input_data, device="neuron")
            source_neuron = torch.ones(source_data, device="neuron")
            indices_neuron = torch.tensor(indices, device="neuron")
            result = input_neuron.index_add(dim, indices_neuron, source_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_add")

    @pytest.mark.parametrize(
        "input_data,source_data,indices,dim",
        [
            ([5, 3], [2, 3], [0, 2], 0),
            ([3, 5], [3, 2], [1, 3], 1),
        ],
    )
    def test_index_add_inplace_runs_on_neuron(self, input_data, source_data, indices, dim):
        """Test that index_add_ runs on Neuron without CPU fallback"""
        input_cpu = torch.zeros(input_data)
        source_cpu = torch.ones(source_data)
        indices_cpu = torch.tensor(indices)
        input_cpu.index_add_(dim, indices_cpu, source_cpu)

        with track_neuron_ops():
            input_neuron = torch.zeros(input_data, device="neuron")
            source_neuron = torch.ones(source_data, device="neuron")
            indices_neuron = torch.tensor(indices, device="neuron")
            input_neuron.index_add_(dim, indices_neuron, source_neuron)

            torch.testing.assert_close(input_neuron.cpu(), input_cpu)
            assert_op_runs_on_neuron(
                "aten::index_add_" if use_mlir_aten_ops() else "aten::index_add.out"
            )

    def test_index_add_with_alpha(self):
        """Test index_add with alpha parameter"""
        input_cpu = torch.zeros(3, 4)
        source_cpu = torch.ones(2, 4)
        indices_cpu = torch.tensor([0, 2])
        dim = 0
        expected = input_cpu.index_add(dim, indices_cpu, source_cpu, alpha=2.0)

        with track_neuron_ops():
            input_neuron = torch.zeros(3, 4, device="neuron")
            source_neuron = torch.ones(2, 4, device="neuron")
            indices_neuron = torch.tensor([0, 2], device="neuron")
            result = input_neuron.index_add(dim, indices_neuron, source_neuron, alpha=2.0)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_add")

    def test_index_add_duplicate_indices(self):
        """Test index_add with duplicate indices (should accumulate)"""
        input_cpu = torch.zeros(3, 2)
        source_cpu = torch.ones(3, 2)
        indices_cpu = torch.tensor([0, 0, 1])  # Duplicate index 0
        dim = 0
        expected = input_cpu.index_add(dim, indices_cpu, source_cpu)

        with track_neuron_ops():
            input_neuron = torch.zeros(3, 2, device="neuron")
            source_neuron = torch.ones(3, 2, device="neuron")
            indices_neuron = torch.tensor([0, 0, 1], device="neuron")
            result = input_neuron.index_add(dim, indices_neuron, source_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_add")

    def test_index_add_0d_tensors(self):
        """Test index_add with 0-d input tensors"""
        indices = [0]
        dim = 0

        input_cpu = torch.tensor(2.0)  # 0D tensor with scalar value
        source_cpu = torch.tensor(3.0)  # 0D tensor with scalar value
        indices_cpu = torch.tensor(indices)
        expected = input_cpu.index_add(dim, indices_cpu, source_cpu)

        with track_neuron_ops():
            input_neuron = torch.tensor(2.0, device="neuron")
            source_neuron = torch.tensor(3.0, device="neuron")
            indices_neuron = torch.tensor(indices, device="neuron")
            result = input_neuron.index_add(dim, indices_neuron, source_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index_add")
