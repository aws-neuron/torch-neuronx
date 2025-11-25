import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises, cpu_fallback_test


class TestIndexCopy:
    def test_index_copy_basic(self):
        """Test index copy basic"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        out_cpu = input.index_copy(0, indices, source)
        out_neuron = input_neuron.index_copy(0, indices_neuron, source_neuron)

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron("aten::index_copy")

    def test_index_copy_in_place(self):
        """Test index copy in place"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input.index_copy(0, indices, source)
        input_neuron.index_copy(0, indices_neuron, source_neuron)

        torch.testing.assert_close(input_neuron.cpu(), input)
        assert_op_runs_on_neuron("aten::index_copy")

    def test_index_copy_one_tensor_not_on_device(self):
        """Test index select with indices not on device"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input_neuron.index_copy(0, indices_neuron, source)
        assert "aten::index_copy" in torch_neuronx.get_fallback_ops()

    @assert_raises(
        IndexError,
        match=r"index_copy_\(\): When source and destination are not scalars, their "
        r"dimensionality must match. Source dimensionality \(2\), destination dimensionality \(1\)",
    )
    def test_index_copy_with_input_empty(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.empty((0,))
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input_neuron.index_copy(0, indices_neuron, source)

    @assert_raises(
        ValueError,
        match=r"Incompatible shapes for broadcasting: \(3, 3\) and requested shape \(3, 4\)",
    )
    def test_index_copy_with_different_slice_shapes(self):
        """Test index select with different input and source shapes"""

        device = "neuron"
        input = torch.zeros(5, 4)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input_neuron.index_copy(0, indices_neuron, source_neuron)

    @assert_raises(IndexError, match="Dimension out of range")
    def test_index_copy_with_dimension_out_of_range(self):
        """Test index select with dim out of range"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input_neuron.index_copy(2, indices_neuron, source)

    @cpu_fallback_test
    @assert_raises((IndexError, RuntimeError), match=r".*index_copy[\s\S]*out of bound.*")
    def test_index_copy_with_indices_out_of_range(self):
        """Test index select with indices out of range"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([0, 6, 2])
        indices_neuron = indices.to(device)
        input_neuron.index_copy(0, indices_neuron, source_neuron)

    @cpu_fallback_test
    @assert_raises(
        RuntimeError, match=r"index_copy_\(\): Expected a long tensor for index, but got Float"
    )
    def test_index_copy_empty_indices(self):
        """Test index select with empty index"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([])
        indices_neuron = indices.to(device)

        # CPU error message does not seem to be descriptive neither
        input_neuron.index_copy(0, indices_neuron, source_neuron)

    @assert_raises(
        TypeError,
        match="Indexer must have integer or boolean type, "
        "got indexer with type float32 at position 0",
    )
    def test_index_copy_with_index_dtype_float(self):
        """Test index select with float type index"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        source_neuron = source.to(device)
        indices = torch.tensor([0, 4, 2], dtype=torch.float)
        indices_neuron = indices.to(device)

        input_neuron.index_copy(0, indices_neuron, source_neuron)
