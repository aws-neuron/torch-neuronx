import os
import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises
from torch_neuronx.utils import use_mlir_aten_ops


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

        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            input_neuron.index_copy(0, indices_neuron, source)

    @assert_raises(
        RuntimeError,
        match=r"index_copy_\(\): When source and destination are not scalars, their "
        r"dimensionality must match. Source dimensionality \(2\), destination dimensionality \(1\)",
    )
    @pytest.mark.xfail(
        condition=use_mlir_aten_ops(),
        reason="MLIR throws broadcasting error.",
    )
    def test_index_copy_with_input_empty(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.empty((0,))
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)
        input_neuron.index_copy(0, indices_neuron, source)

    @assert_raises(
        ValueError,
        match=re.escape("Incompatible shapes for broadcasting: (3, 3) and requested shape (3,)"),
    )
    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(),
        reason="CPU throws Value error.",
    )
    def test_index_copy_with_input_empty_broadcast(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.empty((0,))
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)
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

    @assert_raises(IndexError, match="(Dimension out of range|list assignment index out of range)")
    def test_index_copy_with_dimension_out_of_range(self):
        """Test index select with dim out of range"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)
        indices = torch.tensor([0, 4, 2])
        indices_neuron = indices.to(device)

        input_neuron.index_copy(2, indices_neuron, source)
