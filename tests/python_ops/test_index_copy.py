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
        (ValueError, IndexError),
        match=r"(Incompatible shapes for broadcasting: \(3, 3\) and requested shape \(3,\)|"
        r"dimensionality must match)",
    )
    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(),
        reason="CPU dispatch throws different error than Python lowering path.",
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bfloat16, torch.bool])
    def test_index_copy_0d(self, dtype):
        """0-d (scalar) tensor index_copy must match CPU behavior."""
        x_cpu = torch.tensor(5, dtype=dtype) if dtype != torch.bool else torch.tensor(True)
        idx_cpu = torch.tensor([0], dtype=torch.int64)
        src_cpu = torch.tensor(3, dtype=dtype) if dtype != torch.bool else torch.tensor(False)

        expected = torch.index_copy(x_cpu, 0, idx_cpu, src_cpu)

        x_n = x_cpu.to("neuron")
        idx_n = idx_cpu.to("neuron")
        src_n = src_cpu.to("neuron")
        result = torch.index_copy(x_n, 0, idx_n, src_n).cpu()

        torch.testing.assert_close(result, expected)

    @assert_raises(
        IndexError,
        match=r"When source is scalar, index should have one element \(got 2\)",
    )
    def test_index_copy_0d_multi_index_scalar_source(self):
        """index_copy on a 0-d destination with a 0-d source must reject
        multi-element indexes, matching CPU PyTorch behavior."""
        device = "neuron"
        x = torch.tensor(5.0, device=device)
        idx = torch.tensor([0, 0], dtype=torch.int64, device=device)
        src = torch.tensor(3.0, device=device)
        torch.index_copy(x, 0, idx, src)
