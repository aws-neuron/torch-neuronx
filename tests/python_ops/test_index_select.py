import os
import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops


class TestIndexSelect:
    def test_index_select_basic(self):
        """Test index select basic"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        input_neuron.requires_grad = True
        indices = torch.tensor([0, 2])

        out_cpu = torch.index_select(input, dim=0, index=indices)

        with track_neuron_ops():
            out_neuron = torch.index_select(input_neuron, dim=0, index=indices.to(device))
            assert_op_runs_on_neuron("aten::index_select")

        assert out_neuron.grad_fn is not None
        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_index_select_one_tensor_not_on_device(self):
        """Test index select with indices not on device"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2])

        # When indices are not on device, can_handle returns False
        # Operation should fail as it does for GPU if no fallback is active
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            torch.index_select(input_neuron, dim=0, index=indices)

    def test_index_select_scalar_index(self):
        """Test index select with scalar index tensor"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)

        indices_scalar = torch.tensor(0, dtype=torch.int32)

        with track_neuron_ops():
            result = torch.index_select(input_neuron, dim=0, index=indices_scalar.to(device))
            assert_op_runs_on_neuron("aten::index_select")

        assert result.shape == torch.Size([1, 4])

    @pytest.mark.xfail(reason="produces nan instead of OOB error")
    @assert_raises(IndexError, match="index out of range in self")
    def test_index_select_with_indices_out_of_range(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.ones((3, 4))
        indices = torch.tensor([10])

        torch.index_select(input.to(device), dim=0, index=indices.to(device))

    @assert_raises(
        TypeError if use_mlir_aten_ops() else ValueError,
        match=(
            re.escape("index_select(): Expected dtype int32 or int64 for index")
            if use_mlir_aten_ops()
            else "indices must have an integer type"
        ),
    )
    def test_index_select_with_index_dtype_float(self):
        """Test index select with index dtype float"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2], dtype=torch.float32)

        # When indices have float dtype, can_handle returns False
        # This triggers CPU fallback which fails with RuntimeError about dtype
        torch.index_select(input_neuron, dim=0, index=indices.to(device))

    @assert_raises(
        IndexError,
        match=(
            re.escape("index_select(): Index is supposed to be a vector")
            if use_mlir_aten_ops()
            else re.escape("index_select(): Expected 1-D index tensor, got dim=2")
        ),
    )
    def test_index_select_with_incorrect_indices_size(self):
        """Test index select with incorrect indices size"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.ones((2, 2), dtype=torch.int32)

        # When indices have incorrect size (len(index.shape) > 1), can_handle returns False
        # This triggers CPU fallback which fails with IndexError about vector requirement
        torch.index_select(input_neuron, dim=0, index=indices.to(device))

    @assert_raises(TypeError, match=r"received an invalid combination of arguments")
    def test_index_select_with_incorrect_value(self):
        """Test index select with index dtype float"""

        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2], dtype=torch.int32)

        torch.index_select(input_neuron, dim=1.0, index=indices.to(device))
