import os

import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    assert_op_runs_on_neuron,
    cpu_fallback_test,
    track_neuron_ops,
)


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

    @pytest.mark.xfail(
        condition=os.environ.get("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1",
        reason="CPU fallback is disabled with this env var",
    )
    def test_index_select_one_tensor_not_on_device(self):
        """Test index select with indices not on device
        - CPU fallback when indices are not on device"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2])

        # When indices are not on device, can_handle returns False
        # Operation should fall back to CPU
        _ = torch.index_select(input_neuron, dim=0, index=indices)

        # Verify that the operation was offloaded to CPU
        assert "aten::index_select" in torch_neuron.get_fallback_ops()

    @pytest.mark.xfail(
        condition=os.environ.get("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1",
        reason="Empty input will fail the op in Neuron and fall back to CPU, but CPU fallback is disabled with this env var",  # noqa E501
    )
    def test_index_select_with_input_empty(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.empty((0,))
        indices = torch.tensor([0])

        with pytest.raises(IndexError, match="index out of range in self"):
            torch.index_select(input.to(device), dim=0, index=indices.to(device))

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
    def test_index_select_with_indices_out_of_range(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.ones((3, 4))
        indices = torch.tensor([10])

        with pytest.raises(IndexError, match="index out of range in self"):
            torch.index_select(input.to(device), dim=0, index=indices.to(device))

    @cpu_fallback_test
    def test_index_select_empty_indices(self):
        """Test index select with empty indices"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.empty((0,), dtype=torch.int32)

        # empty tensor handler should handle this without running on neuron or fall back to CPU
        with (
            track_neuron_ops(),
            pytest.raises(
                AssertionError,
                match="NKI kernel variant of aten::index_select ran on Neuron instead of falling back to CPU",  # noqa E501
            ),
        ):
            result = torch.index_select(input_neuron, dim=0, index=indices.to(device))
            assert result.numel() == 0
            assert_op_falls_back_on_cpu("aten::index_select")

    def test_index_select_with_index_dtype_float(self):
        """Test index select with index dtype float"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2], dtype=torch.float32)

        # When indices have float dtype, can_handle returns False
        # This triggers CPU fallback which fails with RuntimeError about dtype
        with pytest.raises(ValueError, match="indices must have an integer type"):
            torch.index_select(input_neuron, dim=0, index=indices.to(device))

    def test_index_select_with_incorrect_indices_size(self):
        """Test index select with incorrect indices size"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.ones((2, 2), dtype=torch.int32)

        # When indices have incorrect size (len(index.shape) > 1), can_handle returns False
        # This triggers CPU fallback which fails with IndexError about vector requirement
        with pytest.raises(
            IndexError, match="index_select\\(\\): Expected 1-D index tensor, got dim=2"
        ):
            torch.index_select(input_neuron, dim=0, index=indices.to(device))

    def test_index_select_with_incorrect_value(self):
        """Test index select with index dtype float"""

        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.tensor([0, 2], dtype=torch.int32)

        with pytest.raises(
            TypeError,
            match=r"received an invalid combination of arguments",
        ):
            torch.index_select(input_neuron, dim=1.0, index=indices.to(device))
