"""Test that index operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    assert_op_runs_on_neuron,
    cpu_fallback_test,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuron.device_count() == 0, reason="No Neuron devices available")
class TestIndexRegistration:
    """Test index operation registration and functionality."""

    @pytest.mark.parametrize(
        "input_data,index_data",
        [
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1]),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1]),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [-1]),
        ],
    )
    def test_basic_index_operations_run_on_neuron(self, input_data, index_data):
        """Test that basic index operations run on Neuron without CPU fallback"""
        x_cpu = torch.tensor(input_data)
        index_cpu = torch.tensor(index_data)
        expected = x_cpu[index_cpu]

        with track_neuron_ops():
            x = torch.tensor(input_data, device="neuron")
            index = torch.tensor(index_data, device="neuron")
            result = x[index]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index.Tensor")

    @pytest.mark.parametrize(
        "input_data,row_indices,col_indices",
        [
            (
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [0, 2],
                [1, 3],
            ),
            (
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [-1, 0],
                [0, -1],
            ),
        ],
    )
    def test_multidimensional_index_operations_run_on_neuron(
        self, input_data, row_indices, col_indices
    ):
        """Test that multidimensional index operations run on Neuron without CPU fallback"""
        x_cpu = torch.tensor(input_data)
        row_idx_cpu = torch.tensor(row_indices)
        col_idx_cpu = torch.tensor(col_indices)
        expected = x_cpu[row_idx_cpu, col_idx_cpu]

        with track_neuron_ops():
            x = torch.tensor(input_data, device="neuron")
            row_idx = torch.tensor(row_indices, device="neuron")
            col_idx = torch.tensor(col_indices, device="neuron")
            result = x[row_idx, col_idx]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index.Tensor")

    @pytest.mark.parametrize(
        "input_data,depth_indices,row_indices,col_indices",
        [
            ([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [0], None, [1]),
            ([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], None, [0], [1]),
            ([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], None, [0], None),
        ],
    )
    def test_mixed_index_with_none_run_on_neuron(
        self, input_data, depth_indices, row_indices, col_indices
    ):
        """Test that 3D mixed indexing with None runs on Neuron without CPU fallback"""
        x_cpu = torch.tensor(input_data)
        d_idx_cpu = slice(None) if depth_indices is None else torch.tensor(depth_indices)
        r_idx_cpu = slice(None) if row_indices is None else torch.tensor(row_indices)
        c_idx_cpu = slice(None) if col_indices is None else torch.tensor(col_indices)
        expected = x_cpu[d_idx_cpu, r_idx_cpu, c_idx_cpu]

        with track_neuron_ops():
            x = torch.tensor(input_data, device="neuron")
            d_idx = (
                slice(None)
                if depth_indices is None
                else torch.tensor(depth_indices, device="neuron")
            )
            r_idx = (
                slice(None) if row_indices is None else torch.tensor(row_indices, device="neuron")
            )
            c_idx = (
                slice(None) if col_indices is None else torch.tensor(col_indices, device="neuron")
            )
            result = x[d_idx, r_idx, c_idx]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::index.Tensor")

    @cpu_fallback_test
    def test_boolean_mask_indexing_cpu_fallback(self):
        """Test boolean mask indexing (expected to fail due to dynamic shapes)"""
        x_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask_cpu = torch.tensor([True, False, True, False, False])
        expected = x_cpu[mask_cpu]

        with track_neuron_ops():
            x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="neuron")
            mask = torch.tensor([True, False, True, False, False], device="neuron")
            result = x[mask]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_falls_back_on_cpu("aten::index.Tensor")

    def test_out_of_bounds_indexing_raises_error(self):
        """Test that out-of-bounds indexing raises IndexError like PyTorch."""
        input_data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        x = torch.tensor(input_data, device="neuron")
        row_idx = torch.tensor([1, 15], device="neuron")  # 15 is out of bounds
        with pytest.raises(
            IndexError, match=r"index .* is out of bounds for dimension 0 with size 4"
        ):
            x[row_idx, :]
