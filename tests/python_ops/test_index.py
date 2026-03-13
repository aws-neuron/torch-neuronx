"""Test that index operation is properly registered with PyTorch dispatcher."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIndexRegistration:
    """Test index operation registration and functionality."""

    @pytest.mark.parametrize(
        "input_data,index_data,index_device",
        [
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1], "neuron"),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1], "neuron"),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [-1], "neuron"),
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1], "cpu"),
        ],
    )
    def test_basic_index_operations_run_on_neuron(self, input_data, index_data, index_device):
        """Test that basic index operations run on Neuron without CPU fallback"""
        x_cpu = torch.tensor(input_data)
        index_cpu = torch.tensor(index_data)
        expected = x_cpu[index_cpu]

        with track_neuron_ops():
            x = torch.tensor(input_data, device="neuron")
            index = torch.tensor(index_data, device=index_device)
            result = x[index]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("index")

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
            assert_op_runs_on_neuron("index")

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
            assert_op_runs_on_neuron("index")

    @pytest.mark.parametrize(
        "shape,index_device",
        [
            ((5,), "neuron"),
            ((4, 5), "neuron"),
            ((3, 4, 5), "neuron"),
            ((3, 4, 5), "cpu"),
        ],
    )
    def test_boolean_mask_indexing(self, shape, index_device):
        """Test boolean mask indexing"""
        x_cpu = torch.randn(shape)
        mask_cpu = torch.randint(0, 2, shape).bool()
        expected = x_cpu[mask_cpu]

        with track_neuron_ops():
            x = x_cpu.to(device="neuron")
            mask = mask_cpu.to(device="neuron")
            result = x[mask]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("index")

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (5,),
            (4, 5),
            (3, 4, 5),
        ],
    )
    def test_boolean_mask_all_false(self, shape):
        """Test boolean mask indexing where all mask values are False"""
        x_cpu = torch.randn(shape)
        mask_cpu = torch.zeros(shape, dtype=torch.bool)
        expected = x_cpu[mask_cpu]

        with track_neuron_ops():
            x = x_cpu.to(device="neuron")
            mask = mask_cpu.to(device="neuron")
            result = x[mask]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_did_not_run_on_neuron("index")
            assert_op_runs_on_neuron("nonzero")

    @assert_raises(RuntimeError, match=r"index .* is out of bounds for dimension 0 with size 4")
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
        x[row_idx, :]

    @pytest.mark.parametrize(
        "input_dtype",
        [
            pytest.param(torch.float64, marks=pytest.mark.xfail(reason="float64 not supported")),
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            pytest.param(torch.int64, marks=pytest.mark.xfail(reason="int64 not supported")),
        ],
    )
    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    def test_valid_dtype(self, input_dtype, index_dtype):
        """Test different valid types for input and index"""
        input_data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        x_cpu = torch.tensor(input_data, dtype=input_dtype)
        index_cpu = torch.tensor([1, 2], dtype=index_dtype)
        expected = x_cpu[index_cpu]

        with track_neuron_ops():
            x = x_cpu.to(device="neuron")
            index = index_cpu.to(device="neuron")
            result = x[index]
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("index")

    @pytest.mark.parametrize(
        "tensor_shape,indices_pattern",
        [
            ((1, 5, 512), "slice,empty,empty"),
            ((10, 5, 512), "empty,slice,empty"),
            ((2, 3, 4, 5), "slice,empty,slice,empty"),
            ((2, 3, 4, 5), "empty,slice,slice,empty"),
            ((10, 5, 512), "int,empty,slice"),
            ((10, 5, 512), "slice_num,empty,slice"),
        ],
    )
    def test_empty_index_patterns(self, tensor_shape, indices_pattern):
        """Test empty index patterns"""
        x_cpu = torch.randn(tensor_shape, device="cpu")
        empty_index_cpu = torch.empty(0, dtype=torch.long, device="cpu")

        indices_cpu = []
        for part in indices_pattern.split(","):
            if part == "empty":
                indices_cpu.append(empty_index_cpu)
            elif part == "slice":
                indices_cpu.append(slice(None))
            elif part == "slice_num":
                indices_cpu.append(slice(5))
            elif part == "int":
                indices_cpu.append(0)

        result_cpu = x_cpu[tuple(indices_cpu)]

        x_neuron = x_cpu.to(device="neuron")
        indices_neuron = [
            empty_index_cpu.to(device="neuron") if isinstance(idx, torch.Tensor) else idx
            for idx in indices_cpu
        ]

        with track_neuron_ops():
            result = x_neuron[tuple(indices_neuron)]
            assert result.shape == result_cpu.shape
            torch.testing.assert_close(result.cpu(), result_cpu)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("index")
