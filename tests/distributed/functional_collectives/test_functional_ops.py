import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

# Import the functions to test
from torch_neuron.distributed.ops.functional_collectives import (
    all_gather_into_tensor,
    all_gather_into_tensor_coalesced,
    all_gather_into_tensor_out,
    all_reduce,
    all_reduce_,
    reduce_scatter_tensor,
)

from ..utils import get_core_start_index


class TestFunctionalOps:
    """Test cases for functional collective operations."""

    def setup_method(self):
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(get_core_start_index())

    def teardown_method(self):
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_basic(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test basic all_gather_into_tensor functionality."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]  # 4 processes

        # Mock the all_gather_op to simulate gathering
        def mock_gather(input_tensor, replica_groups, out):
            # Simulate concatenating 4 copies of input along dim 0
            input_data = input_tensor.cpu()
            gathered_data = torch.cat([input_data] * 4, dim=0)
            out.copy_(gathered_data)

        mock_all_gather_op.side_effect = mock_gather

        # Test input
        input_tensor = torch.randn(2, 3, device="neuron")

        # Call function
        result = all_gather_into_tensor(input_tensor, 4, "test_group")

        # Assertions
        assert result.shape == (8, 3)  # 2 * 4 = 8
        assert result.device.type == "neuron"
        assert result.dtype == input_tensor.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_gather_op.assert_called_once()

    @patch("torch_neuron.distributed.ops.functional_collectives.all_reduce_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_reduce_basic(self, mock_get_ranks, mock_resolve_group, mock_all_reduce_op):
        """Test basic all_reduce functionality."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]

        def mock_reduce(input_list, replica_groups, reduce_type, out):
            # input_list is [tensor], out is (tensor,)
            out[0].copy_(input_list[0] * 4)

        mock_all_reduce_op.side_effect = mock_reduce

        input_tensor = torch.tensor([1.0, 2.0, 3.0], device="neuron")

        input_copy = input_tensor.clone()

        result = all_reduce(input_tensor, "sum", "test_group")

        # Assertions
        assert not torch.equal(result, input_copy)  # Result should be different
        assert torch.equal(input_tensor, input_copy)  # Original should be unchanged
        assert torch.equal(result, input_copy * 4)  # Result should be multiplied by 4
        assert result.shape == input_tensor.shape  # Shape should be preserved
        assert result.device.type == "neuron"
        assert result.dtype == input_tensor.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_reduce_op.assert_called_once()

        # More detailed verification
        call_args = mock_all_reduce_op.call_args
        assert call_args[0][0][0] is not input_tensor  # Should be a clone, not the original
        assert call_args[0][1] == [mock_get_ranks.return_value]  # replica_groups
        assert call_args[0][2] == "SUM"  # reduce_type should be uppercase
        assert call_args[1]["out"][0] is not input_tensor  # out param should be the clone

    @patch("torch_neuron.distributed.ops.functional_collectives.all_reduce_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_reduce_inplace_basic(self, mock_get_ranks, mock_resolve_group, mock_all_reduce_op):
        """Test basic all_reduce_ in-place functionality."""
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]

        def mock_reduce(input_tensors, replica_groups, reduce_type, out):
            # input_tensors is [tensor], out is (tensor,)
            out[0].copy_(input_tensors[0] * 4)

        mock_all_reduce_op.side_effect = mock_reduce

        input_tensor = torch.tensor([1.0, 2.0, 3.0], device="neuron")

        expected_result = input_tensor.clone() * 4

        result = all_reduce_(input_tensor, "sum", "test_group")

        assert result is input_tensor
        assert torch.equal(input_tensor, expected_result)
        assert result.shape == expected_result.shape
        assert result.device.type == "neuron"
        assert result.dtype == expected_result.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_reduce_op.assert_called_once()

        call_args = mock_all_reduce_op.call_args
        assert call_args[0][0][0] is input_tensor  # First element of list
        assert call_args[0][1] == [mock_get_ranks.return_value]
        assert call_args[0][2] == "SUM"
        assert call_args[1]["out"][0] is input_tensor  # First element of tuple

    @pytest.mark.parametrize(
        "reduce_op,expected_reduce_type,transform_fn",
        [
            ("sum", "SUM", lambda x: x * 4),
            ("max", "MAX", lambda x: x),
            ("min", "MIN", lambda x: x),
            ("product", "PRODUCT", lambda x: x**4),
            ("avg", "AVG", lambda x: x),
            ("band", "BAND", lambda x: x),
            ("bor", "BOR", lambda x: x),
            ("bxor", "BXOR", lambda x: x),
        ],
    )
    def test_all_reduce_different_reduce_ops(self, reduce_op, expected_reduce_type, transform_fn):
        """Test all_reduce with different reduction operations."""
        with (
            patch(
                "torch_neuron.distributed.ops.functional_collectives.all_reduce_op"
            ) as mock_all_reduce_op,
            patch(
                "torch_neuron.distributed.ops.functional_collectives._resolve_process_group"
            ) as mock_resolve_group,
            patch("torch.distributed.get_process_group_ranks") as mock_get_ranks,
        ):
            mock_group = MagicMock()
            mock_resolve_group.return_value = mock_group
            mock_get_ranks.return_value = [0, 1, 2, 3]

            def mock_reduce(input_list, replica_groups, reduce_type, out):
                # input_list is [tensor], out is (tensor,)
                assert reduce_type == expected_reduce_type
                out[0].copy_(transform_fn(input_list[0]))

            mock_all_reduce_op.side_effect = mock_reduce

            input_tensor = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            expected_output = transform_fn(input_tensor)

            result = all_reduce(input_tensor, reduce_op, "test_group")

            assert torch.equal(result, expected_output)
            assert result is not input_tensor

            mock_resolve_group.assert_called_once_with("test_group")
            mock_get_ranks.assert_called_once_with(mock_group)
            mock_all_reduce_op.assert_called_once()

            call_args = mock_all_reduce_op.call_args
            assert call_args[0][0][0] is not input_tensor  # First element of list
            assert call_args[0][1] == [mock_get_ranks.return_value]
            assert call_args[0][2] == expected_reduce_type
            assert call_args[1]["out"][0] is not input_tensor  # First element of tuple

    def test_all_reduce_empty_input(self):
        """Test that empty input raises assertion error."""
        empty_tensor = torch.empty(0, device="neuron")

        with pytest.raises(AssertionError, match="Cannot run all-reduce on empty input tensor"):
            all_reduce(empty_tensor, "sum", "test_group")

    def test_all_reduce_inplace_empty_input(self):
        """Test that empty input raises assertion error for in-place operation."""
        empty_tensor = torch.empty(0, device="neuron")

        with pytest.raises(AssertionError, match="Cannot run all-reduce_ on empty input tensor"):
            all_reduce_(empty_tensor, "sum", "test_group")

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_stride_preservation(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test that all_gather_into_tensor preserves stride."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1]  # 2 processes

        # Mock all_gather_op to do nothing (we just want to test tensor creation)
        mock_all_gather_op.return_value = None

        # Create non-contiguous input tensor
        input_tensor = torch.randn(4, 6, 8, device="neuron")
        input_tensor = input_tensor.permute(1, 2, 0)  # Shape: (6, 8, 4), stride: (8, 1, 48)

        # Call function
        result = all_gather_into_tensor(input_tensor, 2, "test_group")

        # Check that stride is preserved
        expected_shape = (12, 8, 4)  # 6 * 2 = 12
        expected_stride = input_tensor.stride()  # Should be same as input

        assert result.shape == expected_shape
        assert result.stride() == expected_stride
        assert result.device.type == "neuron"

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_coalesced_basic(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test basic all_gather_into_tensor_coalesced functionality with multiple tensors."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]  # 4 processes

        # Mock the all_gather_op to simulate gathering
        def mock_gather(input_tensors, replica_groups, out):
            # Simulate concatenating 4 copies of each input along dim 0
            for i, input_tensor in enumerate(input_tensors):
                input_data = input_tensor.cpu()
                gathered_data = torch.cat([input_data] * 4, dim=0)
                out[i].copy_(gathered_data)

        mock_all_gather_op.side_effect = mock_gather

        # Test inputs
        input_tensor1 = torch.randn(2, 3, device="neuron")
        input_tensor2 = torch.randn(3, 5, device="neuron")
        inputs = [input_tensor1, input_tensor2]

        # Call function
        results = all_gather_into_tensor_coalesced(inputs, 4, "test_group")

        # Assertions
        assert len(results) == 2
        assert results[0].shape == (8, 3)  # 2 * 4 = 8
        assert results[1].shape == (12, 5)  # 3 * 4 = 12
        assert all(res.device.type == "neuron" for res in results)
        assert results[0].dtype == input_tensor1.dtype
        assert results[1].dtype == input_tensor2.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_gather_op.assert_called_once()

        # More detailed verification
        call_args = mock_all_gather_op.call_args
        assert call_args[0][0] is inputs  # Should be the original input list
        assert call_args[0][1] == [mock_get_ranks.return_value]  # replica_groups
        assert len(call_args[1]["out"]) == 2  # Should have two output tensors
        assert call_args[1]["out"][0].shape == (8, 3)  # First output shape
        assert call_args[1]["out"][1].shape == (12, 5)  # Second output shape

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_coalesced_stride_preservation(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test that all_gather_into_tensor_coalesced preserves stride."""
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1]  # 2 processes

        # Mock all_gather_op to do nothing (we just want to test tensor creation)
        mock_all_gather_op.return_value = None

        # Create non-contiguous input tensors
        input_tensor1 = torch.randn(4, 6, 8, device="neuron")
        input_tensor1 = input_tensor1.permute(1, 2, 0)  # Shape: (6, 8, 4), stride: (8, 1, 48)

        input_tensor2 = torch.randn(2, 5, 3, device="neuron")
        input_tensor2 = input_tensor2.permute(2, 0, 1)  # Shape: (3, 2, 5), stride: (10, 5, 1)

        inputs = [input_tensor1, input_tensor2]

        # Call function
        results = all_gather_into_tensor_coalesced(inputs, 2, "test_group")

        # Check that stride is preserved
        assert results[0].shape == (12, 8, 4)  # 6 * 2 = 12
        assert results[0].stride() == input_tensor1.stride()  # Should be same as input

        assert results[1].shape == (6, 2, 5)  # 3 * 2 = 6
        assert results[1].stride() == input_tensor2.stride()  # Should be same as input

        assert all(t.device.type == "neuron" for t in results)

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_gather_op.assert_called_once()

        # More detailed verification
        call_args = mock_all_gather_op.call_args
        assert call_args[0][0] is inputs  # Should be the original input list
        assert call_args[0][1] == [mock_get_ranks.return_value]
        assert len(call_args[1]["out"]) == 2

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_out_basic(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test basic all_gather_into_tensor_out functionality."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]  # 4 processes

        # Mock the all_gather_op to simulate gathering
        def mock_gather(input_tensor, replica_groups, out):
            # Simulate concatenating 4 copies of input along dim 0
            input_data = input_tensor.cpu()
            gathered_data = torch.cat([input_data] * 4, dim=0)
            out.copy_(gathered_data)

        mock_all_gather_op.side_effect = mock_gather

        # Test input
        input_tensor = torch.randn(2, 3, device="neuron")
        output_tensor = torch.empty(8, 3, device="neuron")  # Pre-allocated output tensor

        # Call function
        result = all_gather_into_tensor_out(input_tensor, 4, "test_group", output_tensor)

        # Assertions
        assert result is output_tensor  # Should return the output tensor
        assert result.shape == (8, 3)  # 2 * 4 = 8
        assert result.device.type == "neuron"
        assert result.dtype == input_tensor.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_gather_op.assert_called_once()

        call_args = mock_all_gather_op.call_args
        assert call_args[0][0] is input_tensor  # Should be the original input tensor
        assert call_args[0][1] == [mock_get_ranks.return_value]  # replica_groups
        assert call_args[1]["out"] is output_tensor  # out param should be our output tensor

    @patch("torch_neuron.distributed.ops.functional_collectives.all_gather_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_out_stride_preservation(
        self, mock_get_ranks, mock_resolve_group, mock_all_gather_op
    ):
        """Test that all_gather_into_tensor_out preserves stride."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1]  # 2 processes

        # Mock all_gather_op to do nothing (we just want to test tensor handling)
        mock_all_gather_op.return_value = None

        # Create non-contiguous input tensor
        input_tensor = torch.randn(4, 6, 8, device="neuron")
        input_tensor = input_tensor.permute(1, 2, 0)  # Shape: (6, 8, 4), stride: (8, 1, 48)

        # Create output tensor with same stride pattern
        output_shape = list(input_tensor.shape)
        output_shape[0] *= 2  # Double first dimension for 2 processes
        output_tensor = torch.empty_strided(
            output_shape, input_tensor.stride(), device="neuron", dtype=input_tensor.dtype
        )

        # Call function
        result = all_gather_into_tensor_out(input_tensor, 2, "test_group", output_tensor)

        # Check that result is the same object as output_tensor
        assert result is output_tensor
        assert result.shape == (12, 8, 4)  # 6 * 2 = 12
        assert result.stride() == input_tensor.stride()  # Should maintain same stride

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_all_gather_op.assert_called_once()

        # More detailed verification
        call_args = mock_all_gather_op.call_args
        assert call_args[0][0] is input_tensor  # Should be our non-contiguous input tensor
        assert call_args[0][1] == [mock_get_ranks.return_value]  # replica_groups
        assert (
            call_args[1]["out"] is output_tensor
        )  # out param should be our stride-preserved tensor

    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_all_gather_into_tensor_out_wrong_size(self, mock_get_ranks, mock_resolve_group):
        """Test that wrong output size raises assertion error."""
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]  # 4 processes

        input_tensor = torch.randn(2, 3, device="neuron")

        # Output with wrong size
        wrong_output = torch.empty(7, 3, device="neuron")  # Should be (8, 3)

        with pytest.raises(AssertionError, match="Output tensor shape mismatch"):
            all_gather_into_tensor_out(input_tensor, 4, "test_group", wrong_output)

        # Verify we didn't proceed to calling the operation
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        # No call to all_gather_op should have occurred

    @patch("torch_neuron.distributed.ops.functional_collectives.reduce_scatter_op")
    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    @patch("torch.distributed.get_process_group_ranks")
    def test_reduce_scatter_tensor_basic(
        self, mock_get_ranks, mock_resolve_group, mock_reduce_scatter_op
    ):
        """Test basic reduce_scatter_tensor functionality."""
        # Setup mocks
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group
        mock_get_ranks.return_value = [0, 1, 2, 3]  # 4 processes

        # Mock the reduce_scatter_op
        def mock_reduce_scatter(input_list, replica_groups, reduce_type, out):
            # Simulate reducing and scattering - just copy a slice
            input_tensor = input_list[0]
            slice_size = input_tensor.size(0) // 4
            out.copy_(input_tensor[:slice_size])

        mock_reduce_scatter_op.side_effect = mock_reduce_scatter

        # Test input (divisible by group size)
        input_tensor = torch.randn(8, 3, device="neuron")  # 8 / 4 = 2

        # Call function
        result = reduce_scatter_tensor(input_tensor, "SUM", 4, "test_group")

        # Assertions
        assert result.shape == (2, 3)  # 8 / 4 = 2
        assert result.device.type == "neuron"
        assert result.dtype == input_tensor.dtype

        # Verify mocks were called correctly
        mock_resolve_group.assert_called_once_with("test_group")
        mock_get_ranks.assert_called_once_with(mock_group)
        mock_reduce_scatter_op.assert_called_once()

    def test_all_gather_into_tensor_empty_input(self):
        """Test that empty input raises assertion error."""
        empty_tensor = torch.empty(0, device="neuron")

        with pytest.raises(AssertionError, match="Cannot run all-gather on empty input tensor"):
            all_gather_into_tensor(empty_tensor, 4, "test_group")

    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    def test_all_gather_into_tensor_coalesced_contains_empty_tensor(self, mock_resolve_group):
        """Test that list containing an empty tensor raises assertion error."""
        input_tensor1 = torch.randn(2, 3, device="neuron")
        empty_tensor = torch.empty(0, device="neuron")

        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group

        with patch("torch.distributed.get_process_group_ranks") as mock_get_ranks:
            mock_get_ranks.return_value = [0, 1, 2, 3]

            with pytest.raises(
                AssertionError, match="Cannot run all-gather on empty input tensors"
            ):
                all_gather_into_tensor_coalesced([input_tensor1, empty_tensor], 4, "test_group")

    def test_all_gather_into_tensor_coalesced_empty_list(self):
        """Test that empty list of tensors raises assertion error."""
        # Empty list of tensors
        with pytest.raises(AssertionError, match="Cannot run all-gather on empty input list"):
            all_gather_into_tensor_coalesced([], 4, "test_group")

    @patch("torch_neuron.distributed.ops.functional_collectives._resolve_process_group")
    def test_all_gather_into_tensor_coalesced_group_size_mismatch(self, mock_resolve_group):
        """Test that group size mismatch raises assertion error."""
        mock_group = MagicMock()
        mock_resolve_group.return_value = mock_group

        with patch("torch.distributed.get_process_group_ranks") as mock_get_ranks:
            mock_get_ranks.return_value = [0, 1, 2]  # 3 processes

            input_tensor = torch.randn(2, 3, device="neuron")

            # Group size different from actual group size
            with pytest.raises(
                AssertionError,
                match="group_size is not equal to the length of replica groups, Expected 3, got 4",
            ):
                all_gather_into_tensor_coalesced([input_tensor], 4, "test_group")  # 4 != 3

    def test_all_gather_into_tensor_out_empty_input(self):
        """Test that empty input raises assertion error."""
        empty_tensor = torch.empty(0, device="neuron")
        output_tensor = torch.empty(0, device="neuron")

        with pytest.raises(AssertionError, match="Cannot run all-gather on empty input tensor"):
            all_gather_into_tensor_out(empty_tensor, 4, "test_group", output_tensor)

    def test_reduce_scatter_tensor_empty_input(self):
        """Test that empty input raises assertion error."""
        empty_tensor = torch.empty(0, device="neuron")

        with pytest.raises(AssertionError, match="Cannot run reduce-scatter on empty input tensor"):
            reduce_scatter_tensor(empty_tensor, "SUM", 4, "test_group")

    def test_reduce_scatter_tensor_indivisible_size(self):
        """Test that indivisible tensor size raises assertion error."""
        input_tensor = torch.randn(5, 3, device="neuron")  # 5 not divisible by 4

        with pytest.raises(
            AssertionError, match="Input tensor size 5 is not divisible by group size 4"
        ):
            reduce_scatter_tensor(input_tensor, "SUM", 4, "test_group")
