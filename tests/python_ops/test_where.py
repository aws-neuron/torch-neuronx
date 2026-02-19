"""Test that where operation is properly registered with PyTorch dispatcher.

Note: where.ScalarOther, where.ScalarSelf, and where.Scalar are NOT registered
because they are CompositeImplicitAutograd ops that decompose to where.self.
This preserves the autograd chain.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

DTYPES = [torch.bool, torch.float32, torch.float64, torch.int32, torch.int64]
SHAPES = [(1,), (1, 8), (8, 15, 29), (15, 32, 64, 128)]


def _assert_equal(x, y):
    return torch.testing.assert_close(x, y, atol=0, rtol=0)


class TestWhereRegistration:
    """Test where operation registration and functionality."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("out", [False, True])
    def test_where_same_shape(self, dtype, shape, out):
        """Test for all args with matching shapes. Only
        this case can have optional out arg to store result."""
        condition = torch.randn(shape) > 0.5
        input = torch.randn(shape).to(dtype)
        other = torch.randn(shape).to(dtype)

        if out:
            expected = torch.empty_like(input)
            torch.where(condition, input, other, out=expected)
        else:
            expected = torch.where(condition, input, other)

        with track_neuron_ops():
            condition_neuron = condition.to("neuron")
            input_neuron = input.to("neuron")
            other_neuron = other.to("neuron")

            if out:
                out_neuron = torch.empty_like(input_neuron)
                torch.where(condition_neuron, input_neuron, other_neuron, out=out_neuron)
            else:
                out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)

            # check if op ran on neuron device
            assert_op_runs_on_neuron("aten::where.self")

        # check correctness
        _assert_equal(out_neuron.cpu(), expected)

    def test_where_autograd_self(self):
        """Test that where.self preserves autograd."""
        condition = torch.randn(3, 4) > 0.5
        x = torch.randn(3, 4, device="neuron", requires_grad=True)
        y = torch.randn(3, 4, device="neuron", requires_grad=True)
        result = torch.where(condition.to("neuron"), x, y)

        # Should have grad_fn (WhereBackward0)
        assert result.grad_fn is not None
        assert "Backward" in str(type(result.grad_fn))

    def test_where_autograd_scalar_other(self):
        """Test that where.ScalarOther preserves autograd via decomposition."""
        condition = torch.randn(3, 4) > 0.5
        x = torch.randn(3, 4, device="neuron", requires_grad=True)
        result = torch.where(condition.to("neuron"), x, 0.5)

        # Should have grad_fn from decomposed where.self
        assert result.grad_fn is not None
        assert "Backward" in str(type(result.grad_fn))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_where_scalar_input(self, dtype, shape):
        """Test for input (second arg) as scalar - decomposes to where.self"""
        condition = torch.randn(shape) > 0.5
        input = torch.randn(1).item()
        other = torch.randn(shape).to(dtype)
        expected = torch.where(condition, input, other)

        with track_neuron_ops():
            condition_neuron = condition.to("neuron")
            input_neuron = input  # scalar
            other_neuron = other.to("neuron")
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)

            # ScalarSelf decomposes to where.self
            assert_op_runs_on_neuron("aten::where.self")

        # check correctness
        _assert_equal(out_neuron.cpu(), expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_where_scalar_other(self, dtype, shape):
        """Test for other (third arg) as scalar - decomposes to where.self"""
        condition = torch.randn(shape) > 0.5
        input = torch.randn(shape).to(dtype)
        other = torch.rand(1).item()
        expected = torch.where(condition, input, other)

        with track_neuron_ops():
            condition_neuron = condition.to("neuron")
            input_neuron = input.to("neuron")
            other_neuron = other  # scalar
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)

            # ScalarOther decomposes to where.self
            assert_op_runs_on_neuron("aten::where.self")

        # check correctness
        _assert_equal(out_neuron.cpu(), expected)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_where_scalar(self, shape):
        """Test for input (second arg) and other (third arg) as scalar - decomposes to where.self"""
        condition = torch.randn(shape) > 0.5
        input = torch.rand(1).item()
        other = torch.rand(1).item()
        expected = torch.where(condition, input, other)

        with track_neuron_ops():
            condition_neuron = condition.to("neuron")
            input_neuron = input  # scalar
            other_neuron = other  # scalar
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)

            # Scalar decomposes to where.self
            assert_op_runs_on_neuron("aten::where.self")

        # check correctness
        _assert_equal(out_neuron.cpu(), expected)

    @pytest.mark.parametrize("dtype,shape", [(torch.float32, (3, 4))])
    def test_where_only_condition_runs_on_cpu(self, dtype, shape):
        """Test when only condition (first arg) is provided."""
        condition = (torch.randn(shape) > 0.5).to(dtype)
        expected = torch.where(condition)

        with track_neuron_ops():
            condition_neuron = condition.to("neuron")
            out_neuron = torch.where(condition_neuron)

            # When only condition is passed in, nonzero gets dispatched
            # https://github.com/pytorch/pytorch/blob/df71b7072799c451a008cb36142dfdb1487f0d5e/aten/src/ATen/native/TensorCompare.cpp#L667
            assert_op_runs_on_neuron("nonzero")

        assert isinstance(expected, tuple)
        assert isinstance(out_neuron, tuple)
        assert len(expected) == len(out_neuron)

        for out_neuron_i, expected_i in zip(out_neuron, expected, strict=False):
            assert out_neuron_i.device.type == "neuron"
            _assert_equal(out_neuron_i.cpu(), expected_i)

    # Broadcast can happen with "missing_dims" or "full_dims".
    # In "missing_dims" case, one of the tensors may have less
    # number of dims. "full_dims" case will have same number
    # of dims for all tensors with some singleton dims.
    @pytest.mark.parametrize("broadcast_tensor", ["condition", "input", "other"])
    @pytest.mark.parametrize("broadcast_type", ["missing_dims", "full_dims"])
    def test_where_same_shape_broadcast(self, broadcast_tensor, broadcast_type):
        dtype = torch.float32
        shape = (2, 5, 8, 17)

        condition = torch.randn(shape) > 0.5
        input = torch.randn(shape).to(dtype)
        other = torch.randn(shape).to(dtype)

        for broadcast_idx in range(len(shape)):
            if broadcast_type == "missing_dims" and broadcast_idx == len(shape) - 1:
                continue

            new_shape = list(shape)
            if broadcast_type == "missing_dims":
                new_shape = new_shape[broadcast_idx:]
            else:
                new_shape[broadcast_idx] = 1

            if broadcast_tensor == "condition":
                condition = torch.randn(new_shape) > 0.5
            else:
                new_tensor = torch.randn(new_shape).to(dtype)
                input = new_tensor if broadcast_tensor == "input" else input
                other = new_tensor if broadcast_tensor == "other" else other

            condition_neuron = condition.to("neuron")
            input_neuron = input.to("neuron")
            other_neuron = other.to("neuron")

            expected = torch.where(condition, input, other)
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)
            _assert_equal(out_neuron.cpu(), expected)

    @pytest.mark.parametrize("broadcast_tensor", ["condition", "other"])
    @pytest.mark.parametrize("broadcast_type", ["missing_dims", "full_dims"])
    def test_where_scalar_input_broadcast(self, broadcast_tensor, broadcast_type):
        dtype = torch.float32
        shape = (2, 5, 8, 17)

        condition = torch.randn(shape) > 0.5
        input = torch.rand(1).item()
        other = torch.randn(shape).to(dtype)

        for broadcast_idx in range(len(shape)):
            if broadcast_type == "missing_dims" and broadcast_idx == len(shape) - 1:
                continue

            new_shape = list(shape)
            if broadcast_type == "missing_dims":
                new_shape = new_shape[broadcast_idx:]
            else:
                new_shape[broadcast_idx] = 1

            if broadcast_tensor == "condition":
                condition = torch.randn(new_shape) > 0.5
            else:
                other = torch.randn(new_shape).to(dtype)

            condition_neuron = condition.to("neuron")
            input_neuron = input  # scalar
            other_neuron = other.to("neuron")

            expected = torch.where(condition, input, other)
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)
            _assert_equal(out_neuron.cpu(), expected)

    @pytest.mark.parametrize("broadcast_tensor", ["condition", "input"])
    @pytest.mark.parametrize("broadcast_type", ["missing_dims", "full_dims"])
    def test_where_scalar_other_broadcast(self, broadcast_tensor, broadcast_type):
        dtype = torch.float32
        shape = (2, 5, 8, 17)

        condition = torch.randn(shape) > 0.5
        input = torch.randn(shape).to(dtype)
        other = torch.rand(1).item()

        for broadcast_idx in range(len(shape)):
            if broadcast_type == "missing_dims" and broadcast_idx == len(shape) - 1:
                continue

            new_shape = list(shape)
            if broadcast_type == "missing_dims":
                new_shape = new_shape[broadcast_idx:]
            else:
                new_shape[broadcast_idx] = 1

            if broadcast_tensor == "condition":
                condition = torch.randn(new_shape) > 0.5
            else:
                input = torch.randn(new_shape).to(dtype)

            condition_neuron = condition.to("neuron")
            input_neuron = input.to("neuron")
            other_neuron = other  # scalar

            expected = torch.where(condition, input, other)
            out_neuron = torch.where(condition_neuron, input_neuron, other_neuron)
            _assert_equal(out_neuron.cpu(), expected)

    def test_where_cpu_scalar_tensor_with_neuron(self):
        """Test that CPU scalar tensor works with neuron tensors"""
        scalar_cpu = torch.tensor(1, device="cpu")
        tensor_neuron = torch.randn(10, 5, device="neuron")
        condition_neuron = torch.randn(10, 5, device="neuron") > 0.5

        result = torch.where(condition_neuron, tensor_neuron, scalar_cpu)
        assert result.device.type == "neuron"

    def test_where_cpu_non_scalar_tensor_with_neuron(self):
        """Test that CPU non-scalar tensor raises error with neuron tensors"""
        tensor_cpu = torch.tensor([1], device="cpu")
        tensor_neuron = torch.randn(10, 5, device="neuron")
        condition_neuron = torch.randn(10, 5, device="neuron") > 0.5
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            torch.where(condition_neuron, tensor_neuron, tensor_cpu)
