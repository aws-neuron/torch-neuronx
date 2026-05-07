"""Test that logical operations are properly registered with PyTorch dispatcher."""

import os
import sys

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)

# Test data generation
test_cases = []

# Basic operations
for op_name, op_func in [
    ("logical_and", torch.logical_and),
    ("logical_or", torch.logical_or),
    ("logical_xor", torch.logical_xor),
]:
    # Basic bool tensors
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([True, False, True, False]),
            torch.tensor([True, True, False, False]),
            id=f"{op_name}_bool_basic",
        )
    )

    # Different dtypes
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([1, 0, 2, -1], dtype=torch.int32),
            torch.tensor([1, 1, 0, 0], dtype=torch.int32),
            id=f"{op_name}_int32",
        )
    )

    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([1.0, 0.0, 2.5, -1.5], dtype=torch.float32),
            torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            id=f"{op_name}_float32",
        )
    )

    # Broadcasting shapes
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([[True, False], [True, False]]),
            torch.tensor([True, False]),
            id=f"{op_name}_broadcast_2d_1d",
        )
    )
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([True, False, True]),
            torch.tensor([[True], [False], [True]]),
            id=f"{op_name}_broadcast_1d_2d",
        )
    )

    # Scalar tensors (0-dim)
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([True, False, True, False]),
            torch.tensor(True),
            id=f"{op_name}_scalar",
        )
    )

    # # Different sizes
    test_cases.append(
        pytest.param(
            op_name,
            op_func,
            torch.tensor([True]),
            torch.tensor([False]),
            id=f"{op_name}_single_element",
        )
    )


@pytest.mark.parametrize(
    "op_name,op_func,input1,input2",
    test_cases,
)
class TestLogicalOpsRegistration:
    """Test logical operations registration and functionality."""

    def test_logical_op_runs_on_neuron(self, op_name, op_func, input1, input2):
        """Test that logical op runs on Neuron without CPU fallback"""
        a, a_device = input1, input1.to("neuron")
        b, b_device = input2, input2.to("neuron")
        with track_neuron_ops():
            output = op_func(a_device, b_device)
            assert_op_runs_on_neuron(f"aten::{op_name}")

        expected = op_func(a, b)
        torch.testing.assert_close(expected, output.cpu())

    def test_logical_op_inplace_runs_on_neuron(self, op_name, op_func, input1, input2):
        """Test that inplace logical op runs on Neuron without CPU fallback"""
        a, a_device = input1, input1.to("neuron")
        b, b_device = input2, input2.to("neuron")
        inplace_op_name = op_name + "_"

        # Test CPU first to see if it should fail
        inplace_op = getattr(a, inplace_op_name)
        try:
            inplace_op(b)
            cpu_success = True
        except RuntimeError:
            cpu_success = False

        # Test Neuron - should have same behavior as CPU
        inplace_op_device = getattr(a_device, inplace_op_name)
        if cpu_success:
            with track_neuron_ops():
                inplace_op_device(b_device)
                assert_op_runs_on_neuron(op_name)
            torch.testing.assert_close(a, a_device.cpu())
        else:
            self._test_inplace_runtime_error(inplace_op_device, b_device)

    @assert_raises(RuntimeError)
    def _test_inplace_runtime_error(self, inplace_op_device, b_device):
        """Helper method to test inplace operation runtime error"""
        inplace_op_device(b_device)
        pytest.xfail(
            reason=(
                "cpu raises runtime error but neuron prints warning, cpu runtime error hits"
                " https://github.com/pytorch/pytorch/blob/5d819f3fafe68ad6fdc133c58b5f5591a34c2d9f/aten/src/ATen/TensorIterator.cpp#L1213"
            )
        )

    def test_logical_op_with_output(self, op_name, op_func, input1, input2):
        """Test logical op with pre-allocated output tensor"""
        a, a_device = input1, input1.to("neuron")
        b, b_device = input2, input2.to("neuron")

        # Get broadcast shape for output tensor
        broadcast_shape = torch.broadcast_shapes(a.shape, b.shape)
        expected = torch.empty(broadcast_shape, dtype=torch.bool)

        with track_neuron_ops():
            output = torch.empty(broadcast_shape, dtype=torch.bool, device="neuron")
            op_func(a_device, b_device, out=output)
            assert_op_runs_on_neuron(op_name)

        op_func(a, b, out=expected)

        torch.testing.assert_close(expected, output.cpu())
