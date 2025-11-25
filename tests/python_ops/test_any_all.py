"""Test that any and all operations are properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# Test data generation - List 1: All inputs with basic test
basic_test_cases = []
for op_name, op_func in [("any", torch.any), ("all", torch.all)]:
    basic_test_cases.extend(
        [
            pytest.param(
                op_name, op_func, torch.tensor([False, True, False]), id=f"{op_name}_bool_mixed"
            ),
            pytest.param(
                op_name,
                op_func,
                torch.tensor([False, False, False]),
                id=f"{op_name}_bool_all_false",
            ),
            pytest.param(
                op_name, op_func, torch.tensor([True, True, True]), id=f"{op_name}_bool_all_true"
            ),
            pytest.param(
                op_name, op_func, torch.tensor([0, 1, 2], dtype=torch.int32), id=f"{op_name}_int32"
            ),
            pytest.param(
                op_name,
                op_func,
                torch.tensor([1.0, 0.0, 2.5], dtype=torch.float32),
                id=f"{op_name}_float32",
            ),
            pytest.param(
                op_name, op_func, torch.tensor([0, 1, 2], dtype=torch.uint8), id=f"{op_name}_uint8"
            ),
            pytest.param(op_name, op_func, torch.tensor([True]), id=f"{op_name}_single_element"),
            pytest.param(op_name, op_func, torch.empty(0, dtype=torch.bool), id=f"{op_name}_empty"),
        ]
    )

# Test data generation - List 2: One input with all variants
variant_test_cases = [
    pytest.param(
        "any",
        torch.any,
        torch.tensor([[True, False, False], [False, False, True]], dtype=torch.bool),
        id="any_variant",
    ),
    pytest.param(
        "all",
        torch.all,
        torch.tensor([[True, False, False], [False, False, True]], dtype=torch.bool),
        id="all_variant",
    ),
]


class TestAnyAllRegistration:
    """Test any and all operation registration and functionality."""

    @pytest.mark.parametrize(
        "op_name,op_func,input_tensor",
        basic_test_cases,
    )
    def test_any_all_reduction_basic(self, op_name, op_func, input_tensor):
        """Test torch.any/all function and method with various dtypes (no dimension specified)."""
        # Create neuron tensor
        a_neuron = input_tensor.detach().clone().to("neuron")
        a_cpu = input_tensor.detach().clone()

        # Test function with neuron op tracking
        with track_neuron_ops():
            result = op_func(a_neuron)
            assert_op_runs_on_neuron(f"aten::{op_name}")

        expected = op_func(a_cpu)

        # Verify result
        assert result.shape == ()  # Scalar tensor

        # Special handling for uint8 dtype
        if input_tensor.dtype == torch.uint8:
            assert result.dtype == torch.uint8

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op_name,op_func,input_tensor", variant_test_cases)
    def test_any_all_reduction_with_dim(self, op_name, op_func, input_tensor):
        """Test function with dim parameter."""
        a_neuron = input_tensor.detach().clone().to("neuron")
        with track_neuron_ops():
            result = op_func(a_neuron, dim=1)
            assert_op_runs_on_neuron(f"aten::{op_name}")
        expected = op_func(input_tensor, dim=1)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op_name,op_func,input_tensor", variant_test_cases)
    def test_any_all_reduction_method_with_dim(self, op_name, op_func, input_tensor):
        """Test method with dim parameter."""
        a_neuron = input_tensor.detach().clone().to("neuron")
        with track_neuron_ops():
            result = getattr(a_neuron, op_name)(dim=0)
            assert_op_runs_on_neuron(f"aten::{op_name}")
        expected = getattr(input_tensor, op_name)(dim=0)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op_name,op_func,input_tensor", variant_test_cases)
    def test_any_all_reduction_dim_out(self, op_name, op_func, input_tensor):
        """Test dim out parameter."""
        a_neuron = input_tensor.detach().clone().to("neuron")
        out = torch.empty(2, dtype=torch.bool).to("neuron")
        with track_neuron_ops():
            result = op_func(a_neuron, dim=1, out=out)
            assert_op_runs_on_neuron(f"aten::{op_name}")
        expected = op_func(input_tensor, dim=1)
        assert result is out
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op_name,op_func,input_tensor", variant_test_cases)
    def test_any_all_reduction_multiple_dims(self, op_name, op_func, input_tensor):
        """Test multiple dimensions."""
        c = torch.tensor(
            [[[True, False], [False, True]], [[False, False], [True, True]]], dtype=torch.bool
        )
        c_neuron = c.to("neuron")
        with track_neuron_ops():
            result = op_func(c_neuron, dim=(0, 2))
            assert_op_runs_on_neuron(f"aten::{op_name}")
        expected = op_func(c, dim=(0, 2))
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op_name,op_func,input_tensor", variant_test_cases)
    def test_any_all_reduction_3d_keepdim(self, op_name, op_func, input_tensor):
        """Test 3D tensor with keepdim."""
        d = torch.randint(0, 2, (2, 3, 4)).bool()
        d_neuron = d.to("neuron")
        with track_neuron_ops():
            result = op_func(d_neuron, dim=1, keepdim=True)
            assert_op_runs_on_neuron(f"aten::{op_name}")
        expected = op_func(d, dim=1, keepdim=True)
        assert result.shape == (2, 1, 4)
        torch.testing.assert_close(result.cpu(), expected)
