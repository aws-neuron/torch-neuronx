"""Test that clamp and clamp_min operations are properly registered with PyTorch dispatcher."""

import os
import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestClamp:
    def test_clamp_basic(self):
        """Test basic clamp functionality with min/max/none combinations."""
        with track_neuron_ops():
            # Create inputs on CPU first
            input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
            # Move to neuron
            input_arr = input_cpu.to("neuron")
            # Test with both min and max
            expected_both = torch.clamp(input_cpu, min=0.0, max=5.0)
            result_both = torch.clamp(input_arr, min=0.0, max=5.0)
            assert torch.all(result_both.cpu() == expected_both)
            assert_op_runs_on_neuron("aten::clamp")
            # Test with only max
            expected_max = torch.clamp(input_cpu, max=3.0)
            result_max = torch.clamp(input_arr, max=3.0)
            assert torch.all(result_max.cpu() == expected_max)
            assert_op_runs_on_neuron("aten::clamp")
            # Test with only min
            expected_min = torch.clamp(input_cpu, min=3.0)
            result_min = torch.clamp(input_arr, min=3.0)
            assert torch.all(result_min.cpu() == expected_min)
            assert_op_runs_on_neuron("aten::clamp")

    def test_clamp_none_args(self):
        """Test that clamp properly errors when both min and max are None."""
        # Create input tensor
        input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])

        # Get CPU error message for implicit None
        implicit_none_error = ""
        try:
            torch.clamp(input_cpu)
        except RuntimeError as e:
            implicit_none_error = str(e)

        # Get CPU error message for explicit None
        explicit_none_error = ""
        try:
            torch.clamp(input_cpu, min=None, max=None)
        except RuntimeError as e:
            explicit_none_error = str(e)

        # Now test neuron behavior
        with track_neuron_ops():
            input_arr = input_cpu.to("neuron")

            # Test implicit None
            self._test_clamp_implicit_none_error(input_arr, implicit_none_error)

            # Test explicit None
            self._test_clamp_explicit_none_error(input_arr, explicit_none_error)

    @assert_raises(RuntimeError)
    def _test_clamp_implicit_none_error(self, input_arr, expected_error):
        """Helper method to test clamp implicit None error"""
        torch.clamp(input_arr)

    @assert_raises(RuntimeError)
    def _test_clamp_explicit_none_error(self, input_arr, expected_error):
        """Helper method to test clamp explicit None error"""
        torch.clamp(input_arr, min=None, max=None)

    def test_clamp_min_greater_than_max(self):
        """Test clamp behavior when min is greater than max (should set all elements to max)."""
        with track_neuron_ops():
            # Create a tensor with various values
            input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
            input_arr = input_cpu.to("neuron")

            # Set min > max
            min_val = 8.0
            max_val = 4.0

            # According to PyTorch docs, when min > max, all elements should become max
            expected = torch.clamp(input_cpu, min=min_val, max=max_val)
            result = torch.clamp(input_arr, min=min_val, max=max_val)

            # Verify all elements are set to max_val
            assert torch.all(expected == max_val)
            assert torch.all(result.cpu() == expected)

            # Try a different combination of min > max
            min_val2 = 0.0
            max_val2 = -5.0

            expected2 = torch.clamp(input_cpu, min=min_val2, max=max_val2)
            result2 = torch.clamp(input_arr, min=min_val2, max=max_val2)

            # Verify all elements are set to max_val2
            assert torch.all(expected2 == max_val2)
            assert torch.all(result2.cpu() == expected2)

            assert_op_runs_on_neuron("aten::clamp")

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": 0.0, "max": 5.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": 0.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max", {"max": 5.0}),
        ],
    )
    @pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 2)])
    def test_clamp_ops_shapes(self, op_name, op_func, aten_name, kwargs, shape):
        """Test clamp operations with different tensor shapes."""
        with track_neuron_ops():
            # Create input tensor with appropriate shape containing values outside clamp range
            if shape == ():
                # Scalar case
                input_cpu = torch.tensor(7.0)
                input_arr = input_cpu.to("neuron")
                expected = op_func(input_cpu, **kwargs)
                result = op_func(input_arr, **kwargs)
                assert result.cpu().item() == expected.item()
            else:
                # Create tensor with mixed values (-5, 3, 10) repeating
                tensor_size = 1
                for dim in shape:
                    tensor_size *= dim

                values = []
                for i in range(tensor_size):
                    values.append([-5.0, 3.0, 10.0][i % 3])

                input_cpu = torch.tensor(values).reshape(shape)
                input_arr = input_cpu.to("neuron")

                expected = op_func(input_cpu, **kwargs)
                result = op_func(input_arr, **kwargs)

                assert result.shape == expected.shape
                assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": 0.0, "max": 5.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": 0.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max", {"max": 5.0}),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_clamp_ops_dtypes(self, op_name, op_func, aten_name, kwargs, dtype):
        """Test clamp operations with different data types."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                input_cpu = torch.tensor([-5.5, 1.1, 3.3, 7.7, 10.0], dtype=dtype)
            else:
                input_cpu = torch.tensor([-5, 1, 3, 7, 10], dtype=dtype)

            input_arr = input_cpu.to("neuron")

            # Run on CPU to get expected result
            if dtype.is_floating_point:
                expected = op_func(input_cpu, **kwargs)
                result = op_func(input_arr, **kwargs)
            else:
                int_kwargs = {k: int(v) for k, v in kwargs.items()}
                expected = op_func(input_cpu, **int_kwargs)
                result = op_func(input_arr, **int_kwargs)
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": 0.0, "max": 5.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": 0.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max", {"max": 5.0}),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_clamp_ops_scalar_tensor(self, op_name, op_func, aten_name, kwargs, dtype):
        """Test clamp operations with scalar tensors and different data types."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                input_cpu = torch.tensor(7.5, dtype=dtype)
            else:
                input_cpu = torch.tensor(7, dtype=dtype)

            input_arr = input_cpu.to("neuron")

            # Run on CPU to get expected result
            if dtype.is_floating_point:
                expected = op_func(input_cpu, **kwargs)
                result = op_func(input_arr, **kwargs)
            else:
                int_kwargs = {k: int(v) for k, v in kwargs.items()}
                expected = op_func(input_cpu, **int_kwargs)
                result = op_func(input_arr, **int_kwargs)

            assert result.cpu().item() == expected.item()
            assert result.shape == expected.shape == torch.Size([])
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": 0.0, "max": 5.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": 0.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max", {"max": 5.0}),
        ],
    )
    def test_clamp_ops_tensor_method(self, op_name, op_func, aten_name, kwargs):
        """Test tensor method variants of clamp operations."""
        input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
        input_arr = input_cpu.to("neuron")
        with track_neuron_ops():
            expected = getattr(input_cpu, op_name)(**kwargs)
            result = getattr(input_arr, op_name)(**kwargs)
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp.out", {"min": 0.0, "max": 5.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min.out", {"min": 0.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max.out", {"max": 5.0}),
        ],
    )
    def test_clamp_ops_out_variant(self, op_name, op_func, aten_name, kwargs):
        """Test out variants of clamp operations."""
        with track_neuron_ops():
            input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
            input_arr = input_cpu.to("neuron")

            # Get expected result from CPU
            expected = op_func(input_cpu, **kwargs)

            # Use out variant on neuron
            output = torch.zeros_like(input_arr)
            op_func(input_arr, out=output, **kwargs)
            assert torch.all(output.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": -10.0, "max": 10.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": -10.0}),
            pytest.param(
                "clamp_max",
                torch.clamp_max,
                "aten::clamp_max",
                {"max": 10.0},
                marks=pytest.mark.xfail(
                    condition=os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
                    reason="-inf preservation issues in torch.clamp_max for multi-element tensors",
                ),
            ),
        ],
    )
    def test_clamp_ops_extreme_values(self, op_name, op_func, aten_name, kwargs):
        """Test clamp operations with extreme values (inf, large numbers)."""
        with track_neuron_ops():
            # Empty tensor
            empty_cpu = torch.tensor([])
            empty = empty_cpu.to("neuron")
            expected_empty = op_func(empty_cpu, **kwargs)
            result_empty = op_func(empty, **kwargs)
            assert result_empty.cpu().numel() == expected_empty.numel()
            assert result_empty.cpu().shape == expected_empty.shape

            # Extreme values (excluding NaN)
            extremes_cpu = torch.tensor([float("-inf"), -1e30, -5.0, 0.0, 5.0, 1e30, float("inf")])
            extremes = extremes_cpu.to("neuron")

            # Get expected result from CPU
            expected = op_func(extremes_cpu, **kwargs)
            result = op_func(extremes, **kwargs)

            # Check all elements
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.xfail(
        reason="Neuron clamp implementation doesn't preserve NaN values like CPU does. "
        "On CPU NaNs remain NaN after clamping, but on Neuron they get clamped to min value."
    )
    @pytest.mark.parametrize(
        "op_name,op_func,aten_name,kwargs",
        [
            ("clamp", torch.clamp, "aten::clamp", {"min": 0.0, "max": 10.0}),
            ("clamp_min", torch.clamp_min, "aten::clamp_min", {"min": 3.0}),
            ("clamp_max", torch.clamp_max, "aten::clamp_max", {"max": 10.0}),
        ],
    )
    def test_clamp_ops_with_nan(self, op_name, op_func, aten_name, kwargs):
        """Test clamp operations behavior with NaN values."""
        with track_neuron_ops():
            # Create tensor with NaN
            nan_tensor_cpu = torch.tensor([1.0, float("nan"), 5.0])
            nan_tensor = nan_tensor_cpu.to("neuron")

            # On CPU, NaN should remain NaN after clamping
            expected = op_func(nan_tensor_cpu, **kwargs)
            result = op_func(nan_tensor, **kwargs)

            # The second element should be NaN in both cases
            assert torch.isnan(expected[1])
            assert torch.isnan(result.cpu()[1])
            assert_op_runs_on_neuron(aten_name)

    @pytest.mark.parametrize(
        "op_name,op_func",
        [
            ("clamp", torch.clamp),
            ("clamp_min", torch.clamp_min),
            ("clamp_max", torch.clamp_max),
        ],
    )
    def test_clamp_ops_broadcasting(self, op_name, op_func):
        """Test clamp operations with broadcasted tensors."""
        suffix = (
            ".Tensor" if os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1" else ".Tensor_out"
        )
        aten_name = f"aten::{op_name}{suffix}"

        with track_neuron_ops():
            # 2x3 input tensor
            input_cpu = torch.tensor([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]])
            input_arr = input_cpu.to("neuron")

            if op_name == "clamp":
                # Vector min (will broadcast across rows)
                min_cpu = torch.tensor([2.0, 4.0, 6.0])
                # Vector max (will broadcast across columns)
                max_cpu = torch.tensor([[6.0], [8.0]])
                expected = op_func(input_cpu, min=min_cpu, max=max_cpu)
                result = op_func(input_arr, min=min_cpu.to("neuron"), max=max_cpu.to("neuron"))
            elif op_name == "clamp_min":
                # Vector min (will broadcast across rows)
                min_cpu = torch.tensor([2.0, 4.0, 6.0])
                expected = op_func(input_cpu, min=min_cpu)
                result = op_func(input_arr, min=min_cpu.to("neuron"))
            else:  # clamp_max
                # Vector max (will broadcast across columns)
                max_cpu = torch.tensor([[6.0], [8.0]])
                expected = op_func(input_cpu, max=max_cpu)
                result = op_func(input_arr, max=max_cpu.to("neuron"))

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron(aten_name)
