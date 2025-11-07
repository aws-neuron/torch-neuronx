"""Test that clamp operation is properly registered with PyTorch dispatcher."""

import re

import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuron.device_count() == 0, reason="No Neuron devices available")
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

    def test_clamp_tensor_method(self):
        """Test the tensor.clamp() method variant."""
        # Create input tensor
        input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
        input_arr = input_cpu.to("neuron")

        with track_neuron_ops():
            # Test regular method variant
            expected_method = input_cpu.clamp(min=0.0, max=5.0)
            method_result = input_arr.clamp(min=0.0, max=5.0)
            assert torch.all(method_result.cpu() == expected_method)

            # The tensor method should dispatch to the same ATen op
            assert_op_runs_on_neuron("aten::clamp")

    def test_clamp_none_args(self):
        """Test that clamp properly errors when both min and max are None."""
        # Create input tensor
        input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])

        # Get CPU error message for implicit None
        try:
            torch.clamp(input_cpu)
        except RuntimeError as e:
            implicit_none_error = str(e)

        # Get CPU error message for explicit None
        try:
            torch.clamp(input_cpu, min=None, max=None)
        except RuntimeError as e:
            explicit_none_error = str(e)

        # Now test neuron behavior
        with track_neuron_ops():
            input_arr = input_cpu.to("neuron")

            # Test implicit None
            with pytest.raises(RuntimeError, match=re.escape(implicit_none_error)):
                torch.clamp(input_arr)

            # Test explicit None
            with pytest.raises(RuntimeError, match=re.escape(explicit_none_error)):
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
        "shape",
        [
            (),  # scalar
            (5,),  # 1D vector
            (2, 3),  # 2D matrix
            (2, 3, 2),  # 3D tensor
        ],
    )
    def test_clamp_shapes(self, shape):
        """Test clamp with different tensor shapes."""
        with track_neuron_ops():
            # Create input tensor with appropriate shape containing values outside clamp range
            if shape == ():
                # Scalar case
                input_cpu = torch.tensor(7.0)
                input_arr = input_cpu.to("neuron")
                expected = torch.clamp(input_cpu, min=0.0, max=5.0)
                result = torch.clamp(input_arr, min=0.0, max=5.0)
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

                expected = torch.clamp(input_cpu, min=0.0, max=5.0)
                result = torch.clamp(input_arr, min=0.0, max=5.0)

                assert result.shape == expected.shape
                assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::clamp")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_clamp_dtypes(self, dtype):
        """Test clamp with different data types."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                input_cpu = torch.tensor([-5.5, 1.1, 3.3, 7.7, 10.0], dtype=dtype)
            else:
                input_cpu = torch.tensor([-5, 1, 3, 7, 10], dtype=dtype)

            input_arr = input_cpu.to("neuron")

            # Run on CPU to get expected result
            if dtype.is_floating_point:
                expected = torch.clamp(input_cpu, min=0.0, max=5.0)
                result = torch.clamp(input_arr, min=0.0, max=5.0)
            else:
                expected = torch.clamp(input_cpu, min=0, max=5)
                result = torch.clamp(input_arr, min=0, max=5)

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::clamp")

    def test_clamp_extreme_values(self):
        """Test clamp with extreme values (inf, large numbers)."""
        with track_neuron_ops():
            # Empty tensor
            empty_cpu = torch.tensor([])
            empty = empty_cpu.to("neuron")
            expected_empty = torch.clamp(empty_cpu, min=0.0, max=5.0)
            result_empty = torch.clamp(empty, min=0.0, max=5.0)
            assert result_empty.cpu().numel() == expected_empty.numel()
            assert result_empty.cpu().shape == expected_empty.shape

            # Extreme values (excluding NaN)
            extremes_cpu = torch.tensor([float("-inf"), -1e30, -5.0, 0.0, 5.0, 1e30, float("inf")])
            extremes = extremes_cpu.to("neuron")

            # Get expected result from CPU
            expected = torch.clamp(extremes_cpu, min=-10.0, max=10.0)
            result = torch.clamp(extremes, min=-10.0, max=10.0)

            # Check all elements
            assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::clamp")

    @pytest.mark.xfail(
        reason="Neuron clamp implementation doesn't preserve NaN values like CPU does. "
        "On CPU NaNs remain NaN after clamping, but on Neuron they get clamped to min value."
    )
    def test_clamp_with_nan(self):
        """Test clamp behavior with NaN values."""
        with track_neuron_ops():
            # Create tensor with NaN
            nan_tensor_cpu = torch.tensor([1.0, float("nan"), 5.0])
            nan_tensor = nan_tensor_cpu.to("neuron")

            # On CPU, NaN should remain NaN after clamping
            expected = torch.clamp(nan_tensor_cpu, min=0.0, max=10.0)
            result = torch.clamp(nan_tensor, min=0.0, max=10.0)

            # The second element should be NaN in both cases
            assert torch.isnan(expected[1])
            assert torch.isnan(result.cpu()[1])

            # Try another case with min only
            expected_min = torch.clamp(nan_tensor_cpu, min=3.0)
            result_min = torch.clamp(nan_tensor, min=3.0)
            assert torch.isnan(expected_min[1])
            assert torch.isnan(result_min.cpu()[1])

            # And with max only
            expected_max = torch.clamp(nan_tensor_cpu, max=3.0)
            result_max = torch.clamp(nan_tensor, max=3.0)
            assert torch.isnan(expected_max[1])
            assert torch.isnan(result_max.cpu()[1])

            assert_op_runs_on_neuron("aten::clamp")

    def test_clamp_out_variant(self):
        """Test the clamp.out variant."""
        with track_neuron_ops():
            input_cpu = torch.tensor([-5.0, 1.0, 3.0, 7.0, 10.0])
            input_arr = input_cpu.to("neuron")

            # Get expected result from CPU
            expected = torch.clamp(input_cpu, min=0.0, max=5.0)

            # Use out variant on neuron
            output = torch.zeros_like(input_arr)
            torch.clamp(input_arr, min=0.0, max=5.0, out=output)

            assert torch.all(output.cpu() == expected)
            assert_op_runs_on_neuron("aten::clamp.out")

    def test_clamp_broadcasting(self):
        """Test clamp with broadcasted min/max tensors."""
        with track_neuron_ops():
            # 2x3 input tensor
            input_cpu = torch.tensor([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]])

            # Vector min (will broadcast across rows)
            min_cpu = torch.tensor([2.0, 4.0, 6.0])

            # Vector max (will broadcast across columns)
            max_cpu = torch.tensor([[6.0], [8.0]])

            # Get expected result from CPU
            expected = input_cpu.clamp(min=min_cpu, max=max_cpu)

            # Now run on neuron
            input_arr = input_cpu.to("neuron")
            min_tensor = min_cpu.to("neuron")
            max_tensor = max_cpu.to("neuron")
            result = input_arr.clamp(min=min_tensor, max=max_tensor)
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::clamp.Tensor_out")
