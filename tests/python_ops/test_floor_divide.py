"""Test that floor_divide operation is properly registered with PyTorch dispatcher."""

import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestFloorDivide:
    def test_floor_divide_basic(self):
        """Test basic floor_divide functionality."""
        with track_neuron_ops():
            x = torch.tensor([10, 9, 8, 7, 6]).to("neuron")
            y = torch.tensor([3, 3, 3, 3, 3]).to("neuron")

            # CPU reference
            expected = torch.floor_divide(x.cpu(), y.cpu())

            # Neuron result
            result = torch.floor_divide(x, y)

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_empty_tensor(self):
        """Test floor_divide with empty tensors."""
        # First check CPU behavior
        x_cpu = torch.tensor([2], dtype=torch.float32)
        y_cpu = torch.tensor([], dtype=torch.float32)
        expected = torch.floor_divide(x_cpu, y_cpu)

        # Now test on neuron
        with track_neuron_ops():
            x = x_cpu.to("neuron")
            y = y_cpu.to("neuron")

            result = torch.floor_divide(x, y)

            # Compare with CPU result
            assert result.size() == expected.size()
            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_tensor_method(self):
        """Test tensor.floor_divide() method works as expected."""
        with track_neuron_ops():
            x = torch.tensor([10, 9, 8, 7, 6]).to("neuron")
            y = torch.tensor([3, 3, 3, 3, 3]).to("neuron")

            # CPU reference
            expected = x.cpu().floor_divide(y.cpu())

            # Use method syntax
            result = x.floor_divide(y)

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    @pytest.mark.parametrize(
        "x_val, y_val, dtype",
        [
            # Float division (different from standard division)
            ([3.9, -3.9, 2.5], [2.0, 2.0, 1.5], torch.float32),
            # Negative numbers (edge case)
            ([-9.0, -9.0, 9.0, 9.0], [4.0, -4.0, 4.0, -4.0], torch.float32),
        ],
        ids=["float", "negatives"],
    )
    def test_floor_divide_float_values(self, x_val, y_val, dtype):
        """Test floor_divide with floating point values (which have special behavior)."""
        with track_neuron_ops():
            x = torch.tensor(x_val, dtype=dtype).to("neuron")
            y = torch.tensor(y_val, dtype=dtype).to("neuron")

            # Use PyTorch CPU as reference for validation
            x_cpu = torch.tensor(x_val, dtype=dtype)
            y_cpu = torch.tensor(y_val, dtype=dtype)
            expected = torch.floor_divide(x_cpu, y_cpu)

            result = torch.floor_divide(x, y)

            assert torch.allclose(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_broadcasting(self):
        """Test floor_divide with broadcasting."""
        with track_neuron_ops():
            x = torch.tensor([[10, 9, 8], [7, 6, 5]]).to("neuron")
            y = torch.tensor([2, 3, 4]).to("neuron")

            # CPU reference
            expected = torch.floor_divide(x.cpu(), y.cpu())

            result = torch.floor_divide(x, y)

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_scalar(self):
        """Test floor_divide with scalar divisor."""
        with track_neuron_ops():
            x = torch.tensor([10, 9, 8, 7, 6]).to("neuron")
            scalar = 3

            # CPU reference
            expected = torch.floor_divide(x.cpu(), scalar)

            result = torch.floor_divide(x, scalar)

            assert torch.all(result.cpu() == expected)
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_out_parameter(self):
        """Test floor_divide with output tensor."""
        with track_neuron_ops():
            x = torch.tensor([10, 9, 8]).to("neuron")
            y = torch.tensor([3, 3, 3]).to("neuron")
            out = torch.zeros_like(x).to("neuron")

            # CPU reference with out parameter
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            out_cpu = torch.zeros_like(x_cpu)
            torch.floor_divide(x_cpu, y_cpu, out=out_cpu)

            # Neuron with out parameter
            result = torch.floor_divide(x, y, out=out)

            # Check that out was modified in-place
            assert torch.all(out.cpu() == out_cpu)
            # Check that result is the same as out
            assert result is out
            assert_op_runs_on_neuron("aten::floor_divide")

    def test_floor_divide_type_promotion(self):
        """Test floor_divide handles type promotion correctly."""
        with track_neuron_ops():
            # Integer tensor / float tensor should promote to float
            x = torch.tensor([10, 9, 8], dtype=torch.int32).to("neuron")
            y = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32).to("neuron")

            # CPU reference
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            expected = torch.floor_divide(x_cpu, y_cpu)

            result = torch.floor_divide(x, y)

            # Check output values and dtype
            assert torch.allclose(result.cpu(), expected)
            assert result.dtype == expected.dtype
            assert_op_runs_on_neuron("aten::floor_divide")

    @pytest.mark.xfail(reason="Division by zero does not align between neuron and CPU")
    def test_floor_divide_zero_division(self):
        """Test floor_divide with zero divisor."""
        # Check CPU behavior
        x_cpu = torch.tensor([1.0, 2.0, 3.0])
        y_cpu = torch.tensor([1.0, 0.0, 2.0])
        expected = torch.floor_divide(x_cpu, y_cpu)

        # Now check Neuron behavior
        with track_neuron_ops():
            x = x_cpu.to("neuron")
            y = y_cpu.to("neuron")
            result = torch.floor_divide(x, y)

            # Directly compare with CPU results - they should match exactly
            # Including handling of inf values for division by zero
            assert torch.all(torch.isnan(result.cpu()) == torch.isnan(expected))
            assert torch.all(torch.isinf(result.cpu()) == torch.isinf(expected))
            assert torch.all(torch.isfinite(result.cpu()) == torch.isfinite(expected))

            # For finite values, check exact equality
            finite_mask = torch.isfinite(expected)
            assert torch.all(result.cpu()[finite_mask] == expected[finite_mask])

            # For inf values, make sure the signs match
            inf_mask = torch.isinf(expected)
            if inf_mask.any():
                assert torch.all(
                    result.cpu()[inf_mask] == expected[inf_mask]
                ), "Inf values should match including sign"

            assert_op_runs_on_neuron("aten::floor_divide")
