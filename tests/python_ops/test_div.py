import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestDiv:
    """Test cases for element-wise division operation"""

    def setup_method(self):
        """Set random seed for reproducible tests"""
        torch.manual_seed(42)

    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_runs_on_neuron(self, rounding_mode):
        """Test div runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([10.0, 15.0, 20.0], device="neuron")
            b = torch.tensor([2.0, 3.0, 4.0], device="neuron")
            result = torch.div(a, b, rounding_mode=rounding_mode)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div")

    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_basic(self, rounding_mode):
        """Test basic element-wise division"""
        with track_neuron_ops():
            # Create CPU tensors
            a_cpu = torch.randn((3, 4))
            b_cpu = torch.randn((3, 4))

            # Move to Neuron device
            a_neuron = a_cpu.to("neuron")
            b_neuron = b_cpu.to("neuron")

            # Perform division
            result_neuron = torch.div(a_neuron, b_neuron, rounding_mode=rounding_mode)

            # Compare with CPU result
            expected = torch.div(a_cpu, b_cpu, rounding_mode=rounding_mode)
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert result_neuron.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div")

    def test_div_out(self):
        """Test division with output tensor"""
        with track_neuron_ops():
            # Create CPU tensors
            a_cpu = torch.tensor([4.0, 6.0, 8.0, 10.0])
            b_cpu = torch.tensor([2.0, 3.0, 4.0, 5.0])
            out_cpu = torch.empty_like(a_cpu)

            # Move to Neuron device
            a_neuron = a_cpu.to("neuron")
            b_neuron = b_cpu.to("neuron")
            out_neuron = torch.empty_like(a_neuron)

            # Perform division with output tensor
            result_neuron = torch.div(a_neuron, b_neuron, out=out_neuron)

            # Verify result is the same as output tensor
            assert result_neuron is out_neuron

            # Compare with CPU result
            expected = torch.div(a_cpu, b_cpu, out=out_cpu)
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert result_neuron.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div.out")

    def test_div_in_place(self):
        """Test division with in-place div_ operation"""
        with track_neuron_ops():
            # Create CPU tensors
            a_cpu = torch.randn((3, 4))
            b_cpu = torch.randn((3, 4))

            # Move to Neuron device
            a_neuron = a_cpu.to("neuron")
            b_neuron = b_cpu.to("neuron")

            # Perform in-place devision
            a_cpu.div_(b_cpu)
            torch.ops.aten.div_(a_neuron, b_neuron)

            # Compare with CPU result
            torch.testing.assert_close(a_neuron.cpu(), a_cpu)
            assert a_neuron.device.type == "neuron"
            # The in-place operation div_ is being handled by the div/div.out variant
            assert_op_runs_on_neuron("aten::div")

    def test_div_broadcast(self):
        """Test division with broadcasting"""
        with track_neuron_ops():
            # Create CPU tensors with different shapes
            a_cpu = torch.tensor([[4.0, 6.0], [8.0, 10.0]])
            b_cpu = torch.tensor([2.0, 3.0])

            # Move to Neuron device
            a_neuron = a_cpu.to("neuron")
            b_neuron = b_cpu.to("neuron")

            # Perform division (should broadcast)
            result_neuron = torch.div(a_neuron, b_neuron)

            # Compare with CPU result
            expected = torch.div(a_cpu, b_cpu)
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert a_neuron.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
    )
    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_different_dtypes(self, dtype, rounding_mode):
        """Test division with different data types and rounding modes"""
        with track_neuron_ops():
            # Create CPU tensors
            if dtype in [torch.int32, torch.int64]:
                a_cpu = torch.randint(1, 10, (3, 4), dtype=dtype)
                b_cpu = torch.randint(1, 5, (3, 4), dtype=dtype)
            else:
                a_cpu = torch.randn((3, 4), dtype=dtype)
                b_cpu = torch.randn((3, 4), dtype=dtype)

            # Move to Neuron device
            a_neuron = a_cpu.to("neuron")
            b_neuron = b_cpu.to("neuron")

            # Perform division with rounding
            result_neuron = torch.div(a_neuron, b_neuron, rounding_mode=rounding_mode)

            # Compare with CPU result
            expected = torch.div(a_cpu, b_cpu, rounding_mode=rounding_mode)
            torch.testing.assert_close(result_neuron.cpu(), expected)
            assert result_neuron.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div")

    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_empty(self, rounding_mode):
        """Test division with empty tensors"""
        # Create empty CPU tensors
        a_cpu = torch.empty(0)
        b_cpu = torch.empty(0)

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform division
        result_neuron = torch.div(a_neuron, b_neuron, rounding_mode=rounding_mode)
        # Compare with CPU result
        expected = torch.div(a_cpu, b_cpu, rounding_mode=rounding_mode)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_scalar_broadcast(self, rounding_mode):
        """Test division with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.randn((3, 4))
        b_cpu = torch.tensor(2.0)  # scalar-like tensor

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform division
        result_neuron = torch.div(a_neuron, b_neuron, rounding_mode=rounding_mode)
        # Compare with CPU result
        expected = torch.div(a_cpu, b_cpu, rounding_mode=rounding_mode)
        torch.testing.assert_close(result_neuron.cpu(), expected)

    @pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
    def test_div_by_zero(self, rounding_mode):
        """Test division by zero behavior"""
        # Create CPU tensors
        a_cpu = torch.tensor([1.0, -1.0, 0.0])
        b_cpu = torch.tensor([0.0, 0.0, 0.0])

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform division
        result_neuron = torch.div(a_neuron, b_neuron, rounding_mode=rounding_mode)

        # Compare with CPU result (should be inf, -inf, nan)
        expected = torch.div(a_cpu, b_cpu, rounding_mode=rounding_mode)
        torch.testing.assert_close(result_neuron.cpu(), expected, equal_nan=True)

    def test_div_first_operand_python_scalar(self):
        """Test division with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.randn((3, 4))
        b_cpu = 2.0
        expected = b_cpu / a_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform division
        result_neuron = b_cpu / a_neuron

        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_div_second_operand_python_scalar(self):
        """Test division with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.randn((3, 4))
        b_cpu = 2.0
        expected = a_cpu / b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")

        # Perform division
        result_neuron = a_neuron / b_cpu

        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_div_type_promotions(self):
        """Test division with scalar-like tensor"""
        # Create CPU tensors
        a_cpu = torch.randn(4, dtype=torch.float32)
        b_cpu = torch.randint(-10, 10, size=(), dtype=torch.int32)
        expected = a_cpu / b_cpu

        # Move to Neuron device
        a_neuron = a_cpu.to("neuron")
        b_neuron = b_cpu.to("neuron")

        # Perform division
        result_neuron = a_neuron / b_neuron

        torch.testing.assert_close(result_neuron.cpu(), expected)

    def test_div_scalar_zero_divisor(self):
        """Test div with scalar zero divisor"""
        with track_neuron_ops():
            a = torch.tensor([10.0, 20.0], device="neuron")
            result = torch.div(a, 0, rounding_mode="floor")
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::div")
