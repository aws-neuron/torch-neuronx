import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestTanh:
    """Test cases for tanh registration and operation"""

    def test_tanh_runs_on_neuron(self):
        """Test tanh runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            a = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            result = torch.tanh(a)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::tanh")

    def test_tanh_basic(self):
        """Test basic tanh operation"""
        input_cpu = torch.tensor([4.0, -9.0, 0.0, 16.0, -25.0])
        input_neuron = torch.tensor([4.0, -9.0, 0.0, 16.0, -25.0], device="neuron")

        result = torch.tanh(input_neuron)
        expected = torch.tanh(input_cpu)

        torch.testing.assert_close(result.cpu(), expected)

    def test_tanh_output(self):
        """Test tanh with Pre-allocate output tensors"""
        input_neuron = torch.tensor([1.0, -4.0, 0.0, 9.0], device="neuron")
        output_neuron = torch.empty_like(input_neuron)

        torch.tanh(input_neuron, out=output_neuron)

        input_cpu = torch.tensor([1.0, -4.0, 0.0, 9.0])
        expected = torch.empty_like(input_cpu)

        torch.tanh(input_cpu, out=expected)

        torch.testing.assert_close(output_neuron.cpu(), expected)

    def test_tanh_with_inf(self):
        """Test tanh operation with Inf and large input values (should approach 1 or -1)"""
        input_neuron = torch.tensor([float("-inf"), -1000.0, 1000, float("inf")], device="neuron")
        input_cpu = torch.tensor([float("-inf"), -1000.0, 1000, float("inf")])

        result = torch.tanh(input_neuron)
        expected = torch.tanh(input_cpu)

        torch.testing.assert_close(result.cpu(), expected)

    def test_tanh_with_nan(self):
        """Test tanh operation with Inf and large input values (should approach 1 or -1)"""
        input_neuron = torch.tensor([float("nan")], device="neuron", dtype=torch.float32)
        input_cpu = input_neuron.cpu()

        result = torch.tanh(input_neuron)
        expected = torch.tanh(input_cpu)

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4, equal_nan=True)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_tanh_dtypes(self, dtype):
        """Test tanh with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            tensor_neuron = torch.randint(0, 2, (2, 3), device=device, dtype=dtype)
            tensor_cpu = tensor_neuron.cpu()

            result_neuron = torch.tanh(tensor_neuron)
            result_cpu = torch.tanh(tensor_cpu)

            assert result_neuron.dtype == dtype

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::tanh")

    def test_tanh_requires_grad(self):
        """Test tanh with requires_grad parameter"""
        device = "neuron"
        with track_neuron_ops():
            tensor_neuron = torch.randn(3, 4, device=device, requires_grad=True)
            tensor_cpu = tensor_neuron.cpu()

            result_neuron = torch.tanh(tensor_neuron)
            result_cpu = torch.tanh(tensor_cpu)
            assert result_neuron.requires_grad

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::tanh")

    def test_tanh_backward(self):
        """Test tanh with backward function"""
        with track_neuron_ops():
            a = torch.randn(4, device="cpu", requires_grad=True)
            a_neuron = a.detach().clone().to("neuron")
            a_neuron.requires_grad = True

            b = torch.tanh(a)
            loss = b.sum()
            loss.backward()

            b_neuron = torch.tanh(a_neuron)
            loss_neuron = b_neuron.sum()
            loss_neuron.backward()

            torch.testing.assert_close(a_neuron.grad.cpu(), a.grad)
            assert_op_runs_on_neuron("aten::tanh_backward")

    @pytest.mark.parametrize(
        "shape",
        [
            (0,),
            (1,),
            (2, 2),
            (2, 1, 2),
            (1, 2, 1, 2),
            (0, 5),
            (3, 0),
            (0, 0),
        ],
    )
    def test_reciprocal_different_shapes(self, shape):
        """Test tanh operator with different tensor shapes"""
        with track_neuron_ops():
            tensor_cpu = torch.randn(shape).abs() + 0.1
            tensor_neuron = tensor_cpu.to("neuron")

            result = torch.tanh(tensor_neuron)
            expected = torch.tanh(tensor_cpu)

            assert result.shape == shape
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::tanh")
