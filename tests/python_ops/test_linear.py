"""Test that linear operation is properly registered with PyTorch dispatcher."""

import logging
import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLinearRegistration:
    """Test zeros operation registration and functionality."""

    def setup_method(self):
        """Set up test environment before each test method."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            pytest.param((4,), (4,)),
            pytest.param((4,), (2, 4)),
            pytest.param((3, 4), (4,)),
            pytest.param((3, 4), (5, 4)),
            pytest.param((2, 3, 4), (6, 4)),
            pytest.param((1, 2, 3, 4), (6, 4)),
            pytest.param((128, 512), (256, 512)),
        ],
    )
    def test_linear_basic(self, shape1, shape2):
        """Test basic torch.nn.functional.linear functionality with various tensor shapes"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1)
            weight_cpu = torch.rand(shape2)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu)
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron)
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3",
        [
            pytest.param((4,), (4,), ()),
            pytest.param((4,), (2, 4), (2,)),
            pytest.param((3, 4), (5, 4), (5,)),
            pytest.param((2, 3, 4), (6, 4), (6,)),
            pytest.param((1, 2, 3, 4), (6, 4), (6,)),
            pytest.param((128, 512), (256, 512), (256,)),
        ],
    )
    def test_linear_bias(self, shape1, shape2, shape3):
        """Test basic gunctionality with various tensor shapes and bias"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1)
            weight_cpu = torch.rand(shape2)
            bias_cpu = torch.rand(shape3)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,dtype",
        [
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.float32,
            ),
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.float16,
            ),
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.bfloat16,
            ),
        ],
    )
    def test_linear_dtype(self, shape1, shape2, shape3, dtype):
        """Test basic torch.nn.functional.linear functionality with various dtypes"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1, dtype=dtype)
            weight_cpu = torch.rand(shape2, dtype=dtype)
            bias_cpu = torch.rand(shape3, dtype=dtype)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert output_neuron.dtype == dtype
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,dtype",
        [
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.int16,
            ),
        ],
    )
    def test_linear_int_dtype(self, shape1, shape2, shape3, dtype):
        """Test that unsupported integer dtypes raise appropriate error messages"""
        with track_neuron_ops():
            input_cpu = torch.randint(0, 10, shape1, dtype=dtype)
            weight_cpu = torch.randint(0, 10, shape2, dtype=dtype)
            bias_cpu = torch.randint(0, 10, shape3, dtype=dtype)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert output_neuron.dtype == dtype
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    def test_linear_identity(self):
        with track_neuron_ops():
            shape1 = (3, 3)
            input_cpu = torch.rand(shape1)
            input_neuron = input_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, input_cpu)
            output_neuron = torch.nn.functional.linear(input_neuron, input_neuron)
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    def test_linear_empty_tensors(self):
        """Test linear with empty tensors"""
        with track_neuron_ops():
            input_neuron = torch.empty(0, 4).to("neuron")
            weight_neuron = torch.randn(5, 4).to("neuron")
            bias_neuron = torch.randn(5).to("neuron")

            output_cpu = torch.nn.functional.linear(
                input_neuron.cpu(), weight_neuron.cpu(), bias_neuron.cpu()
            )
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )

            assert output_neuron.shape == (0, 5)
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::linear")

    @pytest.mark.parametrize(
        "special_value",
        [
            pytest.param(0.0),  # if pass 0, dtype will be int64
            pytest.param(torch.inf),
            pytest.param(-torch.inf),
        ],
    )
    def test_linear_special_values(self, special_value):
        """Test linear with special values (zeros, inf)"""
        with track_neuron_ops():
            # Test with zero input
            input_neuron = torch.full((3, 4), special_value).to("neuron")
            weight_neuron = torch.rand(5, 4).to("neuron")
            bias_neuron = torch.rand(5).to("neuron")

            output_cpu = torch.nn.functional.linear(
                input_neuron.cpu(), weight_neuron.cpu(), bias_neuron.cpu()
            )
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron, bias_neuron)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::linear")

    def test_linear_nan(self):
        with track_neuron_ops():
            # Test with zero input
            input_neuron = torch.full((3, 4), float("nan")).to("neuron")
            weight_neuron = torch.rand(5, 4).to("neuron")
            bias_neuron = torch.rand(5).to("neuron")
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron, bias_neuron)

            assert torch.isnan(output_neuron).all()
            assert_op_runs_on_neuron("aten::linear")


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLinearBackward:
    """Test cases for linear backward operations"""

    @pytest.mark.parametrize(
        "input_shape,weight_shape,bias_shape",
        [
            ((3, 4), (5, 4), (5,)),  # Basic 2D with bias
            ((4,), (5, 4), (5,)),  # 1D input with bias
            ((3, 4), (5, 4), None),  # 2D without bias
            ((4,), (5, 4), None),  # 1D input without bias
            ((2, 3, 4), (5, 4), (5,)),  # 3D input with bias
            ((1, 2, 3, 4), (5, 4), (5,)),  # 4D input with bias
        ],
    )
    @pytest.mark.parametrize(
        "dtype,rtol,atol",
        [
            (torch.float32, 1e-4, 1e-4),
            (torch.float16, 1e-2, 1e-2),
            (torch.bfloat16, 1e-2, 1e-2),
        ],
    )
    @pytest.mark.parametrize(
        "requires_grads",
        [
            # input, weight, bias
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
            # (False, False, False), skipped, as it trigger runtime error in pytorch
        ],
    )
    def test_linear_backward_basic(
        self, input_shape, weight_shape, bias_shape, dtype, rtol, atol, requires_grads
    ):
        """Test basic linear backward functionality with various tensor shapes"""
        # Create input tensors
        input_tensor = torch.randn(*input_shape, requires_grad=requires_grads[0], dtype=dtype)
        weight_tensor = torch.randn(*weight_shape, requires_grad=requires_grads[1], dtype=dtype)
        bias_tensor = None
        if bias_shape is not None:
            bias_tensor = torch.randn(*bias_shape, requires_grad=requires_grads[2], dtype=dtype)
        elif requires_grads[2]:  # skip cases: bias=None and requires_grad, cause errors in CPU
            return

        # Clone for neuron device
        input_neuron = input_tensor.detach().clone().to("neuron").requires_grad_(requires_grads[0])
        weight_neuron = (
            weight_tensor.detach().clone().to("neuron").requires_grad_(requires_grads[1])
        )
        bias_neuron = None
        if bias_tensor is not None:
            bias_neuron = (
                bias_tensor.detach().clone().to("neuron").requires_grad_(requires_grads[2])
            )

        with track_neuron_ops():
            # Forward pass
            result_cpu = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)

            # Create gradient tensor
            grad_output = torch.randn_like(result_cpu)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            result_cpu.backward(
                grad_output
            )  # we cannot call linear_backward for tensors on CPU, because it's not implemented
            input_neuron.grad, weight_neuron.grad, bias_neuron_grad = (
                torch.ops.aten.linear_backward(
                    input_neuron, grad_output_neuron, weight_neuron, requires_grads
                )
            )
            if bias_neuron is not None:
                bias_neuron.grad = bias_neuron_grad
            # Verify operations ran on Neuron
            assert_op_runs_on_neuron("aten::linear")

        # Check that gradients match
        if requires_grads[0]:
            assert input_neuron.grad is not None
            assert input_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(
                input_neuron.grad.cpu(), input_tensor.grad, rtol=rtol, atol=atol
            )
        else:
            assert input_neuron.grad is None

        if requires_grads[1]:
            assert weight_neuron.grad is not None
            assert weight_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(
                weight_neuron.grad.cpu(), weight_tensor.grad, rtol=rtol, atol=atol
            )
        elif not requires_grads[1] and not requires_grads[2]:
            assert weight_neuron.grad is None

        if bias_tensor is not None and requires_grads[2]:
            assert bias_neuron.grad is not None
            assert bias_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(
                bias_neuron.grad.cpu(), bias_tensor.grad, rtol=rtol, atol=atol
            )
        elif bias_tensor is not None and not requires_grads[1] and not requires_grads[2]:
            assert bias_neuron.grad is None
