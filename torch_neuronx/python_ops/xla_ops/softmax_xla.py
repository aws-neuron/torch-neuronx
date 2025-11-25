"""XLA implementation of softmax and log_softmax operations."""

import jax.nn as jnn
import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation


@neuron_unary_op("aten::_softmax")
class SoftmaxXLAImpl(UnaryOpImplementation):
    """XLA implementation of softmax operation."""

    def __init__(self):
        # Create kernel for softmax operation
        def softmax_computation(x, dim, output_dtype):
            result = jnn.softmax(x, axis=dim)
            # If half_to_float is True, we need to convert the result to float32
            # This is used by PyTorch's _softmax function
            if output_dtype is not None:
                result = result.astype(output_dtype)
            return result

        self.kernel = TorchNeuronXLAKernel(softmax_computation, "softmax", static_argnums=(1, 2))

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if len(args) < 2:
            return False

        input_tensor = args[0]
        dim = args[1]

        # Check that dim is an integer
        if not isinstance(dim, int):
            return False

        # Tensor must be on Neuron device
        return input_tensor.device.type == "neuron"

    def _execute_impl(
        self,
        input: torch.Tensor,
        dim: int,
        half_to_float: bool,
        *,
        out=None,
    ) -> ExecutionResult:
        """Execute softmax operation using XLA."""
        try:
            # If half_to_float is True, we need to convert the result to float32
            if half_to_float and input.dtype == torch.float16:
                output_dtype = torch.float32
            else:
                output_dtype = input.dtype

            # Use provided output tensor or create a new one
            output = (
                torch.empty(
                    input.shape,
                    dtype=output_dtype,
                    device=input.device,
                    requires_grad=input.requires_grad,
                )
                if out is None
                else out
            )

            # Execute kernel
            res = self.kernel(input, dim, output_dtype, output=output)

            return ExecutionResult(success=True, output=res)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_unary_op("aten::_log_softmax")
class LogSoftmaxXLAImpl(UnaryOpImplementation):
    """XLA implementation of log_softmax operation."""

    def __init__(self):
        # Create kernel for log_softmax operation
        def log_softmax_computation(x, dim, output_dtype):
            result = jnn.log_softmax(x, axis=dim)
            # If half_to_float is True, we need to convert the result to float32
            # This is used by PyTorch's internal functions
            if output_dtype is not None:
                result = result.astype(output_dtype)
            return result

        self.kernel = TorchNeuronXLAKernel(
            log_softmax_computation, "log_softmax", static_argnums=(1, 2)
        )

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if len(args) < 2:
            return False

        input_tensor = args[0]
        dim = args[1]

        # Check that dim is an integer
        if not isinstance(dim, int):
            return False

        # Tensor must be on Neuron device
        return input_tensor.device.type == "neuron"

    def _execute_impl(
        self,
        input: torch.Tensor,
        dim: int,
        half_to_float: bool = False,
        *,
        out=None,
    ) -> ExecutionResult:
        """Execute log_softmax operation using XLA."""
        try:
            # If half_to_float is True, we need to convert the result to float32
            if half_to_float and input.dtype == torch.float16:
                output_dtype = torch.float32
            else:
                output_dtype = input.dtype

            output = (
                torch.empty(
                    input.shape,
                    dtype=output_dtype,
                    device=input.device,
                    requires_grad=input.requires_grad,
                )
                if out is None
                else out
            )

            # Execute kernel
            self.kernel(input, dim, output_dtype, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_unary_op("aten::_log_softmax_backward_data")
class LogSoftmaxBackwardDataXLAImpl(UnaryOpImplementation):
    """XLA implementation of log_softmax backward operation."""

    def __init__(self):
        # Create kernel for log_softmax backward operation
        def log_softmax_grad(grad_output, output, dim, input_dtype):
            # Compute sum of gradients
            sum_grad = jnp.sum(grad_output, axis=dim)

            # Reshape sum_grad to match broadcasting
            broadcast_shape = list(grad_output.shape)
            broadcast_shape[dim] = 1
            sum_grad = sum_grad.reshape(broadcast_shape)

            # Compute gradient
            result = grad_output - jnp.exp(output) * sum_grad

            # Convert to input_dtype if specified
            if input_dtype is not None:
                result = result.astype(input_dtype)

            return result

        self.kernel = TorchNeuronXLAKernel(
            log_softmax_grad, "_log_softmax_backward_data", static_argnums=(2, 3)
        )

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if len(args) < 3:
            return False

        grad_output = args[0]
        output = args[1]
        dim = args[2]

        # Check that dim is an integer
        if not isinstance(dim, int):
            return False

        # Tensors must be on Neuron device
        return grad_output.device.type == "neuron" and output.device.type == "neuron"

    def _execute_impl(
        self,
        grad_output: torch.Tensor,
        output: torch.Tensor,
        dim: int,
        input_dtype: torch.dtype = None,
        *,
        out=None,
    ) -> ExecutionResult:
        """Execute log_softmax backward operation using XLA."""
        try:
            # Use provided output tensor or create a new one
            grad_input = (
                torch.empty(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
                if out is None
                else out
            )

            # Execute kernel
            self.kernel(grad_output, output, dim, input_dtype, output=grad_input)

            return ExecutionResult(success=True, output=grad_input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_unary_op("aten::_softmax_backward_data")
class SoftmaxBackwardDataXLAImpl(UnaryOpImplementation):
    """XLA implementation of softmax backward operation."""

    def __init__(self):
        # Create kernel for softmax backward operation
        def softmax_backward_computation(grad_output, output, dim, input_dtype):
            # Compute gradient
            # grad_input = output * (grad_output - sum(grad_output * output, dim))
            sum_grad = jnp.sum(grad_output * output, axis=dim)

            # Reshape sum_grad to match broadcasting
            broadcast_shape = list(grad_output.shape)
            broadcast_shape[dim] = 1
            sum_grad = sum_grad.reshape(broadcast_shape)

            # Compute gradient
            result = output * (grad_output - sum_grad)

            # Convert to input_dtype if specified
            if input_dtype is not None:
                result = result.astype(input_dtype)

            return result

        self.kernel = TorchNeuronXLAKernel(
            softmax_backward_computation, "_softmax_backward_data", static_argnums=(2, 3)
        )

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if len(args) < 3:
            return False

        grad_output = args[0]
        output = args[1]
        dim = args[2]

        # Check that dim is an integer
        if not isinstance(dim, int):
            return False

        # Tensors must be on Neuron device
        return grad_output.device.type == "neuron" and output.device.type == "neuron"

    def _execute_impl(
        self,
        grad_output: torch.Tensor,
        output: torch.Tensor,
        dim: int,
        input_dtype: torch.dtype = None,
        *,
        out=None,
    ) -> ExecutionResult:
        """Execute softmax backward operation using XLA."""
        try:
            # Use provided output tensor or create a new one
            grad_input = (
                torch.empty(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
                if out is None
                else out
            )

            # Execute kernel
            self.kernel(grad_output, output, dim, input_dtype, output=grad_input)

            return ExecutionResult(success=True, output=grad_input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
