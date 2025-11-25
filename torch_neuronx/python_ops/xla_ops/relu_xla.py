"""XLA implementation of RELU activation."""

import jax
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation


@neuron_unary_op("aten::relu")
@neuron_unary_op("aten::relu_")
@neuron_unary_op("aten::relu.out")
class ReluXLAImpl(UnaryOpImplementation):
    """XLA implementation of RELU activation."""

    def __init__(self):
        # We'll create kernels on demand based on approximation type

        def relu_computation(x):
            return jax.nn.relu(x)

        self.kernel = TorchNeuronXLAKernel(relu_computation, "relu")

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        if len(args) != 1:
            return False

        input_tensor = args[0]

        # Tensor must be on Neuron device
        return input_tensor.device.type == "neuron"

    def _execute_impl(self, input: torch.Tensor, *, out=None) -> ExecutionResult:
        """Execute RELU activation using XLA."""
        try:
            # Use provided output tensor or create a new one
            output = (
                torch.empty(input.shape, dtype=input.dtype, device=input.device)
                if out is None
                else out
            )

            # Execute kernel
            self.kernel(input, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
