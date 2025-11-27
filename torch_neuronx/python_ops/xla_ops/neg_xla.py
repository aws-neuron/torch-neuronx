"""XLA implementation of element-wise negation."""

import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation


@neuron_unary_op("aten::neg")
@neuron_unary_op("aten::neg_")
@neuron_unary_op("aten::neg.out")
class NegXLAImpl(UnaryOpImplementation):
    """XLA implementation of element-wise negation."""

    def __init__(self):
        # Define JAX computation
        def neg_computation(x):
            return -x

        self.kernel = TorchNeuronXLAKernel(neg_computation, "neg")

    def _execute_impl(self, input: torch.Tensor, *, out=None) -> ExecutionResult:
        """Execute element-wise negation using XLA - only called for non-empty tensors."""
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
