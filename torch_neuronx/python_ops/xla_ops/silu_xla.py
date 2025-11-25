"""XLA implementation of SiLU (Swish) activation."""

import jax.nn as jnn
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation


@neuron_unary_op("aten::silu")
@neuron_unary_op("aten::silu_")
@neuron_unary_op("aten::silu.out")
class SiluXLAImpl(UnaryOpImplementation):
    """XLA implementation of SiLU (Swish) activation."""

    def __init__(self):
        # Define JAX computation
        def computation(x):
            return jnn.silu(x)

        self.kernel = TorchNeuronXLAKernel(computation, "silu")

    def _execute_impl(self, input: torch.Tensor, out=None) -> ExecutionResult:
        """Execute SiLU activation using XLA."""
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
