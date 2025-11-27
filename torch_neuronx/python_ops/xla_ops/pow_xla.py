"""XLA implementation of element-wise power."""

import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_binary_op
from torch_neuronx.python_ops.base import BinaryOpImplementation, ExecutionResult


@neuron_binary_op("aten::pow.Tensor_Scalar_out")
@neuron_binary_op("aten::pow.Tensor_Tensor_out")
@neuron_binary_op("aten::pow.Scalar_out")
class PowXLAImpl(BinaryOpImplementation):
    """XLA implementation of element-wise power."""

    def __init__(self):
        # Define JAX computation
        def pow_computation(x1, x2):
            return jnp.power(x1, x2)

        self.kernel = TorchNeuronXLAKernel(pow_computation, "pow")
        self.kernel_tensor_scalar = TorchNeuronXLAKernel(
            pow_computation, "pow", static_argnums=(1,)
        )

    def _execute_impl(
        self, input: torch.Tensor, exponent: torch.Tensor | float, out=None
    ) -> ExecutionResult:
        """Execute element-wise power using XLA."""
        try:
            # Use provided output tensor or create a new one
            output = (
                torch.empty(input.shape, dtype=input.dtype, device=input.device)
                if out is None
                else out
            )

            # Execute kernel
            exp_is_tensor = isinstance(exponent, torch.Tensor)
            if not exp_is_tensor:
                self.kernel_tensor_scalar(input, exponent, output=output)
            else:
                self.kernel(input, exponent, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
