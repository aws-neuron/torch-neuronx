"""XLA implementation of GELU activation."""

import jax
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_unary_op
from torch_neuronx.python_ops.base import ExecutionResult, UnaryOpImplementation


@neuron_unary_op("aten::gelu")
@neuron_unary_op("aten::gelu_")
@neuron_unary_op("aten::gelu.out")
class GeluXLAImpl(UnaryOpImplementation):
    """XLA implementation of GELU activation."""

    def __init__(self):
        # We'll create kernels on demand based on approximation type
        self._kernels = {}

    def _get_kernel(self, approximate: str):
        """Get or create kernel for specific approximation type."""
        if approximate not in self._kernels:
            if approximate == "none":
                # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
                def gelu_exact(x):
                    return jax.nn.gelu(x, approximate=False)
            elif approximate == "tanh":
                # Approximate GELU using tanh
                def gelu_tanh(x):
                    return jax.nn.gelu(x, approximate=True)
            else:
                raise ValueError(f"Unknown GELU approximation type: {approximate}")

            # Create kernel with the appropriate function
            if approximate == "none":
                self._kernels[approximate] = TorchNeuronXLAKernel(gelu_exact, "gelu_exact")
            else:
                self._kernels[approximate] = TorchNeuronXLAKernel(gelu_tanh, "gelu_tanh")

        return self._kernels[approximate]

    def _handle_empty_tensor(
        self, input: torch.Tensor, *, approximate="none", out=None
    ) -> ExecutionResult:
        """Handle empty tensor for GELU - override to handle approximate parameter"""
        output = (
            torch.empty(input.shape, dtype=input.dtype, device=input.device) if out is None else out
        )
        return ExecutionResult(success=True, output=output)

    def _execute_impl(
        self, input: torch.Tensor, *, approximate="none", out=None
    ) -> ExecutionResult:
        """Execute GELU activation using XLA - only called for non-empty tensors."""
        # Validate approximation type - raise RuntimeError to match PyTorch
        # This happens outside try-except so it propagates immediately
        if approximate not in ["none", "tanh"]:
            raise RuntimeError("approximate argument must be either none or tanh.")

        try:
            # Use provided output tensor or create a new one
            output = (
                torch.empty(input.shape, dtype=input.dtype, device=input.device)
                if out is None
                else out
            )

            # Get appropriate kernel and execute
            kernel = self._get_kernel(approximate)
            kernel(input, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
