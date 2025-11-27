"""XLA implementation of element-wise subcmul operation.

This is a private ATen op registered only for testing CPU fallback functionality.
It is not meant to be used in production code.
"""

import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_binary_op
from torch_neuronx.python_ops.base import ExecutionResult

from .arithmetic_xla import ArithmeticXLABase


@neuron_binary_op("aten::_test_serialization_subcmul")
class SubcmulXLAImpl(ArithmeticXLABase):
    """XLA implementation of element-wise subcmul operation.

    This implementation is specifically created to test the CPU fallback mechanism.
    The operation is a simple subtraction (x - y) that is registered as a private ATen op.
    The can_handle method always returns False to ensure it falls back to CPU.
    """

    def __init__(self):
        # Define JAX computation for subcmul operation
        def subcmul_computation(x, y):
            return x - y

        self.kernel = TorchNeuronXLAKernel(subcmul_computation, "subcmul")

    def can_handle(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1) -> bool:
        """Check if this implementation can handle the given inputs.

        Always returns False to ensure CPU fallback for testing purposes.
        """
        return False

    def _execute_impl(
        self, input: torch.Tensor, other: torch.Tensor, *, alpha=1
    ) -> ExecutionResult:
        """Execute element-wise subcmul operation using XLA."""
        raise RuntimeError("This op should not reach here!")
