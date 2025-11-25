import logging

import torch

from ...kernels.type_promotion import promote_binary_op
from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult
from .arithmetic_xla import ArithmeticXLABase

logger = logging.getLogger(__name__)


class UnifiedDivXLAImpl(ArithmeticXLABase):
    """Unified XLA implementation for division (tensor-tensor and tensor-scalar) using JAX"""

    def __init__(self):
        """Initialize the division kernel with JAX computation"""
        super().__init__()

        def div_computation(x, y):
            """JAX computation for division (handles both tensor and scalar)"""
            x, y = promote_binary_op(x, y)
            return x / y

        self.kernel = TorchNeuronXLAKernel(div_computation, "div")

    def _execute_impl(self, input, other, *, out=None) -> ExecutionResult:
        """Execute the division operation - only called for non-empty tensors

        Args:
            input: First input (tensor or scalar)
            other: Second input (tensor or scalar)
            out: Optional output tensor to write result into

        Returns:
            ExecutionResult with the division result
        """
        try:
            tensor_args = [input, other]

            for i, tensor in enumerate(tensor_args):
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "neuron":
                    tensor_args[i] = tensor.to("neuron")

            # Use provided output tensor or create a new one
            output = self._get_out_tensor(tensor_args) if out is None else out

            # Execute the kernel
            self.kernel(*tensor_args, output=output)

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            logger.error(f"Failed to execute division: {e}")
            return ExecutionResult(success=False, error_msg=str(e))


# Alias for compatibility - both DivXLAImpl and DivScalarXLAImpl now use the unified implementation
DivXLAImpl = UnifiedDivXLAImpl
DivScalarXLAImpl = UnifiedDivXLAImpl
