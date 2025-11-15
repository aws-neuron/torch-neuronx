import logging

import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..auto_registration import neuron_binary_op
from ..base import ExecutionResult
from .arithmetic_xla import ArithmeticXLABase

logger = logging.getLogger(__name__)


@neuron_binary_op("aten::mul.Tensor")
@neuron_binary_op("aten::mul.out")
@neuron_binary_op("aten::mul.Scalar")
@neuron_binary_op("aten::mul.Scalar_out")
@neuron_binary_op("aten::mul_")
class UnifiedMulXLAImpl(ArithmeticXLABase):
    """Unified XLA implementation for multiplication (tensor-tensor and tensor-scalar) using JAX"""

    def __init__(self):
        """Initialize the multiplication kernel with JAX computation"""
        super().__init__()

        def mul_computation(x, y):
            """JAX computation for multiplication (handles both tensor and scalar)"""
            return x * y

        self.kernel = TorchNeuronXLAKernel(mul_computation, "mul")

    def _execute_impl(self, input, other, *, out=None) -> ExecutionResult:
        """Execute the multiplication operation - only called for non-empty tensors

        Args:
            input: First input (tensor or scalar)
            other: Second input (tensor or scalar)
            out: Optional output tensor to write result into

        Returns:
            ExecutionResult with the multiplication result
        """
        try:
            tensor_args = [input, other]

            for i, tensor in enumerate(tensor_args):
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "neuron":
                    tensor_args[i] = tensor.to("neuron")

            output = self._get_out_tensor(tensor_args) if out is None else out

            # Execute the kernel
            self.kernel(*tensor_args, output=output)

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            logger.error(f"Failed to execute multiplication: {e}")
            return ExecutionResult(success=False, error_msg=str(e))


# Alias until we are sure unification of tensor and scalar mul is here to stay
MulXLAImpl = UnifiedMulXLAImpl
MulScalarXLAImpl = UnifiedMulXLAImpl
