"""XLA implementation of element-wise subtraction."""

import logging

import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.kernels.type_promotion import promote_binary_op
from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult

from .arithmetic_xla import ArithmeticXLABase

logger = logging.getLogger(__name__)


@neuron_op("aten::sub.Tensor")
@neuron_op("aten::sub.out")
@neuron_op("aten::sub_.Tensor")
@neuron_op("aten::sub.Scalar")
@neuron_op("aten::sub.Scalar_out")
class SubXLAImpl(ArithmeticXLABase):
    """Unified XLA implementation for subtraction (tensor-tensor and tensor-scalar)."""

    def __init__(self):
        # Kernel for default alpha=1 (optimized with constant)
        def sub_computation_default(x, y):
            # Handle type promotion using centralized utility
            x, y = promote_binary_op(x, y)
            return x - y

        self.kernel_default = TorchNeuronXLAKernel(sub_computation_default, "sub_default")

        # Kernel for custom alpha values (runtime parameter)
        def sub_computation_custom(x, y, alpha):
            # Handle type promotion using centralized utility
            x, y = promote_binary_op(x, y)
            return x - alpha * y

        self.kernel_custom = TorchNeuronXLAKernel(sub_computation_custom, "sub_alpha")

    def _execute_impl(
        self,
        input: torch.Tensor | int | float,
        other: torch.Tensor | int | float,
        *,
        alpha=1,
        out=None,
    ) -> ExecutionResult:
        """Execute element-wise subtraction using XLA - only called for non-empty tensors."""
        try:
            tensor_args = [input, other]

            for i, tensor in enumerate(tensor_args):
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "neuron":
                    tensor_args[i] = tensor.to("neuron")

            # Use provided output tensor or create a new one
            output = self._get_out_tensor(tensor_args) if out is None else out

            # Decide which kernel to use based on alpha
            # Check tensor type first to avoid costly tensor comparisons on device
            if isinstance(alpha, torch.Tensor):
                # Alpha is a tensor - always use the custom kernel
                # Cannot optimize this case as checking tensor value would be costly
                # Ensure tensor is on the right device and dtype
                if alpha.device != output.device or alpha.dtype != output.dtype:
                    alpha_tensor = alpha.to(device="neuron", dtype=output.dtype)
                else:
                    alpha_tensor = alpha
                # Execute kernel with alpha as a tensor parameter
                self.kernel_custom(*tensor_args, alpha_tensor, output=output)
            elif alpha == 1:
                # Alpha is a scalar with value 1 - use optimized kernel
                self.kernel_default(*tensor_args, output=output)
            else:
                # Alpha is a scalar with non-1 value - use custom kernel
                # Convert scalar to tensor on the same device as output tensor
                alpha_tensor = torch.tensor(alpha, dtype=output.dtype, device=output.device)
                # Execute kernel with alpha as a tensor parameter
                self.kernel_custom(*tensor_args, alpha_tensor, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            logger.error(f"Failed to execute subtraction: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
