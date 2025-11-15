"""XLA implementation of element-wise addition."""

import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult

from .arithmetic_xla import ArithmeticXLABase


@neuron_op("aten::add.Tensor")
@neuron_op("aten::add_.Tensor")
@neuron_op("aten::add.out")
class AddXLAImpl(ArithmeticXLABase):
    """XLA implementation of element-wise addition."""

    def __init__(self):
        # Kernel for default alpha=1 (optimized with constant)
        def add_computation_default(x, y):
            return x + y

        self.kernel_default = TorchNeuronXLAKernel(add_computation_default, "add_default")

        # Kernel for custom alpha values (runtime parameter)
        def add_computation_custom(x, y, alpha):
            return x + alpha * y

        self.kernel_custom = TorchNeuronXLAKernel(add_computation_custom, "add_alpha")

    def _execute_impl(
        self,
        input: torch.Tensor | int | float,
        other: torch.Tensor | int | float,
        *,
        alpha=1,
        out=None,
    ) -> ExecutionResult:
        """Execute element-wise addition using XLA."""
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
                if alpha.device.type != "neuron":
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
                # Convert scalar to tensor on the same device as input
                alpha_tensor = torch.tensor(alpha, dtype=input.dtype, device=input.device)
                # Execute kernel with alpha as a tensor parameter
                self.kernel_custom(*tensor_args, alpha_tensor, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
