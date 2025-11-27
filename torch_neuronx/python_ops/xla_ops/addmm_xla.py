import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.base import ExecutionResult, MatrixOpImplementation


class AddmmXLAImpl(MatrixOpImplementation):
    """XLA implementation of matrix multiply-add using JAX"""

    def __init__(self):
        # Kernel for default alpha=1, beta=1 (optimized with constants)
        def addmm_computation_default(input, mat1, mat2):
            # Compute: input + (mat1 @ mat2) with constants
            mm_result = jnp.matmul(mat1, mat2)
            return input + mm_result

        self.kernel_default = TorchNeuronXLAKernel(addmm_computation_default, "addmm_default")

        # Kernel for custom alpha/beta values (runtime parameters)
        def addmm_computation_custom(input, mat1, mat2, beta, alpha):
            # Compute: beta * input + alpha * (mat1 @ mat2)
            mm_result = jnp.matmul(mat1, mat2)
            # Handle broadcasting for input
            return beta * input + alpha * mm_result

        self.kernel_custom = TorchNeuronXLAKernel(addmm_computation_custom, "addmm_alphabeta")

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        input = args[0]
        mat1 = args[1]
        mat2 = args[2]

        # All inputs must be on Neuron device
        if not (
            input.device.type == "neuron"
            and mat1.device.type == "neuron"
            and mat2.device.type == "neuron"
        ):
            return False

        # mat1 and mat2 must be 2D
        return mat1.dim() == 2 and mat2.dim() == 2

    def _compute_output_shape(self, input, mat1, mat2) -> torch.Size:
        """Compute output shape for addmm operation"""
        return torch.Size([mat1.size(0), mat2.size(1)])

    def _execute_impl(
        self,
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta: float = 1,
        alpha: float = 1,
        out: torch.Tensor | None = None,
    ) -> ExecutionResult:
        """Execute matrix multiply-add on Neuron device - only called for non-empty tensors"""
        # Create output tensor if not provided
        if out is None:
            output = torch.empty(mat1.size(0), mat2.size(1), dtype=input.dtype, device=input.device)
        else:
            output = out

        # Check if using default alpha and beta values
        if (
            alpha == 1
            and beta == 1
            and not isinstance(alpha, torch.Tensor)
            and not isinstance(beta, torch.Tensor)
        ):
            # Use optimized kernel with constants
            self.kernel_default(input, mat1, mat2, output=output)
        else:
            # Handle beta parameter - can be scalar or tensor
            if isinstance(beta, torch.Tensor):
                # If already a tensor, ensure it's on the right device and dtype
                if beta.device != input.device or beta.dtype != input.dtype:
                    beta_tensor = beta.to(device=input.device, dtype=input.dtype)
                else:
                    beta_tensor = beta
            else:
                # Convert scalar to tensor on the same device as input
                beta_tensor = torch.tensor(beta, dtype=input.dtype, device=input.device)

            # Handle alpha parameter - can be scalar or tensor
            if isinstance(alpha, torch.Tensor):
                # If already a tensor, ensure it's on the right device and dtype
                if alpha.device != input.device or alpha.dtype != input.dtype:
                    alpha_tensor = alpha.to(device=input.device, dtype=input.dtype)
                else:
                    alpha_tensor = alpha
            else:
                # Convert scalar to tensor on the same device as input
                alpha_tensor = torch.tensor(alpha, dtype=input.dtype, device=input.device)

            # Execute the kernel with runtime parameters
            self.kernel_custom(input, mat1, mat2, beta_tensor, alpha_tensor, output=output)

        return ExecutionResult(success=True, output=output)
