import torch

from .base import Operation
from .xla_ops.addmm_xla import AddmmXLAImpl


class AddmmOp(Operation):
    """Matrix multiply-add operation: out = beta*input + alpha*(mat1 @ mat2)"""

    def _setup_implementations(self):
        self._implementations.append(AddmmXLAImpl())

    @property
    def op_name(self) -> str:
        return "addmm"

    def _get_expected_output_shape(
        self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, **kwargs
    ) -> torch.Size | None:
        """Get expected output shape for matrix multiply-add"""
        # Validate that mat1 and mat2 are 2D tensors
        if mat1.dim() != 2:
            raise ValueError(f"Expected 2D tensor for mat1, got {mat1.dim()}D")
        if mat2.dim() != 2:
            raise ValueError(f"Expected 2D tensor for mat2, got {mat2.dim()}D")

        # Check matrix dimension compatibility
        if mat1.size(1) != mat2.size(0):
            raise ValueError(
                f"mat1 and mat2 shapes cannot be multiplied "
                f"({mat1.size(0)}x{mat1.size(1)} and {mat2.size(0)}x{mat2.size(1)})"
            )

        # Check that input is broadcastable to result shape
        result_shape = (mat1.size(0), mat2.size(1))
        try:
            # PyTorch's broadcasting rules
            torch.broadcast_shapes(input.shape, result_shape)
        except RuntimeError as e:
            # Re-raise the original RuntimeError to match PyTorch behavior
            raise e

        return torch.Size(result_shape)


# Create singleton instance
addmm_op = AddmmOp()
