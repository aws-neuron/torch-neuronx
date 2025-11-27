"""TopK operation with XLA implementation."""

import torch

from .base import Operation
from .xla_ops.topk_xla import TopKXLAImpl


class TopKOp(Operation):
    """TopK operation with XLA implementation.

    Returns the k largest elements of the given input tensor along a given dimension.
    """

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(TopKXLAImpl())

    @property
    def op_name(self) -> str:
        return "topk"

    def _get_expected_output_shape(
        self, input: torch.Tensor, k: int, dim: int = -1, **kwargs
    ) -> tuple[torch.Size, torch.Size] | None:
        """Compute expected output shapes for values and indices tensors.

        Args:
            input: Input tensor
            k: Number of top elements to return
            dim: Dimension along which to find top-k elements

        Returns:
            Tuple of (values_shape, indices_shape) or None if not applicable
        """
        if not isinstance(input, torch.Tensor):
            return None

        # Normalize dimension
        if dim < 0:
            dim = input.dim() + dim

        # Create output shape by replacing the specified dimension with k
        output_shape = list(input.shape)
        output_shape[dim] = k
        output_shape = torch.Size(output_shape)

        # Both values and indices have the same shape
        return output_shape, output_shape
