"""Very basic class to handle empty tensor operations.."""

import logging

import torch

logger = logging.getLogger(__name__)


class BaseEmptyTensorHandler:
    """Handles operations on empty tensors.
    Note: Not all the APIs are ported over from jax's empty tensor handler.
    """

    def check_for_empty(self, *args) -> bool:
        """Check if any tensor arguments are empty.

        Args:
            *args: Arguments to check

        Returns:
            True if any tensor is empty
        """
        from torch.utils._pytree import tree_flatten

        flat_tensors, _ = tree_flatten(args)
        return any(isinstance(t, torch.Tensor) and t.numel() == 0 for t in flat_tensors)
