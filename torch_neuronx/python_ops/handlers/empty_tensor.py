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
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.numel() == 0:
                    return True
            elif isinstance(arg, list | tuple) and self.check_for_empty(*arg):
                return True
        return False
