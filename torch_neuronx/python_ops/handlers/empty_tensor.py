"""Very basic class to handle empty tensor operations.."""

import logging

import torch

logger = logging.getLogger(__name__)


class BaseEmptyTensorHandler:
    """Detect empty tensors in operation arguments.

    Note:
        This handler does not implement every API from the legacy empty-tensor handler.
    """

    def check_for_empty(self, *args) -> bool:
        """Check recursively for empty tensor arguments.

        Args:
            *args (object): Values to inspect; lists and tuples are traversed recursively.

        Returns:
            bool: ``True`` if any tensor has zero elements; otherwise ``False``.
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.numel() == 0:
                    return True
            elif isinstance(arg, list | tuple) and self.check_for_empty(*arg):
                return True
        return False
