"""Reduction operations (mean, sum, etc.) with XLA implementation."""

from .base import ReductionOperation


class ReductionOp(ReductionOperation):
    """Provide a named reduction operation with XLA output-shape handling.

    Args:
        op_name (str): ATen operation name exposed by the instance.
    """

    def __init__(self, op_name):
        self._op_name = op_name
        super().__init__()

    @property
    def op_name(self) -> str:
        """Return the configured ATen operation name.

        Returns:
            str: Operation name supplied during initialization.
        """
        return self._op_name
