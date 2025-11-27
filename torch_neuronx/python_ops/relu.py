"""RELU activation operation with XLA implementation."""

from .base import UnaryOperation
from .xla_ops.relu_xla import ReluXLAImpl


class ReluOp(UnaryOperation):
    """RELU activation operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(ReluXLAImpl())

    @property
    def op_name(self) -> str:
        return "relu"
