"""GELU activation operation with XLA implementation."""

from .base import UnaryOperation
from .xla_ops.gelu_xla import GeluXLAImpl


class GeluOp(UnaryOperation):
    """GELU activation operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(GeluXLAImpl())

    @property
    def op_name(self) -> str:
        return "gelu"
