"""Element-wise power operation with XLA implementation."""

from .base import BinaryOperation
from .xla_ops.pow_xla import PowXLAImpl


class PowOp(BinaryOperation):
    """Element-wise power operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(PowXLAImpl())

    @property
    def op_name(self) -> str:
        return "pow"
