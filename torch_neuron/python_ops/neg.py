"""Element-wise negation operation with XLA implementation."""

from .base import UnaryOperation
from .xla_ops.neg_xla import NegXLAImpl


class NegOp(UnaryOperation):
    """Element-wise negation operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(NegXLAImpl())

    @property
    def op_name(self) -> str:
        return "neg"
