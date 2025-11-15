"""Element-wise addition operation with XLA implementation."""

from .base import BinaryOperation
from .xla_ops.add_xla import AddXLAImpl


class AddOp(BinaryOperation):
    """Element-wise addition operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(AddXLAImpl())

    @property
    def op_name(self) -> str:
        return "add"
