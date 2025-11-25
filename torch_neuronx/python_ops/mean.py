"""Mean reduction operation with XLA implementation."""

from .base import ReductionOperation
from .xla_ops.mean_xla import MeanXLAImpl


class MeanOp(ReductionOperation):
    """Mean reduction operation with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(MeanXLAImpl())

    @property
    def op_name(self) -> str:
        return "mean"
