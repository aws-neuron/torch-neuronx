"""Max/reduction operation with XLA implementation."""

from .reduction_op import ReductionOp
from .xla_ops.max_xla import MaxXLAImpl


class MaxOp(ReductionOp):
    """Max/reduction of a tensor with XLA implementation."""

    def __init__(self):
        super().__init__("max")

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(MaxXLAImpl())
