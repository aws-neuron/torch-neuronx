"""NLL loss forward and backward operation with XLA implementation."""

from .base import Operation
from .xla_ops.nll_loss_xla import NLLLossBackwardXLAImpl, NLLLossForwardXLAImpl


class NLLLossForwardOp(Operation):
    """NLL loss forward with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(NLLLossForwardXLAImpl())

    @property
    def op_name(self) -> str:
        return "nll_loss_forward"


class NLLLossBackwardOp(Operation):
    """NLL loss backawrd with XLA implementation."""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(NLLLossBackwardXLAImpl())

    @property
    def op_name(self) -> str:
        return "nll_loss_backward"
