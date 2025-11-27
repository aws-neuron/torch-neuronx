"""Reduction operations (mean, sum, etc.) with XLA implementation."""

from .base import ReductionOperation


class ReductionOp(ReductionOperation):
    """Base class for reduction operations with XLA implementation."""

    def __init__(self, op_name):
        self._op_name = op_name
        super().__init__()

    @property
    def op_name(self) -> str:
        return self._op_name
