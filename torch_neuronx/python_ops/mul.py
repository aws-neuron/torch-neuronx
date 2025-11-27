from .base import BinaryOperation
from .xla_ops.mul_xla import UnifiedMulXLAImpl


class MulOp(BinaryOperation):
    """Unified multiplication operation for tensor-tensor and tensor-scalar multiplication"""

    def _setup_implementations(self):
        """Setup available implementations for the multiplication operation"""
        self._implementations.append(UnifiedMulXLAImpl())

    @property
    def op_name(self) -> str:
        """Return the operation name for caching and debugging"""
        return "mul"


# Alias until we are sure unification of tensor and scalar mul is here to stay
MulScalarOp = MulOp
