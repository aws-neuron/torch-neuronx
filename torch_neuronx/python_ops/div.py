from .base import BinaryOperation
from .xla_ops.div_xla import DivXLAImpl


class DivOp(BinaryOperation):
    """Division operation for element-wise tensor division"""

    def _setup_implementations(self):
        """Setup available implementations for the division operation"""
        self._implementations.append(DivXLAImpl())

    @property
    def op_name(self) -> str:
        """Return the operation name for caching and debugging"""
        return "div"


DivScalarOp = DivOp
