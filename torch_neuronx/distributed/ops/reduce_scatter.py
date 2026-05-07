from torch_neuronx.python_ops.base import Operation
from torch_neuronx.python_ops.xla_ops.reduce_scatter_xla import ReduceScatterXLAOp


class ReduceScatterOp(Operation):
    """Reduce Scatter implementation that calls the XLA implementation"""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(ReduceScatterXLAOp())

    @property
    def op_name(self) -> str:
        return "reduce_scatter"
