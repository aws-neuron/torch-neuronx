from torch_neuronx.python_ops.base import Operation
from torch_neuronx.python_ops.xla_ops.all_reduce_xla import AllReduceXLAOp


class AllReduceOp(Operation):
    """All reduce implementation that calls the XLA implementation"""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(AllReduceXLAOp())

    @property
    def op_name(self) -> str:
        return "all_reduce"
