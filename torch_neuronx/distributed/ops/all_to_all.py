from torch_neuronx.python_ops.base import Operation
from torch_neuronx.python_ops.xla_ops.all_to_all_xla import AllToAllXlaOp


class AllToAllOp(Operation):
    """All-to-all implementation that calls the XLA implementation"""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(AllToAllXlaOp())

    @property
    def op_name(self) -> str:
        return "all_to_all"
