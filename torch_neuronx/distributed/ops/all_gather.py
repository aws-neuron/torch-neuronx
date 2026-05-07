from torch_neuronx.python_ops.base import Operation
from torch_neuronx.python_ops.xla_ops.all_gather_xla import AllGatherXlaOp


class AllGatherOp(Operation):
    """All reduce implementation that calls the XLA implementation"""

    def _setup_implementations(self):
        """Register available implementations."""
        self._implementations.append(AllGatherXlaOp())

    @property
    def op_name(self) -> str:
        return "all_gather"
