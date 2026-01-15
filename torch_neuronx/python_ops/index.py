from torch_neuronx.utils import use_mlir_aten_ops

from .base import Operation


class IndexOp(Operation):
    """Nonzero operation"""

    def _setup_implementations(self):
        if use_mlir_aten_ops():
            from .torch_mlir.ops.index import IndexMLIRImpl

            self._implementations.append(IndexMLIRImpl())
        else:
            from .xla_ops.index_xla import IndexXLAImpl

            self._implementations.append(IndexXLAImpl())

    @property
    def op_name(self) -> str:
        return "index"


# Create index operation instance
index_op = IndexOp()
