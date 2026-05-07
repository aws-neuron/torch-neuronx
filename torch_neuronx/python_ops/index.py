from .base import Operation


class IndexOp(Operation):
    """Index operation"""

    def _setup_implementations(self):
        from .torch_mlir.ops.index import IndexMLIRImpl

        self._implementations.append(IndexMLIRImpl())

    @property
    def op_name(self) -> str:
        return "index"


# Create index operation instance
index_op = IndexOp()
