from .base import Operation


class IndexOp(Operation):
    """Provide the ATen index operation with an MLIR implementation."""

    def _setup_implementations(self):
        from .torch_mlir.ops.index import IndexMLIRImpl

        self._implementations.append(IndexMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the ATen operation name.

        Returns:
            str: ``"index"``.
        """
        return "index"


# Create index operation instance
index_op = IndexOp()
