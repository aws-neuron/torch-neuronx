from .base import Operation


class NonzeroOp(Operation):
    """Nonzero operation"""

    def _setup_implementations(self):
        from .torch_mlir.ops.nonzero import NonzeroMLIRImpl

        self._implementations.append(NonzeroMLIRImpl())

    @property
    def op_name(self) -> str:
        return "nonzero"


# Create nonzero operation instance
nonzero_op = NonzeroOp()
