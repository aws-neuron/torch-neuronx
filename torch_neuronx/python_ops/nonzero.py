from .base import Operation


class NonzeroOp(Operation):
    """Provide the ATen nonzero operation with an MLIR implementation."""

    def _setup_implementations(self):
        from .torch_mlir.ops.nonzero import NonzeroMLIRImpl

        self._implementations.append(NonzeroMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the ATen operation name.

        Returns:
            str: ``"nonzero"``.
        """
        return "nonzero"


# Create nonzero operation instance
nonzero_op = NonzeroOp()
