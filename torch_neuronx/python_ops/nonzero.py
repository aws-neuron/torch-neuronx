from torch_neuronx.utils import use_mlir_aten_ops

from .base import Operation


class NonzeroOp(Operation):
    """Nonzero operation"""

    def _setup_implementations(self):
        if use_mlir_aten_ops():
            from .torch_mlir.ops.nonzero import NonzeroMLIRImpl

            self._implementations.append(NonzeroMLIRImpl())
        else:
            from .xla_ops.nonzero_xla import NonzeroXLAImpl

            self._implementations.append(NonzeroXLAImpl())

    @property
    def op_name(self) -> str:
        return "nonzero"


class NonzeroStaticOp(Operation):
    """Nonzero_static operation"""

    def _setup_implementations(self):
        from .xla_ops.nonzero_xla import NonzeroStaticXLAImpl

        self._implementations.append(NonzeroStaticXLAImpl())

    @property
    def op_name(self) -> str:
        return "nonzero_static"


# Create nonzero operation instances
nonzero_static_op = NonzeroStaticOp()
nonzero_op = NonzeroOp()
