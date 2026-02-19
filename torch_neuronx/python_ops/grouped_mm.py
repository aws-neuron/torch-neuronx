"""Grouped matrix multiplication with NKI, JAX, and MLIR implementations"""

import torch

from torch_neuronx.utils import get_gmm_align, skip_op_preconditions

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation
from .torch_mlir.op_impl import TorchMlirOpImpl


def validate_offs(offs, t, align=128):
    """Validate offs tensor against t and align constraints."""
    if skip_op_preconditions():
        return True
    if t % align != 0:
        raise ValueError(f"t must be divisible by {align}")
    if (offs < 0).any() or (offs > t).any():
        raise ValueError(f"offs must be between 0 and {t}")
    if (offs % align != 0).any():
        raise ValueError(f"offs must be divisible by {align}")
    if len(offs) > 1 and (offs[1:] < offs[:-1]).any():
        raise ValueError("offs must be sorted")
    return True


@neuron_op("aten::_grouped_mm", priority=60)
class GroupedMMNKIImpl(OperationImplementation):
    """2D x 2D: a (d1, t), b (t, d2) -> (g, d1, d2)"""

    def can_handle(self, a, b, offs) -> bool:
        return a.dim() == 2 and b.dim() == 2 and validate_offs(offs, a.shape[1])

    def _execute_impl(self, a, b, offs) -> ExecutionResult:
        try:
            from .nki_kernels.grouped_mm import grouped_mm_2d_2d_op

            return ExecutionResult(success=True, output=grouped_mm_2d_2d_op(a, b, offs))
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_op("aten::_grouped_mm", priority=40)
class GroupedMMMLIRImpl(TorchMlirOpImpl):
    """2D x 3D: a (t, d1), b (g, d1, d2) -> (t, d2) - MLIR path only"""

    def __init__(self):
        super().__init__(aten_op_name="aten::_grouped_mm", torch_fn=torch._grouped_mm)

    def can_handle(self, a, b, offs) -> bool:
        return a.dim() == 2 and b.dim() == 3 and validate_offs(offs, a.shape[0], get_gmm_align())
