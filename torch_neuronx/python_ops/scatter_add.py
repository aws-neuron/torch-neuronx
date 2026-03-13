"""Scatter add operation with NKI, MLIR and XLA implementations"""

import torch

from .base import ExecutionResult, Operation, OperationImplementation
from .torch_mlir.ops.scatter_add import (
    ScatterAddInplaceMLIRImpl,
    ScatterAddMLIRImpl,
    ScatterAddOutMLIRImpl,
)


class ScatterAddNKIImpl(OperationImplementation):
    """NKI implementation for both scatter_add and scatter_add_ operations"""

    def can_handle(self, input, dim, index, src) -> bool:
        """Check if NKI can handle these inputs"""
        if input.ndim != 2 or src.ndim != 2:
            return False
        if dim != 0:
            return False
        if input.device.type != "neuron":
            return False
        if input.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            return False
        if index.ndim != 2:
            return False
        return index.stride(1) == 0

    def _execute_kernel(self, input, dim, index, src):
        """Execute the NKI kernel"""
        from torch_neuronx.utils import convert_for_neuron

        from .nki_kernels.scatter_add import scatter_add_kernel

        index = convert_for_neuron(index)
        index_1d = index[:, 0]
        result = scatter_add_kernel(input, dim, index_1d, src)
        return result

    def _execute_impl(self, input, dim, index, src) -> ExecutionResult:
        try:
            result = self._execute_kernel(input, dim, index, src)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    @property
    def priority(self) -> int:
        return super().priority + 10


class ScatterAddOp(Operation):
    """Scatter add operation (out-of-place)"""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddNKIImpl())
        self._implementations.append(ScatterAddMLIRImpl())

    @property
    def op_name(self) -> str:
        return "scatter_add"


class ScatterAddInplaceOp(Operation):
    """Scatter add operation (in-place)"""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddNKIImpl())
        self._implementations.append(ScatterAddInplaceMLIRImpl())

    @property
    def op_name(self) -> str:
        return "scatter_add_"


class ScatterAddOutOp(Operation):
    """Scatter add operation with out parameter (scatter_add.out)"""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddOutMLIRImpl())

    @property
    def op_name(self) -> str:
        return "scatter_add.out"


_scatter_add_op = ScatterAddOp()
_scatter_add_inplace_op = ScatterAddInplaceOp()
_scatter_add_out_op = ScatterAddOutOp()


def scatter_add_neuron(input, dim, index, src):
    """Neuron implementation of scatter_add (out-of-place)"""
    return _scatter_add_op(input, dim, index, src)


def scatter_add_inplace_neuron(input, dim, index, src):
    """Neuron implementation of scatter_add_ (in-place)"""
    return _scatter_add_inplace_op(input, dim, index, src)


def scatter_add_out_neuron(input, dim, index, src, *, out=None):
    """Neuron implementation of scatter_add.out"""
    return _scatter_add_out_op(input, dim, index, src, out=out)
