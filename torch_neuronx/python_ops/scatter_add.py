"""Scatter add operation with NKI, MLIR and XLA implementations"""

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, Operation
from .torch_mlir.op_impl import TorchMlirOpImpl
from .torch_mlir.ops.scatter_add import (
    ScatterAddInplaceMLIRImpl,
    ScatterAddMLIRImpl,
    ScatterAddOutMLIRImpl,
)


def _scatter_add_nki_traceable(
    input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """Traceable version of NKI scatter_add for make_fx tracing."""
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    index_1d = index[:, 0]
    from .nki_kernels.scatter_add import scatter_add_kernel

    return scatter_add_kernel(input, dim, index_1d, src)


def _can_handle_scatter_add(input, dim, index, src) -> bool:
    """Shared can_handle logic for scatter_add and scatter_add_."""
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


@neuron_op("aten::scatter_add", priority=60)
class ScatterAddNKIImpl(TorchMlirOpImpl):
    """NKI implementation of scatter_add (out-of-place) — dispatched via C++ DispatchImpl."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::scatter_add",
            torch_fn=_scatter_add_nki_traceable,
            static_argnums=(1,),
        )

    def can_handle(self, input, dim, index, src) -> bool:
        return _can_handle_scatter_add(input, dim, index, src)

    def execute(self, *args, **kwargs) -> ExecutionResult:
        try:
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.scatter_add import scatter_add_kernel

            input, dim, index, src = args[0], args[1], args[2], args[3]
            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = scatter_add_kernel(input, dim, index_1d, src)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_op("aten::scatter_add_", priority=60)
class ScatterAddInplaceNKIImpl(TorchMlirOpImpl):
    """NKI implementation of scatter_add_ (inplace) — dispatched via C++ DispatchImpl."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::scatter_add_",
            torch_fn=_scatter_add_nki_traceable,
            static_argnums=(1,),
        )

    def can_handle(self, input, dim, index, src) -> bool:
        return _can_handle_scatter_add(input, dim, index, src)

    def execute(self, *args, **kwargs) -> ExecutionResult:
        try:
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.scatter_add import scatter_add_kernel

            input, dim, index, src = args[0], args[1], args[2], args[3]
            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = scatter_add_kernel(input, dim, index_1d, src)
            input.copy_(result)
            return ExecutionResult(success=True, output=input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class ScatterAddOp(Operation):
    """Scatter add operation (out-of-place)"""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddMLIRImpl())

    @property
    def op_name(self) -> str:
        return "scatter_add"


class ScatterAddInplaceOp(Operation):
    """Scatter add operation (in-place)"""

    def _setup_implementations(self):
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
