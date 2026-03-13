"""Gather operation with NKI, MLIR and XLA implementations"""

import torch

from .base import ExecutionResult, Operation, OperationImplementation
from .torch_mlir.ops.gather import GatherMLIRImpl, GatherOutMLIRImpl


class GatherNKIImpl(OperationImplementation):
    """NKI implementation of gather"""

    def can_handle(self, input, dim, index, sparse_grad=False) -> bool:
        """Check if NKI can handle these inputs"""
        if input.ndim != 2:
            return False
        if dim != 0:
            return False
        if input.device.type != "neuron":
            return False
        if input.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            return False
        if sparse_grad:
            return False
        # Only handle 2D index with repeated values
        if index.ndim != 2:
            return False
        return index.stride(1) == 0

    def _execute_impl(self, input, dim, index, sparse_grad=False) -> ExecutionResult:
        """Execute NKI gather"""
        try:
            from torch_neuronx.python_ops.cast_policy import (
                copy_cpu_to_neuron,
                copy_neuron_to_cpu,
                is_64bit,
            )
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.gather import gather_kernel

            original_input_dtype = input.dtype

            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = gather_kernel(input, dim, index_1d)
            # Cast back to 64-bit if needed
            if is_64bit(original_input_dtype):
                cpu_cast = copy_neuron_to_cpu(result, target_dtype=original_input_dtype)
                result = copy_cpu_to_neuron(cpu_cast, result.device, original_input_dtype)

            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    @property
    def priority(self) -> int:
        return super().priority + 10


class GatherOp(Operation):
    """Gather operation with NKI + XLA fallback"""

    def _setup_implementations(self):
        self._implementations.append(GatherNKIImpl())
        self._implementations.append(GatherMLIRImpl())

    @property
    def op_name(self) -> str:
        return "gather"


class GatherOutOp(Operation):
    """Gather operation with out parameter (gather.out)"""

    def _setup_implementations(self):
        self._implementations.append(GatherOutMLIRImpl())

    @property
    def op_name(self) -> str:
        return "gather.out"


_gather_op = GatherOp()
_gather_out_op = GatherOutOp()


def gather_neuron(input, dim, index, sparse_grad=False):
    """Neuron implementation of gather"""
    return _gather_op(input, dim, index, sparse_grad=sparse_grad)


def gather_out_neuron(input, dim, index, sparse_grad=False, *, out=None):
    """Neuron implementation of gather.out"""
    return _gather_out_op(input, dim, index, sparse_grad=sparse_grad, out=out)
