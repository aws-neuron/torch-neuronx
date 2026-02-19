"""MLIR implementation of nonzero operations."""

import logging

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import nonzero_with_count
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

logger = logging.getLogger(__name__)


# TODO uncomment the following once fully migrate to MLIR
# @neuron_op("aten::nonzero")
# @neuron_op("aten::nonzero.out")
class NonzeroMLIRImpl(OperationImplementation):
    """nonzero implementation using MLIR"""

    def __init__(self):
        """Initialize the nonzero kernel"""
        super().__init__()

        self.kernel = TorchMlirKernel(
            nonzero_with_count, "aten::nonzero", static_argnames=("size", "with_count")
        )

    def can_handle(self, tensor, out=None):
        return tensor.device.type == "neuron"

    def _handle_empty_tensor(self, tensor, *, out=None) -> ExecutionResult:
        """Handle empty tensor - return empty (0, ndim) tensor."""
        result = torch.empty((0, tensor.ndim), dtype=torch.int64, device=tensor.device)
        if out is not None:
            out.resize_(result.shape)
            out.copy_(result)
        return ExecutionResult(success=True, output=result)

    def _execute_impl(self, tensor: torch.Tensor, *, out=None) -> ExecutionResult:
        """Execute nonzero kernel and return result indices."""
        # TODO how to handle out tensor with smaller size
        # pytorch will automatically resize the out tensor

        if out is not None and out.dtype != torch.int64:
            # PyTorch requires output dtype to be torch.int64
            raise RuntimeError(
                "nonzero: Expected out tensor to have scalar type Long but got scalar typeFloat"
            )

        try:
            indices, size = self.kernel(tensor, size=tensor.numel(), with_count=True)
            result = indices[:size].to(torch.int64)

            if out is not None:
                out.copy_(result)

            return ExecutionResult(success=True, output=result)

        except Exception as e:
            logger.error(f"Failed to execute nonzero: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
