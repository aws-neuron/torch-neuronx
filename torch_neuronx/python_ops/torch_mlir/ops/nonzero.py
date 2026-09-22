"""MLIR implementation of nonzero operations."""

import logging

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import nonzero_with_count
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

logger = logging.getLogger(__name__)


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

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of nonzero.

        The kernel traces nonzero_with_count which returns (indices, count)
        as 2 tensors. The C++ DispatchNonzeroImpl handles post-processing
        (slicing and int64 cast).

        Injects size=tensor.numel() and with_count=True since the ATen
        schema only provides (Tensor self) but the kernel needs these.
        """
        kernel = self.kernel

        def lower_func(*args, **kwargs):
            kwargs.pop("original_args", None)
            kwargs.pop("original_kwargs", None)
            kwargs.pop("out", None)  # out= handled by C++ post-processing
            tensor = args[0]
            # Pass size/with_count as kwargs. The kernel has these as
            # static_argnames so they get baked into the HLO.
            # Don't pass original_args — the C++ dispatcher's original args
            # don't include size/with_count, which confuses dtype inference.
            return kernel.lower(
                tensor,
                size=tensor.numel(),
                with_count=True,
            )

        return lower_func

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
