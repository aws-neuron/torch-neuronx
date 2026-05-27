"""MLIR implementation of torch.index."""

import logging

import torch

from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.shared import ExecutionContext
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel
from torch_neuronx.python_ops.torch_mlir.ops.indexing import convert_indices, index_checking

logger = logging.getLogger(__name__)


class IndexMLIRImpl(OperationImplementation):
    """index implementation using MLIR"""

    def __init__(self):
        """Initialize the index kernel"""
        super().__init__()

        def index_fn(tensor, indices, *, out=None):
            # Handle negative index values
            normalized_indices = []
            for i, idx in enumerate(indices):
                if idx is None:
                    normalized_indices.append(idx)
                else:
                    size = tensor.shape[i]
                    normalized_indices.append(torch.where(idx < 0, idx + size, idx))

            return torch.ops.aten.index(tensor, normalized_indices)

        self.kernel = TorchMlirKernel(index_fn, "aten::index")

    def can_handle(self, tensor, indices, out=None):
        return tensor.device.type == "neuron"

    def execute(self, *args, **kwargs) -> ExecutionResult:
        # Capture original tensor before base class downcasts float64→float32,
        # so the kernel can infer correct output dtypes for upcast.
        self._original_tensor = args[0]
        return super().execute(*args, **kwargs)

    def _check_and_handle_empty(
        self, tensor: torch.Tensor, indices: list[torch.Tensor], *, out=None
    ) -> ExecutionResult | None:
        """Check for empty indices"""
        has_empty_indices = any(
            isinstance(idx, torch.Tensor) and idx.numel() == 0 for idx in indices if idx is not None
        )

        if has_empty_indices:
            return self._handle_empty_tensor(tensor, indices, out=out)
        return None

    def _handle_empty_tensor(
        self, tensor: torch.Tensor, indices: list[torch.Tensor], *, out=None
    ) -> ExecutionResult:
        """Handle empty output tensor case"""
        try:
            if out is None:
                _, _, _, _, _, outputs, _, _ = self.kernel._get_or_compile_hlo(
                    (tensor, indices), {"out": out}
                )
                result = torch.empty(outputs.shape, dtype=tensor.dtype, device=tensor.device)
            else:
                result = out

            return ExecutionResult(success=True, output=result)
        except Exception as e:
            logger.error(f"Failed to handle empty tensor case: {e}")
            return ExecutionResult(success=False, error_msg=str(e))

    def _execute_impl(
        self, tensor: torch.Tensor, indices: list[torch.Tensor], *, out=None
    ) -> ExecutionResult:
        """Execute index kernel and return selected values."""
        try:
            index_checking(tensor, indices)
            integer_indices = convert_indices(indices)

            # Handle empty indices after converting boolean indices to integer indices
            result = self._check_and_handle_empty(tensor, integer_indices)
            if result is not None:
                return result

            # Pass original inputs via context so the kernel can infer
            # original output dtypes for proper upcast (e.g. float64)
            context = ExecutionContext(
                original_inputs=(self._original_tensor, integer_indices),
                original_kwargs={},
            )
            result = self.kernel(tensor, integer_indices, context=context)
            return ExecutionResult(success=True, output=result)

        except Exception as e:
            if isinstance(e, IndexError | ValueError | TypeError):
                raise
            logger.error(f"Failed to execute index: {e}")
            return ExecutionResult(success=False, error_msg=str(e))

    def get_lower_func(self):
        """Return a lower function for C++ dispatch of index.

        Boolean→integer index conversion is done here (before tracing)
        to avoid dynamic shapes in the traced graph. The kernel's index_fn
        only handles negative-index normalization (static shapes).
        """
        kernel = self.kernel

        def index_lower(tensor, indices, **kwargs):
            original_args = kwargs.pop("original_args", None)
            original_kwargs = kwargs.pop("original_kwargs", None)
            kwargs.pop("out", None)

            # Convert boolean indices to integer BEFORE tracing
            # (same as convert_indices in _execute_impl)
            converted = []
            for idx in indices:
                if idx is not None and hasattr(idx, "dtype") and idx.dtype == torch.bool:
                    converted.extend(torch.nonzero(idx, as_tuple=True))
                else:
                    converted.append(idx)

            return kernel.lower(
                tensor,
                converted,
                original_args=original_args,
                original_kwargs=original_kwargs,
                **kwargs,
            )

        return index_lower
