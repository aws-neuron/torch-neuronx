"""MLIR implementation of torch.index."""

import logging

import torch

from torch_neuronx.python_ops import op_error_handlers
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.shared import ExecutionContext
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel
from torch_neuronx.python_ops.torch_mlir.ops.indexing import (
    _device_bounds_check,
    convert_indices,
    index_checking,
)
from torch_neuronx.utils import skip_op_preconditions

logger = logging.getLogger(__name__)


def _index_error_handler(ctx):
    """Raise aten::index's exact out-of-bounds exception on an OOB device fault
    (see op_error_handlers.py). Args: (tensor, indices); only the index tensors
    are copied to CPU, sizes come from metadata. Window [-size, size)."""
    shape = ctx.shape(0)
    indices = ctx.tensor_list(1)
    for dim, idx in enumerate(indices):
        if idx is None or not isinstance(idx, torch.Tensor) or idx.dtype == torch.bool:
            continue
        size = shape[dim]
        if bool(torch.any(idx >= size)) or bool(torch.any(idx < -size)):
            raise IndexError(
                f"index {torch.max(idx).item()} is out of bounds "
                f"for dimension {dim} with size {size}"
            )


op_error_handlers.register("aten::index", _index_error_handler)


class IndexMLIRImpl(OperationImplementation):
    """Implement advanced tensor indexing with Torch-MLIR."""

    def __init__(self):
        """Initialize the index kernel"""
        super().__init__()

        def index_fn(tensor, indices, *, out=None):
            # Handle negative index values
            """Select tensor values with normalized advanced indices.

            Args:
                tensor (torch.Tensor): Source tensor.
                indices (list[torch.Tensor | None]): Integer index tensors and omitted dimensions.
                out (torch.Tensor, optional): Unused out-variant placeholder.
                    Defaults to ``None``.

            Returns:
                torch.Tensor: Values selected by the normalized indices.

            Raises:
                IndexError: If an index is outside its dimension's valid negative or positive range.

            Note:
                Performs device bounds checks unless preconditions are disabled and wraps negative
                indices. Boolean indices are expanded before this function is called.
            """
            normalized_indices = []
            for i, idx in enumerate(indices):
                if idx is None:
                    normalized_indices.append(idx)
                else:
                    size = tensor.shape[i]
                    # In-graph bounds check (no host sync); window [-size, size)
                    # matches the previous host/C++ check. Consume the checked
                    # idx (feeds aten.index) so it survives DCE; an OOB index
                    # faults and _index_error_handler surfaces the exception.
                    if not skip_op_preconditions():
                        idx = _device_bounds_check(tensor, i, idx, wrap=True)
                    normalized_indices.append(torch.where(idx < 0, idx + size, idx))

            return torch.ops.aten.index(tensor, normalized_indices)

        self.kernel = TorchMlirKernel(index_fn, "aten::index")

    def can_handle(self, tensor, indices, out=None):
        """Check whether indexing can run on the Neuron device.

        Args:
            tensor (torch.Tensor): Source tensor to inspect.
            indices (list): Indices accepted for dispatch compatibility.
            out (torch.Tensor, optional): Output tensor accepted for dispatch
                compatibility. Defaults to ``None``.

        Returns:
            bool: ``True`` when ``tensor`` is on a Neuron device.
        """
        return tensor.device.type == "neuron"

    def execute(self, *args, **kwargs) -> ExecutionResult:
        # Capture original tensor before base class downcasts float64→float32,
        # so the kernel can infer correct output dtypes for upcast.
        """Execute indexing while preserving original dtype metadata.

        Args:
            *args (Any): Positional indexing arguments.
            **kwargs (Any): Keyword indexing arguments.

        Returns:
            ExecutionResult: Selected values or failure information.

        Raises:
            IndexError: If an index has a floating-point dtype or is out of bounds.
            TypeError: If an unsupported ``float64`` output buffer is supplied.

        Note:
            Captures the source before base-class ``float64`` downcasting so output dtype inference
                can restore it.
        """
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
        """Return the C++ dispatch lowering callable for indexing.

        Returns:
            Callable: Function that converts boolean indices before MLIR lowering.

        Note:
            Boolean-to-integer conversion occurs before tracing to avoid dynamic shapes; the traced
                kernel only normalizes negative indices.
        """
        kernel = self.kernel

        def index_lower(tensor, indices, **kwargs):
            """Convert indices and lower the indexing operation.

            Args:
                tensor (torch.Tensor): Source tensor.
                indices (list): Integer, boolean, or omitted indices by dimension.
                **kwargs (Any): Original-argument metadata forwarded to the kernel.

            Returns:
                LoweringResult: Lowered indexing operation.
            """
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
