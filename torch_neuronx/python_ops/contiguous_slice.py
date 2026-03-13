"""slice-based implementation for contiguous operation."""

import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


def _row_major_strides(shape: list[int]) -> tuple[int, ...]:
    """Compute row-major (contiguous) strides for a given shape."""
    if not shape:
        return ()

    strides = []
    running_stride = 1

    # Build strides from right to left
    for i in range(len(shape) - 1, -1, -1):
        strides.insert(0, running_stride)
        running_stride *= shape[i]

    return tuple(strides)


def detect_slice_pattern(
    shape: list[int], strides: list[int], storage_offset: int
) -> tuple[int, int, int] | None:
    """Detect if tensor is a slice of a single axis.

    Returns (axis, original_size, start_offset) or None if not a simple single-axis slice.
    """

    # Check if storage_offset indicates slicing in multiple dimensions
    if storage_offset > 0:
        remaining_offset = storage_offset
        offset_dims = 0
        for i in range(len(strides)):
            if strides[i] > 0 and remaining_offset >= strides[i]:
                offset_dims += 1
                remaining_offset %= strides[i]

        if offset_dims > 1:
            return None  # Multi-axis slice detected

    # For slice, we need to find which axis has the "wrong" stride relationship
    # Check stride relationships: stride[i] should equal shape[i+1] * stride[i+1]
    sliced_dims = 0
    slice_axis = None
    original_size = None

    for axis in range(len(shape)):
        if axis == len(shape) - 1:
            # Last dimension: stride must be 1 (element size) - can't be sliced further
            if strides[axis] != 1:
                return None
        else:
            expected_stride = shape[axis + 1] * strides[axis + 1]
            if strides[axis + 1] == 0:
                return None

            if strides[axis] != expected_stride:
                # This dimension has wrong stride - check if it's due to slice
                # The stride should be based on a larger size for the next dimension
                if strides[axis] % strides[axis + 1] == 0:
                    implied_next_size = strides[axis] // strides[axis + 1]
                    if implied_next_size > shape[axis + 1] and _has_contiguous_strides_for_shape(
                        shape, strides, axis + 1, implied_next_size
                    ):
                        sliced_dims += 1
                        if sliced_dims > 1:
                            return None  # Multi-axis slice

                        slice_axis = axis + 1
                        original_size = implied_next_size
                        continue
                return None

    if slice_axis is not None:
        # Calculate start offset in the sliced dimension
        start_offset = (storage_offset // strides[slice_axis]) % original_size
        return slice_axis, original_size, start_offset

    return None


def _has_contiguous_strides_for_shape(
    shape: list[int], strides: list[int], slice_axis: int, original_size: int
) -> bool:
    """Check if strides are contiguous for the implied original shape."""
    # Reconstruct what the original shape would have been
    original_shape = list(shape)
    original_shape[slice_axis] = original_size

    # Check if current strides match what they would be for the original shape
    expected_original_strides = _row_major_strides(original_shape)

    return list(strides) == list(expected_original_strides)


class ContiguousSliceImpl(OperationImplementation):
    """slice-based implementation for contiguous operation."""

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check if this implementation can handle the tensor."""
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False

        if self_tensor.is_complex():
            return False

        if memory_format is not None and (
            memory_format != torch.contiguous_format and memory_format != torch.preserve_format
        ):
            return False

        if self_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return False

        if self_tensor.numel() == 0 or len(self_tensor.shape) == 0:
            return False

        # Check for slice pattern
        shape = list(self_tensor.shape)
        strides = list(self_tensor.stride())
        slice_info = detect_slice_pattern(shape, strides, self_tensor.storage_offset())

        if slice_info is None:
            return False

        # Reject if reconstructed base tensor would exceed 4GB - compiler limitation
        slice_axis, original_size, _ = slice_info
        base_numel = self_tensor.numel() * original_size // shape[slice_axis]
        return base_numel * self_tensor.element_size() < 4 * 1024**3

    def _execute_impl_internal(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using slice kernel."""
        shape = list(self_tensor.shape)
        strides = list(self_tensor.stride())
        slice_info = detect_slice_pattern(shape, strides, self_tensor.storage_offset())

        if slice_info is None:
            return ExecutionResult(success=False, error_msg="No valid slice pattern detected")

        slice_axis, original_size, start_offset = slice_info

        # Create base view with original shape and contiguous strides
        implied_shape = list(shape)
        implied_shape[slice_axis] = original_size
        base_strides = _row_major_strides(implied_shape)

        # Calculate the base storage offset (start of the original tensor)
        base_storage_offset = self_tensor.storage_offset() - start_offset * strides[slice_axis]

        base_view = self_tensor.as_strided(
            size=tuple(implied_shape),
            stride=tuple(base_strides),
            storage_offset=base_storage_offset,
        )

        end_offset = start_offset + shape[slice_axis]
        kernel = self._get_kernel()

        from .shared.context import ExecutionContext

        exec_ctx = ExecutionContext(
            original_inputs=(base_view, slice_axis, start_offset, end_offset),
            original_kwargs={},
        )

        output = kernel(base_view, slice_axis, start_offset, end_offset, context=exec_ctx)

        return output

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute the operation without autograd support"""
        try:
            output = self._execute_impl_internal(self_tensor, memory_format=memory_format)
            if not output.is_contiguous():
                return ExecutionResult(
                    success=False,
                    error_msg="Slice output not contiguous",
                )

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=f"Slice implementation failed: {e}")

    def execute(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute the operation with autograd support."""
        if self_tensor.numel() == 0:
            return ExecutionResult(success=True, output=self_tensor)

        if self_tensor.requires_grad:
            try:
                output = ContiguousSliceFunction.apply(self_tensor, memory_format, self)
                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error_msg=f"Autograd path failed for contiguous slice: {e}",
                )

        return self._execute_impl(self_tensor, memory_format)


@neuron_op("aten::contiguous", priority=90)
class ContiguousSliceMLIRImpl(ContiguousSliceImpl):  # lower priority than ContiguousBroadcastImpl
    """MLIR implementation for slice-based contiguous operation."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousSliceMLIRImpl._kernel is None:

            def slice_fn(x, axis, start_idx, end_idx):
                slices = [slice(None)] * x.ndim
                slices[axis] = slice(start_idx, end_idx)
                return x[tuple(slices)].clone()

            ContiguousSliceMLIRImpl._kernel = TorchMlirKernel(
                slice_fn, op_name="aten::contiguous", static_argnums=(1, 2, 3)
            )

    def _get_kernel(self):
        return ContiguousSliceMLIRImpl._kernel


class ContiguousSliceFunction(torch.autograd.Function):
    """Autograd function for slice-based contiguous implementation."""

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousSliceImpl):
        return impl._execute_impl_internal(self_tensor, memory_format)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
