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
    """Detect whether strides describe a slice along one axis.

    Args:
        shape (list[int]): Sizes of the tensor dimensions.
        strides (list[int]): Storage stride for each dimension.
        storage_offset (int): Tensor offset within its storage.

    Returns:
        tuple[int, int, int] or None: Slice axis, original axis size, and start
        index within that axis, or ``None``.
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
    """Implement contiguous copies of single-axis slices."""

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Determine whether the implementation supports a tensor.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory format. Defaults to
                ``None``.

        Returns:
            bool: ``True`` for a nonempty, noncomplex Neuron device single-axis
            slice whose reconstructed base occupies less than 4 GiB.

        Raises:
            RuntimeError: If a nonscalar tensor is not on the Neuron device.
        """
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False

        if self_tensor.is_complex():
            return False

        # Only handle contiguous_format - channels_last needs special stride handling
        if memory_format is not None and memory_format not in (
            torch.contiguous_format,
            torch.preserve_format,
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

        implied_shape = list(shape)
        implied_shape[slice_axis] = original_size
        base_strides = _row_major_strides(implied_shape)
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
        """Execute a contiguous copy with optional autograd support.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory format. Defaults to
                ``None``.

        Returns:
            ExecutionResult: Successful contiguous output or failure details. For an empty input,
                the successful output aliases ``self_tensor``.
        """
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
    """Implement slice-based contiguous copies through MLIR lowering."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Delegate slice compatibility checks to the base implementation.

        Args:
            self_tensor (torch.Tensor): Tensor to check.
            memory_format (torch.memory_format, optional): Requested memory format. Defaults to
                ``None``.

        Returns:
            bool: Whether the base slice implementation can handle the tensor.

        Raises:
            RuntimeError: If a nonscalar tensor is not on the Neuron device.
        """
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousSliceMLIRImpl._kernel is None:

            def slice_fn(x, axis, start_idx, end_idx):
                """Clone a selected range along one tensor axis.

                Args:
                    x (torch.Tensor): Source tensor.
                    axis (int): Axis to slice.
                    start_idx (int): Inclusive start index.
                    end_idx (int): Exclusive end index.

                Returns:
                    torch.Tensor: Cloned slice.
                """
                slices = [slice(None)] * x.ndim
                slices[axis] = slice(start_idx, end_idx)
                return x[tuple(slices)].clone()

            ContiguousSliceMLIRImpl._kernel = TorchMlirKernel(
                slice_fn, op_name="aten::contiguous", static_argnums=(1, 2, 3)
            )

    def _get_kernel(self):
        return ContiguousSliceMLIRImpl._kernel

    def get_lower_func(self):
        """Create an ATen-to-kernel lowering function.

        Returns:
            Callable: Function that lowers a contiguous slice to MLIR.
        """
        kernel = self._get_kernel()

        def contiguous_slice_lower(*args, **kwargs):
            """Lower a contiguous-slice call through the cached kernel.

            Args:
                *args (Any): Positional ATen arguments, beginning with the input tensor.
                **kwargs (Any): Unused keyword ATen arguments.

            Returns:
                LoweringResult: Lowered kernel result for the reconstructed base view.
            """
            self_tensor = args[0]
            shape = list(self_tensor.shape)
            strides = list(self_tensor.stride())
            slice_axis, original_size, start_offset = detect_slice_pattern(
                shape, strides, self_tensor.storage_offset()
            )
            implied_shape = list(shape)
            implied_shape[slice_axis] = original_size
            base_strides = _row_major_strides(implied_shape)
            base_storage_offset = self_tensor.storage_offset() - start_offset * strides[slice_axis]
            base_view = self_tensor.as_strided(
                size=tuple(implied_shape),
                stride=tuple(base_strides),
                storage_offset=base_storage_offset,
            )
            end_offset = start_offset + shape[slice_axis]
            return kernel.lower(base_view, slice_axis, start_offset, end_offset)

        return contiguous_slice_lower

    def get_infer_func(self):
        """Create an ATen-to-kernel dtype inference function.

        Returns:
            Callable: Function that infers contiguous-slice output dtypes.
        """
        kernel = self._get_kernel()

        def contiguous_slice_infer(*args, **kwargs):
            """Infer output dtypes through the cached slice kernel.

            Args:
                *args (Any): Positional ATen arguments, beginning with the input tensor.
                **kwargs (Any): Unused keyword ATen arguments.

            Returns:
                list[torch.dtype | None]: Output dtype metadata for the selected
                contiguous slice.
            """
            self_tensor = args[0]
            shape = list(self_tensor.shape)
            strides = list(self_tensor.stride())
            slice_axis, original_size, start_offset = detect_slice_pattern(
                shape, strides, self_tensor.storage_offset()
            )
            implied_shape = list(shape)
            implied_shape[slice_axis] = original_size
            base_strides = _row_major_strides(implied_shape)
            base_storage_offset = self_tensor.storage_offset() - start_offset * strides[slice_axis]
            base_view = self_tensor.as_strided(
                size=tuple(implied_shape),
                stride=tuple(base_strides),
                storage_offset=base_storage_offset,
            )
            end_offset = start_offset + shape[slice_axis]
            return kernel.infer_output_dtypes(base_view, slice_axis, start_offset, end_offset)

        return contiguous_slice_infer


class ContiguousSliceFunction(torch.autograd.Function):
    """Implement autograd propagation for slice-based contiguous copies."""

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousSliceImpl):
        """Run the slice implementation for the autograd forward pass.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context, which is unused.
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format | None): Requested memory format.
            impl (ContiguousSliceImpl): Slice implementation to execute.

        Returns:
            torch.Tensor | ExecutionResult: Contiguous slice output, or failure details when no
                valid slice pattern is detected.
        """
        return impl._execute_impl_internal(self_tensor, memory_format)

    @staticmethod
    def jvp(ctx, grad_self, grad_memory_format, grad_impl):
        """Propagate the tensor tangent through the contiguous copy.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context, which is unused.
            grad_self (torch.Tensor): Tangent for the input tensor.
            grad_memory_format (Any): Memory-format tangent, which is ignored.
            grad_impl (Any): Implementation tangent, which is ignored.

        Returns:
            torch.Tensor: Unchanged input tangent.
        """
        return grad_self

    @staticmethod
    def backward(ctx, grad_output):
        """Propagate the output gradient to the input tensor.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context, which is unused.
            grad_output (torch.Tensor): Gradient of the contiguous output.

        Returns:
            tuple[torch.Tensor, None, None]: ``grad_output`` itself and placeholders for
                nondifferentiable inputs.
        """
        return grad_output, None, None
