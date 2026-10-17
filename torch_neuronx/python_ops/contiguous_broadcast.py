import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


def compute_broadcast_for_contiguous(
    shape: list[int], strides: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Compute a broadcast pattern for contiguous conversion.

    Args:
        shape (list[int]): Shape of the current tensor view.
        strides (list[int]): Strides of the current tensor view.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]] or None: Base shape with
        broadcast dimensions reduced to size one and the broadcast dimension
        indices, or ``None`` when no valid broadcast pattern exists.

    Note:
        A valid pattern has at least one size-greater-than-one dimension with
        zero stride and has no negative strides.
    """
    n = len(shape)
    if n == 0:
        return None

    # Identify broadcasted dimensions (stride = 0, size > 1)
    broadcasted_dims = []
    for i in range(n):
        if strides[i] < 0:
            return None  # Reject negative strides
        elif strides[i] == 0 and shape[i] > 1:
            broadcasted_dims.append(i)

    if len(broadcasted_dims) == 0:
        return None

    # Construct base shape with size-1 for broadcasted dims
    base_shape = list(shape)
    for dim in broadcasted_dims:
        base_shape[dim] = 1

    return tuple(base_shape), tuple(broadcasted_dims)


def _prepare_broadcast_inputs(
    self_tensor: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor] | None:
    """Prepare inputs for broadcast-based contiguous operation.

    Returns:
        Tuple of (expand_dims, base_view) or None:
        - expand_dims: Dimensions that need to be expanded
        - base_view: Base tensor view that can be expanded to target shape
    """
    shape = list(self_tensor.shape)
    strides = list(self_tensor.stride())

    result = compute_broadcast_for_contiguous(shape, strides)
    if result is None:
        return None

    base_shape, expand_dims = result

    # Create base view with size-1 for broadcasted dimensions
    base_view = self_tensor.as_strided(
        size=tuple(base_shape),
        stride=tuple(strides),  # Keep same strides, just change shape
        storage_offset=self_tensor.storage_offset(),
    )

    return expand_dims, base_view


class ContiguousBroadcastImpl(OperationImplementation):
    """Make broadcast tensor views contiguous with a repeat kernel.

    Note:
        This implementation handles dimensions with zero stride and size greater
        than one by viewing them at size one and expanding them to the target
        shape.
    """

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check whether the broadcast implementation supports a tensor.

        Args:
            self_tensor (torch.Tensor): Tensor view to inspect.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            bool: ``True`` for a nonempty, noncomplex Neuron device tensor with
            a valid broadcast pattern and supported memory format.

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

        # If already contiguous, we don't need to handle it
        if self_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return False

        # Check if this is a broadcast case
        shape = list(self_tensor.shape)
        strides = list(self_tensor.stride())

        if self_tensor.numel() == 0 or len(shape) == 0:
            return False

        # Check if we can handle this broadcast pattern
        result = compute_broadcast_for_contiguous(shape, strides)
        return result is not None

    def _execute_impl_internal(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using broadcast kernel."""
        expand_result = _prepare_broadcast_inputs(self_tensor)
        if expand_result is None:
            return ExecutionResult(success=False, error_msg="Failed to compute broadcast inputs")

        expand_dims, base_view = expand_result

        kernel = self._get_kernel()
        target_shape = tuple(self_tensor.shape)

        from .shared.context import ExecutionContext

        exec_ctx = ExecutionContext(original_inputs=(base_view, target_shape), original_kwargs={})

        output = kernel(base_view, target_shape, context=exec_ctx)

        return output

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using broadcast kernel."""
        try:
            output = self._execute_impl_internal(self_tensor, memory_format=memory_format)
            if not output.is_contiguous():
                return ExecutionResult(
                    success=False,
                    error_msg="Broadcast output not contiguous",
                )

            return ExecutionResult(success=True, output=output)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=f"Broadcast implementation failed: {e}")

    def execute(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Make a tensor contiguous through the broadcast implementation.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            ExecutionResult: Successful result containing the original tensor object for
            an empty input or a new contiguous tensor otherwise, or a failure.

        Note:
            Tensors requiring gradients use ``ContiguousBroadcastFunction``;
            other tensors execute the broadcast kernel directly.
        """
        if self_tensor.numel() == 0:
            return ExecutionResult(success=True, output=self_tensor)

        if self_tensor.requires_grad:
            try:
                output = ContiguousBroadcastFunction.apply(self_tensor, memory_format, self)
                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error_msg=f"Autograd path failed for contiguous broadcast: {e}",
                )

        return self._execute_impl(self_tensor, memory_format)


@neuron_op("aten::contiguous", priority=95)  # lower priority than ContiguousTransposeImpl
class ContiguousBroadcastMLIRImpl(ContiguousBroadcastImpl):
    """Implement broadcast-based contiguous conversion with an MLIR repeat kernel."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Delegate tensor support checks to the broadcast implementation.

        Args:
            self_tensor (torch.Tensor): Tensor view to inspect.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            bool: Whether the broadcast implementation supports the tensor.

        Raises:
            RuntimeError: If a nonscalar tensor is not on the Neuron device.
        """
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousBroadcastMLIRImpl._kernel is None:

            def expand_fn(x, target_shape):
                """Repeat size-one dimensions to a target shape.

                Args:
                    x (torch.Tensor): Base tensor with size-one broadcast dimensions.
                    target_shape (tuple[int, ...]): Desired output shape.

                Returns:
                    torch.Tensor: Tensor repeated along expanded dimensions.
                """
                repeat_factors = []
                for _, (src_size, tgt_size) in enumerate(zip(x.shape, target_shape, strict=False)):
                    if src_size == 1 and tgt_size > 1:
                        repeat_factors.append(tgt_size)
                    else:
                        repeat_factors.append(1)
                return x.repeat(*repeat_factors)

            ContiguousBroadcastMLIRImpl._kernel = TorchMlirKernel(
                expand_fn, op_name="aten::contiguous", static_argnums=(1,)
            )

    def _get_kernel(self):
        return ContiguousBroadcastMLIRImpl._kernel

    def get_lower_func(self):
        """Return the lowering adapter for the broadcast kernel.

        Returns:
            Callable: Function that prepares a base view and target shape before
            invoking kernel lowering.
        """
        kernel = self._get_kernel()

        def contiguous_broadcast_lower(*args, **kwargs):
            """Lower contiguous arguments through the broadcast kernel.

            Args:
                *args (Any): Positional ATen arguments whose first item is the
                    input tensor.
                **kwargs (Any): Keyword ATen arguments, which are ignored.

            Returns:
                LoweringResult: Result returned by ``kernel.lower``.
            """
            self_tensor = args[0]
            expand_dims, base_view = _prepare_broadcast_inputs(self_tensor)
            target_shape = tuple(self_tensor.shape)
            return kernel.lower(base_view, target_shape)

        return contiguous_broadcast_lower

    def get_infer_func(self):
        """Return the output-dtype inference adapter for the broadcast kernel.

        Returns:
            Callable: Function that prepares a base view and target shape before
            invoking kernel output-dtype inference.
        """
        kernel = self._get_kernel()

        def contiguous_broadcast_infer(*args, **kwargs):
            """Infer output dtypes for contiguous broadcast arguments.

            Args:
                *args (Any): Positional ATen arguments whose first item is the
                    input tensor.
                **kwargs (Any): Keyword ATen arguments, which are ignored.

            Returns:
                list[torch.dtype | None]: Output dtype metadata returned by
                ``kernel.infer_output_dtypes``.
            """
            self_tensor = args[0]
            expand_dims, base_view = _prepare_broadcast_inputs(self_tensor)
            target_shape = tuple(self_tensor.shape)
            return kernel.infer_output_dtypes(base_view, target_shape)

        return contiguous_broadcast_infer


class ContiguousBroadcastFunction(torch.autograd.Function):
    """Provide autograd for broadcast-based contiguous conversion.

    Note:
        Forward applies the broadcast kernel to a size-one base view. Backward
        and forward-mode differentiation pass gradients through unchanged because
        contiguous conversion only changes layout.
    """

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousBroadcastImpl):
        """Run the internal broadcast implementation.

        Args:
            ctx (Any): Autograd context, which is unused.
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format | None): Requested memory format.
            impl (ContiguousBroadcastImpl): Implementation to execute.

        Returns:
            torch.Tensor | ExecutionResult: Contiguous broadcast output, or failure details
                when no valid broadcast pattern is detected.
        """
        return impl._execute_impl_internal(self_tensor, memory_format)

    @staticmethod
    def jvp(ctx, grad_self, grad_memory_format, grad_impl):
        """Pass the input tangent through unchanged.

        Args:
            ctx (Any): Autograd context, which is unused.
            grad_self (torch.Tensor): Tangent for the input tensor.
            grad_memory_format (Any): Tangent for the memory format, which is ignored.
            grad_impl (Any): Tangent for the implementation, which is ignored.

        Returns:
            torch.Tensor: The unchanged input tangent.
        """
        return grad_self

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of contiguous (identity) is identity
        """Pass the output gradient to the input tensor.

        Args:
            ctx (Any): Autograd context, which is unused.
            grad_output (torch.Tensor): Gradient of the contiguous output.

        Returns:
            tuple[torch.Tensor, None, None]: Input gradient followed by no
            gradients for the memory format and implementation.
        """
        return grad_output, None, None
