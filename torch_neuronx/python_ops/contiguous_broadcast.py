import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


def compute_broadcast_for_contiguous(
    shape: list[int], strides: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Compute broadcast pattern for making a broadcasted tensor contiguous.

    This function identifies cases where a non-contiguous tensor can be made
    contiguous using broadcast operation.

    Args:
        shape: Current tensor shape
        strides: Current tensor strides

    Returns:
        Tuple of (base_shape, expand_dims) or None:
        - base_shape: Shape of the minimal underlying data (size-1 for broadcasted dims)
        - expand_dims: Dimensions that need to be expanded from size-1 to full size

        Returns None if this is not a valid broadcast pattern.
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
    """Broadcast-based implementation for contiguous operation.

    This implementation handles cases where the tensor has broadcasted dimensions
    (stride 0 with size > 1) and can be made contiguous using expand operations.
    """

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check if this implementation can handle the given tensor."""
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False

        if self_tensor.is_complex():
            return False

        # Only handle contiguous or preserve formats
        if memory_format is not None and (
            memory_format != torch.contiguous_format and memory_format != torch.preserve_format
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
        """Execute the operation with autograd support."""
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
    """MLIR implementation for broadcast-based contiguous operation."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousBroadcastMLIRImpl._kernel is None:

            def expand_fn(x, target_shape):
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


class ContiguousBroadcastFunction(torch.autograd.Function):
    """Autograd function for broadcast-based contiguous implementation."""

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousBroadcastImpl):
        return impl._execute_impl_internal(self_tensor, memory_format)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of contiguous (identity) is identity
        return grad_output, None, None
