"""Transpose-based implementation for contiguous operation.

This implementation handles cases where a non-contiguous tensor is actually
a transposed/permuted view of a contiguous tensor. Instead of using generic
element-wise copying, it uses optimized transpose kernels.
"""

import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


def compute_permutation_for_contiguous(
    shape: list[int], strides: list[int]
) -> tuple[int, ...] | None:
    """Compute permutation mapping current -> base contiguous axes using stride order.

    Intuition:
    - In a pure permutation (transpose) of a contiguous base tensor, the set of
      stride magnitudes is unchanged; only which axis gets which stride changes.
    - Sorting axes by stride magnitude (descending) reconstructs the base axis
      order (most significant dimension first). Ties can happen for size-1 dims;
      we break them deterministically.
    - Once we have a candidate base order, we verify it by recomputing the
      row-major strides for the base shape (i.e., shapes arranged in that order)
      and mapping them back to the current axes; if all strides match exactly,
      this view is a pure permutation of a contiguous layout.

    Returns:
        Tuple[int, ...]: permutation `perm` such that applying a transpose with
        `axes=perm` to a base contiguous tensor (same storage) reproduces the
        current view. Returns None if not a pure permutation.

    Convention:
        The returned `perm` follows JAX/NumPy semantics used by
        `jnp.transpose(x, axes=perm)`, i.e., the result's axis `i` comes from
        input axis `perm[i]` (new_axis_i = old_axis_perm[i]).
    """
    n = len(shape)
    if n == 0:
        return None

    # Reject negative or zero strides (broadcasting/reversals/irregular)
    # Pure permutations of a contiguous base have positive strides.
    for s in strides:
        if s <= 0:
            return None

    # Determine base axis order by sorting axes by stride magnitude (desc).
    # Tie-breakers:
    #   1) Larger logical span first: stride[i] * max(1, shape[i])
    #   2) Stable index order to keep determinism
    axes = list(range(n))

    def sort_key(i: int):
        return (-strides[i], -(strides[i] * (shape[i] if shape[i] > 0 else 1)), i)

    base_order = sorted(axes, key=sort_key)

    # Compute expected row-major strides for the base shape (shapes in base_order)
    base_shape = [shape[i] for i in base_order]
    base_row_major_strides = list(_row_major_strides(base_shape))

    # Map expected strides back to current axis indices and verify match
    pos_in_base = {axis: idx for idx, axis in enumerate(base_order)}
    for i in range(n):
        expected_stride = base_row_major_strides[pos_in_base[i]]
        if strides[i] != expected_stride:
            return None

    # Build permutation mapping current axis -> base axis index
    permutation = tuple(pos_in_base[i] for i in range(n))

    # Check if this is the identity permutation (tensor is already contiguous)
    if permutation == tuple(range(n)):
        return None

    return permutation


def _row_major_strides(shape: list[int]) -> tuple[int, ...]:
    """Compute row-major (contiguous) strides for a given shape."""
    expected_strides: list[int] = []
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        expected_strides.insert(0, running)
        running *= shape[i]
    return tuple(expected_strides)


def _prepare_transpose_inputs(
    self_tensor: torch.Tensor,
) -> tuple[tuple[int, ...] | None, torch.Tensor | None]:
    """Compute permutation and construct base contiguous view for transpose path.

    Returns a tuple `(permutation, base_view)` where:
    - `permutation` is the axes permutation needed to obtain the current view
      from a contiguous base tensor, or None if the view is not a pure
      permutation of a contiguous layout.
    - `base_view` is a contiguous-strided view over the same storage, shaped so
      that applying the transpose with `permutation` recreates the logical order
      of `self_tensor`. If `permutation` is None, `base_view` is None.
    """
    shape = list(self_tensor.shape)
    strides = list(self_tensor.stride())

    permutation = compute_permutation_for_contiguous(shape, strides)
    if permutation is None:
        return None, None

    n = len(shape)
    inverse_perm = [0] * n
    for i, j in enumerate(permutation):
        inverse_perm[j] = i

    base_shape = [shape[inverse_perm[j]] for j in range(n)]
    base_strides = _row_major_strides(base_shape)

    base_view = self_tensor.as_strided(
        size=tuple(base_shape),
        stride=tuple(base_strides),
        storage_offset=self_tensor.storage_offset(),
    )

    return permutation, base_view


class ContiguousTransposeImpl(OperationImplementation):
    """Transpose-based implementation for contiguous operation.

    This implementation has higher priority and will be tried first.
    It handles cases where the tensor is a transpose/permutation of
    a contiguous tensor by using dynamically created transpose kernels.
    """

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check if this implementation can handle the given tensor.

        This implementation can handle:
        1. Tensors that are already on the Neuron device
        2. Non-contiguous tensors that are permutations of contiguous tensors
        3. Standard contiguous memory format (not channels_last, etc.)
        """
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False

        if self_tensor.is_complex():
            return False

        # Only handle contiguous or preserve formats (validation happens at entry)
        if memory_format is not None and (
            memory_format != torch.contiguous_format and memory_format != torch.preserve_format
        ):
            return False

        # If already contiguous, we don't need to handle it
        if self_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return False

        # Check if this is a transpose/permutation case
        shape = list(self_tensor.shape)
        strides = list(self_tensor.stride())

        # Empty tensors or scalars don't need transpose
        if self_tensor.numel() == 0 or len(shape) == 0:
            return False

        # Check if we can compute a valid permutation
        permutation = compute_permutation_for_contiguous(shape, strides)
        return permutation is not None

    def _execute_impl_internal(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using transpose kernel."""
        # Compute permutation and base contiguous view over same storage
        permutation, base_view = _prepare_transpose_inputs(self_tensor)

        if permutation is None:
            return ExecutionResult(
                success=False, error_msg="Failed to compute permutation for transpose"
            )

        # Get the kernel
        kernel = self._get_kernel()

        from .shared.context import ExecutionContext

        exec_ctx = ExecutionContext(original_inputs=(base_view, permutation), original_kwargs={})

        output = kernel(base_view, permutation, context=exec_ctx)

        return output

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using transpose kernel.
        This uses compiled JAX transpose kernels to efficiently make
        the tensor contiguous without element-wise copying.
        """
        try:
            output = self._execute_impl_internal(self_tensor, memory_format=memory_format)
            # Verify the output is contiguous
            if not output.is_contiguous():
                # This shouldn't happen
                # Fall back to the next implementation
                return ExecutionResult(
                    success=False,
                    error_msg="Transpose output not contiguous",
                )
            return ExecutionResult(success=True, output=output)

        except Exception as e:
            # If anything fails, return failure so the next implementation can try
            return ExecutionResult(success=False, error_msg=f"Transpose implementation failed: {e}")

    def execute(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute the operation.

        Handles both gradient and non-gradient paths.
        """
        # Check for empty tensor first
        if self_tensor.numel() == 0:
            # Empty tensors are already contiguous - return as-is to avoid recursion
            return ExecutionResult(success=True, output=self_tensor)

        # For tensors with gradients, we need to handle autograd
        if self_tensor.requires_grad:
            try:
                output = ContiguousTransposeFunction.apply(self_tensor, memory_format, self)
                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error_msg=f"Autograd path failed for contiguous transpose: {e}",
                )

        # Non-gradient path uses _execute_impl
        return self._execute_impl(self_tensor, memory_format)


@neuron_op("aten::contiguous", priority=100)  # Higher priority than generic implementation
class ContiguousTransposeMLIRImpl(ContiguousTransposeImpl):
    """MLIR implementation for transpose-based contiguous operation."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousTransposeMLIRImpl._kernel is None:

            def transpose_fn(x, axes):
                return torch.permute(x, axes).clone()

            ContiguousTransposeMLIRImpl._kernel = TorchMlirKernel(
                transpose_fn, op_name="aten::contiguous", static_argnums=(1,)
            )

    def _get_kernel(self):
        return ContiguousTransposeMLIRImpl._kernel


class ContiguousTransposeFunction(torch.autograd.Function):
    """Autograd function for transpose-based contiguous implementation.

    Forward uses the JAX transpose kernel on a base contiguous view over the
    underlying storage. Backward is identity (since contiguous is a layout-only
    change and mathematically y == x).
    """

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousTransposeImpl):
        return impl._execute_impl_internal(self_tensor, memory_format)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of contiguous (identity) is identity
        return grad_output, None, None
