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
    """Compute the axis permutation for a contiguous base tensor.

    Args:
        shape (list[int]): Shape of the current tensor view.
        strides (list[int]): Strides of the current tensor view.

    Returns:
        tuple[int, ...] or None: NumPy-style axes that reproduce the current
        view from a contiguous base, or ``None`` for an identity or invalid
        permutation.

    Note:
        The function rejects nonpositive strides. It sorts axes by descending
        stride, then logical span and axis index, and verifies the reconstructed
        row-major strides exactly. The result axis ``i`` comes from input axis
        ``permutation[i]``.
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
        """Build the deterministic descending-stride key for an axis.

        Args:
            i (int): Axis index.

        Returns:
            tuple[int, int, int]: Negated stride, negated logical span, and
            axis index.
        """
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
    """Make pure tensor permutations contiguous with transpose kernels.

    Note:
        This implementation has higher priority than the broadcast and generic
        implementations. It dynamically creates transpose kernels for views
        that are permutations of contiguous tensors.
    """

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check whether the transpose implementation supports a tensor.

        Args:
            self_tensor (torch.Tensor): Tensor view to inspect.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            bool: ``True`` for a nonempty, noncomplex Neuron device tensor that
            is a nonidentity permutation and uses a supported memory format.

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

        # If already contiguous in the requested format, nothing to do
        if self_tensor.is_contiguous(
            memory_format=memory_format if memory_format is not None else torch.contiguous_format
        ):
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
        permutation, base_view = _prepare_transpose_inputs(self_tensor)

        if permutation is None:
            return ExecutionResult(
                success=False, error_msg="Failed to compute permutation for transpose"
            )

        kernel = self._get_kernel()

        from .shared.context import ExecutionContext

        exec_ctx = ExecutionContext(original_inputs=(base_view, permutation), original_kwargs={})

        output = kernel(base_view, permutation, context=exec_ctx)

        return output

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous operation using transpose kernel.
        This uses compiled transpose kernels to efficiently make
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
        """Make a tensor contiguous through the transpose implementation.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            ExecutionResult: Successful output for empty tensors or completed
            execution, otherwise a failure with an error message.

        Note:
            Empty outputs alias ``self_tensor``. Tensors requiring gradients use
            ``ContiguousTransposeFunction``; other tensors execute the kernel directly.
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
    """Implement transpose-based contiguous conversion with an MLIR kernel."""

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Delegate tensor support checks to the transpose implementation.

        Args:
            self_tensor (torch.Tensor): Tensor view to inspect.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            bool: Whether the transpose implementation supports the tensor.

        Raises:
            RuntimeError: If a nonscalar tensor is not on the Neuron device.
        """
        return super().can_handle(self_tensor, memory_format)

    def __init__(self):
        super().__init__()
        if ContiguousTransposeMLIRImpl._kernel is None:

            def transpose_fn(x, axes):
                """Permute and clone a tensor.

                Args:
                    x (torch.Tensor): Contiguous base tensor.
                    axes (tuple[int, ...]): Axis permutation to apply.

                Returns:
                    torch.Tensor: Cloned tensor with the requested permutation.
                """
                return torch.permute(x, axes).clone()

            ContiguousTransposeMLIRImpl._kernel = TorchMlirKernel(
                transpose_fn, op_name="aten::contiguous", static_argnums=(1,)
            )

    def _get_kernel(self):
        return ContiguousTransposeMLIRImpl._kernel

    def get_lower_func(self):
        """Return the lowering adapter for the transpose kernel.

        Returns:
            Callable: Function that prepares a base view and permutation before
            invoking kernel lowering.
        """
        kernel = self._get_kernel()

        def contiguous_transpose_lower(*args, **kwargs):
            """Lower contiguous arguments through the transpose kernel.

            Args:
                *args (Any): Positional ATen arguments whose first item is the
                    input tensor.
                **kwargs (Any): Keyword ATen arguments, which are ignored.

            Returns:
                Any: Result returned by ``kernel.lower``.
            """
            self_tensor = args[0]
            permutation, base_view = _prepare_transpose_inputs(self_tensor)
            return kernel.lower(base_view, permutation)

        return contiguous_transpose_lower

    def get_infer_func(self):
        """Return the output-dtype inference adapter for the transpose kernel.

        Returns:
            Callable: Function that prepares a base view and permutation before
            invoking kernel output-dtype inference.
        """
        kernel = self._get_kernel()

        def contiguous_transpose_infer(*args, **kwargs):
            """Infer output dtypes for contiguous transpose arguments.

            Args:
                *args (Any): Positional ATen arguments whose first item is the
                    input tensor.
                **kwargs (Any): Keyword ATen arguments, which are ignored.

            Returns:
                Any: Result returned by ``kernel.infer_output_dtypes``.
            """
            self_tensor = args[0]
            permutation, base_view = _prepare_transpose_inputs(self_tensor)
            return kernel.infer_output_dtypes(base_view, permutation)

        return contiguous_transpose_infer


class ContiguousTransposeFunction(torch.autograd.Function):
    """Provide autograd for transpose-based contiguous conversion.

    Note:
        Forward applies the transpose kernel to a contiguous view over the same
        storage. Backward and forward-mode differentiation pass gradients
        through unchanged because contiguous conversion only changes layout.
    """

    @staticmethod
    def forward(ctx, self_tensor: torch.Tensor, memory_format, impl: ContiguousTransposeImpl):
        """Run the internal transpose implementation.

        Args:
            ctx (Any): Autograd context, which is unused.
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format | None): Requested format accepted for
                dispatcher compatibility but not inspected during execution.
            impl (ContiguousTransposeImpl): Implementation to execute.

        Returns:
            torch.Tensor | ExecutionResult: Contiguous transpose output, or failure
                details when no valid permutation is available.
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
