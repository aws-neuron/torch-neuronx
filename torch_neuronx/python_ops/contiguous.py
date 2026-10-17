from __future__ import annotations

import torch

import torch_neuronx._C as _C
from torch_neuronx.python_ops.contiguous_broadcast import ContiguousBroadcastMLIRImpl
from torch_neuronx.python_ops.contiguous_slice import ContiguousSliceMLIRImpl
from torch_neuronx.python_ops.contiguous_transpose import ContiguousTransposeMLIRImpl

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation
from .nki_kernels.contiguous_generic import contiguous_generic_kernel


class ContiguousFunction(torch.autograd.Function):
    """Provide autograd support for making a Neuron tensor contiguous."""

    @staticmethod
    def forward(ctx, input, memory_format):
        # Call existing implementation
        """Make an input tensor contiguous for the autograd forward pass.

        Args:
            ctx (Any): Autograd context, unused by this operation.
            input (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format | None): Requested output memory format;
                ``None`` selects contiguous format.

        Returns:
            torch.Tensor: The input itself when already contiguous, otherwise a
                contiguous output tensor.

        Raises:
            RuntimeError: If the contiguous implementation fails.
        """
        impl = ContiguousNKIImpl()
        result = impl.execute_internal(input, memory_format)
        if not result.success:
            raise RuntimeError(f"Contiguous failed: {result.error_msg}")

        output = result.output

        return output

    @staticmethod
    def jvp(ctx, grad_input, grad_memory_format):
        """Propagate the input tangent through the contiguous operation.

        Args:
            ctx (Any): Autograd context, unused by this operation.
            grad_input (torch.Tensor): Tangent associated with the input tensor.
            grad_memory_format (Any): Tangent for the non-tensor memory format, ignored.

        Returns:
            torch.Tensor: The same ``grad_input`` tensor, unchanged.
        """
        return grad_input

    @staticmethod
    def backward(ctx, grad_output):
        # For contiguous operation, the gradient can flow back unchanged
        # since making a tensor contiguous doesn't change the mathematical operation
        """Propagate the output gradient through the contiguous operation.

        Args:
            ctx (Any): Autograd context, unused by this operation.
            grad_output (torch.Tensor): Gradient of the contiguous output.

        Returns:
            tuple: The same ``grad_output`` tensor and ``None`` for the memory format.
        """
        return grad_output, None


@neuron_op("aten::contiguous", disable_dtype_autocast=True)
class ContiguousNKIImpl(OperationImplementation):
    """Implement contiguous layout conversion on a Neuron device.

    Note:
        Empty inputs are returned unchanged before memory-format validation. A nonempty
        input already contiguous in the requested format is also returned unchanged.
        Other nonempty paths allocate an output; channels-last conversion uses the
        native NKI/NRT strided-copy path.
    """

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        # Accept any memory_format here; execute() will validate for Neuron and raise
        # a canonical error for unsupported formats to avoid CPU fallback.
        """Check whether the contiguous implementation accepts the arguments.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory format;
                accepted without validation until execution. Defaults to ``None``.

        Returns:
            bool: Whether the base Neuron device requirements are satisfied.

        Raises:
            RuntimeError: If a non-scalar tensor is not on a Neuron device.
        """
        return super().can_handle(self_tensor, memory_format=memory_format)

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        # If requires_grad, use autograd function
        if self_tensor.requires_grad:
            return ExecutionResult(
                success=True, output=ContiguousFunction.apply(self_tensor, memory_format)
            )

        # Non-grad path uses execute_internal
        return self.execute_internal(self_tensor, memory_format)

    def execute_internal(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Execute contiguous layout conversion and capture failures.

        Args:
            self_tensor (torch.Tensor): Tensor to make contiguous.
            memory_format (torch.memory_format, optional): Requested memory
                format. Defaults to ``None``.

        Returns:
            ExecutionResult: The input itself when already contiguous, a newly
            allocated converted tensor, or a captured error message.
        """
        try:
            # Validate memory_format for Neuron tensors and normalize preserve->contiguous
            from .cast_policy import validate_neuron_memory_format

            validate_neuron_memory_format(memory_format, op_name="aten::contiguous")

            # Default memory format if not specified; map preserve to contiguous
            if memory_format is None or memory_format == torch.preserve_format:
                memory_format = torch.contiguous_format

            # If tensor is already contiguous in the requested format, return as-is
            if self_tensor.is_contiguous(memory_format=memory_format):
                return ExecutionResult(success=True, output=self_tensor)

            # For channels_last / channels_last_3d: use NRT strided copy
            # to directly rearrange elements into the target layout.
            if memory_format in (torch.channels_last, torch.channels_last_3d):
                return self._make_channels_last(self_tensor, memory_format)

            # Create output tensor with the requested memory layout
            output = torch.empty_like(self_tensor, memory_format=memory_format)

            # Always use generic kernel for now
            # Create 1D views of the full storage
            src_storage_size = self_tensor.untyped_storage().size() // self_tensor.element_size()
            src_flat = self_tensor.as_strided((src_storage_size,), (1,), 0)
            dst_flat = output.view(-1)

            # Use original shape and strides for generic kernel
            original_shape = tuple(self_tensor.shape)
            original_strides = tuple(self_tensor.stride())
            output_strides = tuple(output.stride())

            contiguous_generic_kernel(
                src=src_flat,
                dst=dst_flat,
                shape=original_shape,
                src_strides=original_strides,
                dst_strides=output_strides,
                src_storage_offset=self_tensor.storage_offset(),
            )

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def _handle_empty_tensor(self, self_tensor, memory_format=None) -> ExecutionResult:
        """Empty tensors are already contiguous; return as-is."""
        return ExecutionResult(success=True, output=self_tensor)

    # -- channels_last helpers ------------------------------------------------

    def _make_channels_last(
        self, self_tensor: torch.Tensor, memory_format: torch.memory_format
    ) -> ExecutionResult:
        """Reorder a tensor to channels_last layout on-device.

        Uses NRT strided copy to directly copy elements from the source layout
        into a channels_last-strided output in a single pass, with no
        intermediate allocations or MLIR compilation.

        TODO(perf): The current NRT strided copy implementation may issue O(numel)
        individual copy calls for non-trivial stride patterns (e.g., NCHW->NHWC
        where innermost strides differ). For large tensors, consider implementing
        an NKI kernel that uses tiled DMA and SPMD sharding for better performance.
        """
        src = self_tensor
        # Materialize broadcast sources (stride=0) to avoid O(numel) NRT copy calls.
        # The strided copy recurses element-by-element when source strides are 0.
        if any(s == 0 for s in src.stride()):
            src = src.clone()

        # Compute channels_last strides explicitly
        shape = self_tensor.shape
        strides = self._channels_last_strides(shape, memory_format)
        output = torch.empty_strided(
            shape,
            strides,
            dtype=self_tensor.dtype,
            device=self_tensor.device,
        )
        _C._nki_copy_strided_neuron_to_neuron(src, output)
        return ExecutionResult(success=True, output=output)

    def _channels_last_strides(self, shape, memory_format):
        """Compute strides for channels_last or channels_last_3d format."""
        if memory_format == torch.channels_last and len(shape) == 4:
            # NCHW -> NHWC: strides (H*W*C, 1, W*C, C)
            n, c, h, w = shape
            return (h * w * c, 1, w * c, c)
        elif memory_format == torch.channels_last_3d and len(shape) == 5:
            # NCDHW -> NDHWC: strides (D*H*W*C, 1, H*W*C, W*C, C)
            n, c, d, h, w = shape
            return (d * h * w * c, 1, h * w * c, w * c, c)
        else:
            raise ValueError(f"Unsupported memory_format {memory_format} for shape {shape}")


@torch._dynamo.disable
def contiguous_internal(self_tensor, memory_format=None):
    """Make a tensor contiguous without dispatcher or Dynamo recursion.

    Args:
        self_tensor (torch.Tensor): Tensor to make contiguous.
        memory_format (torch.memory_format, optional): Requested memory
            format. Defaults to ``None``.

    Returns:
        torch.Tensor: The input tensor or a contiguous copy in the requested format.

    Raises:
        RuntimeError: If the memory format is unsupported, a nonscalar input is not
            on a Neuron device, or no implementation succeeds.

    Note:
        This C++/pybind11 entry point explicitly clears the Dynamo evaluation frame
        because the decorator may be bypassed; tracing it can corrupt side-effect-only
        ``torch.compile`` subgraphs.
    """
    prior = torch._dynamo.eval_frame.set_eval_frame(None)
    try:
        # Explicitly disable dynamo tracing — the decorator may not take effect when
        # called from C++ via pybind11, because pybind11 may resolve the underlying
        # unwrapped function.
        from .cast_policy import validate_neuron_memory_format

        validate_neuron_memory_format(memory_format, op_name="aten::contiguous")
        return _contiguous_internal_impl(self_tensor, memory_format)
    finally:
        torch._dynamo.eval_frame.set_eval_frame(prior)


def _contiguous_internal_impl(self_tensor, memory_format=None):
    """Implementation of contiguous_internal, runs with dynamo disabled."""
    implementations = [
        ContiguousTransposeMLIRImpl(),
        ContiguousBroadcastMLIRImpl(),
        ContiguousSliceMLIRImpl(),
        ContiguousNKIImpl(),
    ]

    for impl in implementations:
        if impl.can_handle(self_tensor, memory_format):
            try:
                result = impl.execute(self_tensor, memory_format)
                if result.success:
                    return result.output
            except RuntimeError:
                continue

    raise RuntimeError("No contiguous implementation could handle this tensor configuration")
