import torch

import torch_neuronx._C as _C
from torch_neuronx.python_ops.contiguous_broadcast import ContiguousBroadcastMLIRImpl
from torch_neuronx.python_ops.contiguous_slice import ContiguousSliceMLIRImpl
from torch_neuronx.python_ops.contiguous_transpose import ContiguousTransposeMLIRImpl

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation
from .nki_kernels.contiguous_generic import contiguous_generic_kernel


class ContiguousFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, memory_format):
        # Call existing implementation
        impl = ContiguousNKIImpl()
        result = impl.execute_internal(input, memory_format)
        if not result.success:
            raise RuntimeError(f"Contiguous failed: {result.error_msg}")

        output = result.output

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # For contiguous operation, the gradient can flow back unchanged
        # since making a tensor contiguous doesn't change the mathematical operation
        return grad_output, None


@neuron_op("aten::contiguous", disable_dtype_autocast=True)
class ContiguousNKIImpl(OperationImplementation):
    """NKI implementation of contiguous operation"""

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        # Accept any memory_format here; execute() will validate for Neuron and raise
        # a canonical error for unsupported formats to avoid CPU fallback.
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
        try:
            # Validate memory_format for Neuron tensors and normalize preserve->contiguous
            from .cast_policy import validate_neuron_memory_format

            validate_neuron_memory_format(memory_format, op_name="aten::contiguous")

            # Default memory format if not specified; map preserve to contiguous
            if memory_format is None or memory_format == torch.preserve_format:
                memory_format = torch.contiguous_format

            # If tensor is already contiguous, return it as-is
            if self_tensor.is_contiguous(memory_format=memory_format):
                return ExecutionResult(success=True, output=self_tensor)

            # Create output tensor with contiguous layout
            output = torch.empty_like(self_tensor, memory_format=torch.contiguous_format)

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


def contiguous_internal(self_tensor, memory_format=None):
    """Internal contiguous implementation that can be called from C++ without dispatcher"""
    implementations = [
        ContiguousTransposeMLIRImpl(),
        ContiguousBroadcastMLIRImpl(),
        ContiguousSliceMLIRImpl(),
        ContiguousNKIImpl(),
    ]
    from .cast_policy import validate_neuron_memory_format

    validate_neuron_memory_format(memory_format, op_name="aten::contiguous")

    for impl in implementations:
        if impl.can_handle(self_tensor, memory_format):
            try:
                result = impl.execute(self_tensor, memory_format)
                if result.success:
                    return result.output
            except RuntimeError:
                continue

    raise RuntimeError("No contiguous implementation could handle this tensor configuration")


# Register the internal function with C++ so it can be called directly
# This needs to be done even with auto-registration
_C._set_python_contiguous_op(contiguous_internal)
