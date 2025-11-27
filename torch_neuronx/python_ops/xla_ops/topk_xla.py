"""XLA implementation of TopK operation using scribe for custom calls."""

import torch

from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


@neuron_op("aten::topk")
@neuron_op("aten::topk.values")
@neuron_op("aten::topk.out")
class TopKXLAImpl(XLABuilderOpImpl):
    """XLA implementation of TopK operation using scribe-based custom calls."""

    def __init__(self):
        """Initialize TopK XLA implementation with scribe-based kernel."""
        super().__init__(
            op_name="topk",
            static_argnames=["k"],  # k is static
        )

    def hlo_fn(self, input_tensor, **kwargs):
        """Generate XLA computation for TopK operation.

        Args:
            input_tensor: Input tensor for TopK operation (or tuple containing input tensor)
            **kwargs: Keyword arguments including:
                k: Number of top elements to return

        Returns:
            XLA computation for TopK operation with AwsNeuronTopK custom call
        """
        # Handle case where input_tensor is a tuple (processed by XLABuilderCompiler)
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]

        k = kwargs["k"]
        dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(input_tensor.dtype)
        scribe = HloScribe()

        def _topk(scribe):
            # Create shape for input tensor
            dtype_shape = HloShape(scribe, dtype)
            input_shape = dtype_shape[input_tensor.shape]
            input_param = input_shape.Parameter(parameter_number=0)

            # Calculate output shapes - TopK operates on last dimension
            output_shape = list(input_tensor.shape)
            output_shape[-1] = k

            # Create shapes for outputs (values and indices)
            values_shape = dtype_shape[output_shape]
            indices_dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(torch.int32)
            indices_dtype_shape = HloShape(scribe, indices_dtype)
            indices_shape = indices_dtype_shape[output_shape]

            # Create tuple shape for the two outputs
            tuple_shape = scribe.tuple(values_shape, indices_shape)

            # Generate custom call to AwsNeuronTopK
            return tuple_shape.CustomCall(
                input_param,
                custom_call_target="AwsNeuronTopK",
                backend_config=str(k).encode("utf-8"),
            )

        return scribe(_topk)

    def can_handle(
        self,
        input: torch.Tensor,
        k: int,
        dim=-1,
        largest: bool = True,
        sorted: bool = True,
        **kwargs,
    ) -> bool:
        """Check if this implementation can handle the given inputs.

        Args:
            input: Input tensor
            k: Number of top elements to return
            dim: Dimension along which to find top-k elements
            largest: If True, return largest elements; if False, return smallest
            sorted: If True, return elements in sorted order
            **kwargs: Additional keyword arguments

        Returns:
            True if this implementation can handle the inputs, False otherwise
        """
        # Check parent class compatibility
        if not super().can_handle(input, k=k, dim=dim, largest=largest, sorted=sorted, **kwargs):
            return False

        # k must be positive integer
        if not isinstance(k, int) or k <= 0:
            return False

        # Normalize dimension
        if dim < 0:
            dim = input.dim() + dim

        # Currently only support last dimension
        if dim != input.dim() - 1:
            return False

        # Check if k is not larger than the dimension size
        if k > input.shape[dim]:
            return False

        # Currently only support largest=True
        if not largest:
            return False

        # Check supported dtypes
        return input.dtype in [torch.float32, torch.float16, torch.bfloat16]

    def _execute_impl(
        self,
        input: torch.Tensor,
        k: int,
        dim=-1,
        largest: bool = True,
        sorted: bool = True,
        *,
        out=None,
        values=None,
        indices=None,
    ):
        """Execute TopK operation using XLA custom call.

        Args:
            input: Input tensor
            k: Number of top elements to return
            dim: Dimension along which to find top-k elements (default: -1)
            largest: If True, return largest elements; if False, return smallest
            sorted: If True, return elements in sorted order
            out: Optional output tuple (values, indices)
        Returns:
            Tuple of (values, indices) tensors
        Raises:
            NotImplementedError: If largest=False (not currently supported)
        """
        # Normalize dimension
        if dim < 0:
            dim = input.dim() + dim

        # Calculate output shapes
        output_shape = list(input.shape)
        output_shape[dim] = k
        output_shape = torch.Size(output_shape)

        # Create output tensors if not provided
        if values is None and indices is None:
            values_tensor = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            indices_tensor = torch.empty(output_shape, dtype=torch.int32, device=input.device)
            output = (values_tensor, indices_tensor)
        else:
            # Handle case where values and indices are passed as separate keyword arguments
            values_tensor, indices_tensor = values, indices
            output = (values_tensor, indices_tensor)

        # Execute kernel using XLABuilderOpImpl
        result = self.kernel(input, k=k, out=output)

        # TODO: This manual int64 conversion is needed because autocast_neuron doesn't handle
        # operations that take non-int64 inputs but must return int64 outputs. The autocast
        # system assumes output dtypes should match input dtypes, but topk takes float inputs
        # and must always return int64 indices per PyTorch's API contract. This should be
        # fixed by updating the autocast system to handle operation-specific dtype requirements.
        # Convert indices to int64 to match PyTorch's topk behavior
        if isinstance(output, tuple) and len(output) == 2:
            values, indices = output
            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            return (values, indices)
        else:
            # Single tensor case (shouldn't happen for topk)
            return result
