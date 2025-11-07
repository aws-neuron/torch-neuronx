"""XLA implementation of the ReduceScatter collective operation."""

import torch

from torch_neuron.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuron.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuron.python_ops.xla_builder.type_converter import XLABuilderTypeConverter

from .avg_utils import apply_avg_division_single, prepare_avg_inputs


class ReduceScatterXLAOp(XLABuilderOpImpl):
    """XLA implementation of the ReduceScatter collective operation."""

    def __init__(self):
        """Initialize ReduceScatterXLAOp with XLA kernel configuration.
        Sets up the XLA kernel with static arguments for replica groups
        and reduce operation. Enables collective operations for
        distributed computing.
        """
        super().__init__(
            op_name="reduce_scatter",
            static_argnames=["replica_groups", "reduce"],
            has_collectives=True,
        )

    def hlo_fn(self, input_tensor_list, **kwargs):
        """Generate XLA computation for reduce-scatter operation.

        Args:
            input_tensor_list: List of input tensors to be reduced and scattered
            **kwargs: Keyword arguments including:
                replica_groups: Groups of replicas for reduction
                reduce: Reduction operation type
                out: Output tensor

        Returns:
            XLA computation for reduce-scatter operation

        Raises:
            NotImplementedError: If reduction operation is not torch.distributed.ReduceOp.SUM
        """
        replica_groups = kwargs["replica_groups"]
        reduce = kwargs["reduce"]
        first_input_tensor = input_tensor_list[0]
        output_tensor = kwargs.get("out")
        dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(first_input_tensor.dtype)
        scribe = HloScribe()

        from .reduce_functions import get_reduce_function

        reduce_fn = get_reduce_function(reduce, dtype)

        def concatenate_tensors(scribe):
            """Concatenate multiple input tensors along dimension 0.

            Used when multiple tensors need to be combined before reduce-scatter.
            Automatically handles AVG case by excluding divisor parameter.
            """
            parameters = []
            dtype_shape = HloShape(scribe, dtype)

            # Determine tensor count (exclude divisor for AVG)
            tensor_count = len(input_tensor_list) - (1 if reduce == "AVG" else 0)

            # Calculate output shape after concatenation
            out_tensor_shape = list(first_input_tensor.shape)
            out_tensor_shape[0] = out_tensor_shape[0] * tensor_count
            out_shape = dtype_shape[out_tensor_shape]

            # Create parameter for each input tensor (excluding divisor)
            for i in range(tensor_count):
                input_tensor = input_tensor_list[i]
                parameters.append(dtype_shape[input_tensor.shape].Parameter(parameter_number=i))

            return out_shape.Concatenate(*parameters, dimensions=[0])

        def _reduce_scatter(scribe):
            """Main reduce-scatter computation logic."""
            dtype_shape = HloShape(scribe, dtype)

            out_shape = output_tensor.shape
            final_output_shape = dtype_shape[out_shape]

            # Determine tensor count (exclude divisor for AVG)
            original_tensor_count = len(input_tensor_list) - (1 if reduce == "AVG" else 0)

            if original_tensor_count > 1:
                processed_input = concatenate_tensors(scribe)
                scatter_dimension = 0
            else:
                input_shape = list(first_input_tensor.shape)
                output_shape = list(output_tensor.shape)
                diff_indices = [
                    i
                    for i, (a, b) in enumerate(zip(input_shape, output_shape, strict=False))
                    if a != b
                ]
                scatter_dimension = diff_indices[0] if diff_indices else 0
                processed_input = dtype_shape[first_input_tensor.shape].Parameter(
                    parameter_number=0
                )

            # Perform ReduceScatter
            reduce_scatter_result = final_output_shape.ReduceScatter(
                processed_input,
                replica_groups=replica_groups,
                to_apply=reduce_fn,
                dimensions={scatter_dimension},
            )

            # For AVG, divide by number of replicas
            if reduce == "AVG":
                divisor_param = dtype_shape.Parameter(parameter_number=len(input_tensor_list) - 1)
                return apply_avg_division_single(
                    scribe, reduce_scatter_result, divisor_param, final_output_shape
                )

            return reduce_scatter_result

        return scribe(_reduce_scatter)

    def _validate_reduce_op(self, reduce_op):
        """Validate that reduce operation is supported."""
        from .reduce_functions import is_supported_reduce_op

        return is_supported_reduce_op(reduce_op)

    def _validate_input_tensor_list(self, input_tensor_list):
        """Validate input tensors are valid and consistent."""
        if not input_tensor_list or len(input_tensor_list) == 0:
            return False

        # Check that input_tensor_list is a list of torch.Tensor objects, not nested lists
        if not all(isinstance(tensor, torch.Tensor) for tensor in input_tensor_list):
            return False

        if not all(tensor.device.type == "neuron" for tensor in input_tensor_list):
            return False

        if not all(tensor.numel() != 0 for tensor in input_tensor_list):
            return False

        first_input_tensor = input_tensor_list[0]
        if not all(tensor.dtype == first_input_tensor.dtype for tensor in input_tensor_list):
            return False

        first_shape = first_input_tensor.shape
        return all(tensor.shape == first_shape for tensor in input_tensor_list)

    def _validate_output_tensor(self, output_tensor, input_tensor):
        """Validate output tensor compatibility."""
        if output_tensor is None or output_tensor.numel() == 0:
            return False

        if output_tensor.device.type != "neuron":
            return False

        output_shape = list(output_tensor.shape)
        input_shape = list(input_tensor.shape)
        if len(output_shape) != len(input_shape):
            return False

        # Validate input and output tensors have the same dtype
        return output_tensor.dtype == input_tensor.dtype

    def _validate_replica_groups(self, replica_groups):
        """Validate replica_groups structure."""
        return replica_groups and len(replica_groups) > 0

    def _get_replica_group(self, replica_groups):
        """Get first replica group."""
        return replica_groups[0]

    def _validate_single_tensor_scatter(self, input_shape, output_shape, replica_groups):
        """Validate scatter operation for single tensor case."""
        replica_group = self._get_replica_group(replica_groups)
        diff_indices = [
            i for i, (a, b) in enumerate(zip(input_shape, output_shape, strict=False)) if a != b
        ]

        if len(diff_indices) > 1:
            return False

        if len(diff_indices) == 0:
            return len(replica_group) == 1

        scatter_dimension = diff_indices[0]
        len_replica_group = len(replica_group)
        expected_ratio = input_shape[scatter_dimension] / output_shape[scatter_dimension]

        return len_replica_group == expected_ratio

    def _validate_multi_tensor_scatter(self, input_tensor_list, replica_groups):
        """Validate scatter operation for multiple tensor case."""
        replica_group = self._get_replica_group(replica_groups)
        len_replica_group = len(replica_group)
        return len_replica_group == len(input_tensor_list)

    def can_handle(self, *args, **kwargs):
        """Check if inputs can be handled by this implementation.

        Args:
            *args: Positional arguments where:
                args[0]: list of Input tensors
                args[1]: Replica group
                args[2]: Reduction operation
            **kwargs: Additional keyword arguments

        Returns:
            bool: True if implementation can handle inputs, False otherwise
        """
        # Base class validation
        if not super().can_handle(*args, **kwargs):
            return False

        # Extract arguments
        input_tensor_list = args[0]
        replica_groups = args[1]
        reduce_op = args[2]
        output_tensor = kwargs["out"]
        first_input_tensor = input_tensor_list[0]

        # 1. Validate replica groups
        if not self._validate_replica_groups(replica_groups):
            return False

        # 2. Validate reduce function
        if not self._validate_reduce_op(reduce_op):
            return False

        # 3. Validate input tensor list
        if not self._validate_input_tensor_list(input_tensor_list):
            return False

        # 4. Validate output tensor
        if not self._validate_output_tensor(output_tensor, first_input_tensor):
            return False

        # 5. Validate Scatter dimensions
        input_shape = list(first_input_tensor.shape)
        output_shape = list(output_tensor.shape)
        if len(input_tensor_list) == 1:
            return self._validate_single_tensor_scatter(input_shape, output_shape, replica_groups)
        else:
            return self._validate_multi_tensor_scatter(input_tensor_list, replica_groups)

    def _execute_impl(self, inputs, replica_groups, reduce, out=None):
        """Execute the reduce-scatter operation.

        Args:
            inputs (list[torch.Tensor]): Input tensors to be reduced and scattered
            replica_groups (List[List[int]]): Groups of replicas for reduction
            reduce (torch.distributed.ReduceOp): Reduction operation
            out (torch.Tensor): Output tensor.

        Returns:
            ExecutionResult: Object containing:
                - success (bool): Whether execution was successful
                - output (torch.Tensor): Reduced and scattered tensor if successful
                - error_msg (str): Error message if unsuccessful

        Note:
            Output tensor will have reduced size along the scatter dimension
            compared to input tensors.
        """
        # For AVG operation, pass divisor as additional input
        if reduce == "AVG":
            kernel_inputs = prepare_avg_inputs(inputs, reduce, replica_groups)
        else:
            kernel_inputs = inputs

        # Execute the XLA kernel with all parameters
        result = self.kernel(*kernel_inputs, replica_groups=replica_groups, reduce=reduce, out=out)

        return result
