"""XLA implementation of the ReduceScatter collective operation."""

import torch

from torch_neuronx.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter

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
                reduce_scatter_output_shape = final_output_shape
                needs_reshape = False
            else:
                input_shape = list(first_input_tensor.shape)
                output_shape = list(output_tensor.shape)
                # Normalize shapes to handle rank mismatches
                normalized_input_shape, normalized_output_shape = self._normalize_shapes(
                    input_shape, output_shape
                )
                diff_indices = [
                    i
                    for i, (a, b) in enumerate(
                        zip(normalized_input_shape, normalized_output_shape, strict=False)
                    )
                    if a != b
                ]
                scatter_dimension = diff_indices[0] if diff_indices else 0

                # Create parameter with actual input shape
                param = dtype_shape[first_input_tensor.shape].Parameter(parameter_number=0)

                # Check if normalization changed the shapes
                needs_reshape = len(input_shape) != len(normalized_input_shape) or len(
                    output_shape
                ) != len(normalized_output_shape)

                # Reshape to normalized shape if ranks differ
                if needs_reshape:
                    processed_input = dtype_shape[normalized_input_shape].Reshape(param)
                    reduce_scatter_output_shape = dtype_shape[normalized_output_shape]
                else:
                    processed_input = param
                    reduce_scatter_output_shape = final_output_shape

            # Perform ReduceScatter
            reduce_scatter_result = reduce_scatter_output_shape.ReduceScatter(
                processed_input,
                replica_groups=replica_groups,
                to_apply=reduce_fn,
                dimensions={scatter_dimension},
            )

            # Reshape back to expected output shape if needed
            if original_tensor_count == 1 and needs_reshape:
                reduce_scatter_result = final_output_shape.Reshape(reduce_scatter_result)

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
        return self._validate_input_tensor_list_with_error(input_tensor_list) is None

    def _validate_input_tensor_list_with_error(self, input_tensor_list):
        """Validate input tensors and return error message if invalid."""
        if not input_tensor_list or len(input_tensor_list) == 0:
            return "no input tensors provided"

        # Check that input_tensor_list is a list of torch.Tensor objects, not nested lists
        non_tensor_indices = [
            i for i, item in enumerate(input_tensor_list) if not isinstance(item, torch.Tensor)
        ]
        if non_tensor_indices:
            return f"input list contains non-tensor objects at indices: {non_tensor_indices}"

        non_neuron_devices = [
            (i, tensor.device)
            for i, tensor in enumerate(input_tensor_list)
            if tensor.device.type != "neuron"
        ]
        if non_neuron_devices:
            device_info = ", ".join([f"tensor[{i}]: {device}" for i, device in non_neuron_devices])
            return f"all tensors must be on neuron device, found: {device_info}"

        empty_tensor_indices = [
            i for i, tensor in enumerate(input_tensor_list) if tensor.numel() == 0
        ]
        if empty_tensor_indices:
            return (
                f"tensors cannot be empty, found empty tensors at indices: {empty_tensor_indices}"
            )

        first_input_tensor = input_tensor_list[0]
        dtype_mismatches = [
            (i, tensor.dtype)
            for i, tensor in enumerate(input_tensor_list)
            if tensor.dtype != first_input_tensor.dtype
        ]
        if dtype_mismatches:
            dtype_info = ", ".join([f"tensor[{i}]: {dtype}" for i, dtype in dtype_mismatches])
            return (
                f"all tensors must have same dtype, expected {first_input_tensor.dtype} "
                f"but found: {dtype_info}"
            )

        first_shape = first_input_tensor.shape
        shape_mismatches = [
            (i, tensor.shape)
            for i, tensor in enumerate(input_tensor_list)
            if tensor.shape != first_shape
        ]
        if shape_mismatches:
            shape_info = ", ".join([f"tensor[{i}]: {shape}" for i, shape in shape_mismatches])
            return (
                f"all tensors must have same shape, expected {first_shape} but found: {shape_info}"
            )

        return None

    def _validate_output_tensor(self, output_tensor, input_tensor):
        """Validate output tensor compatibility."""
        return self._validate_output_tensor_with_error(output_tensor, input_tensor) is None

    def _normalize_shapes(self, input_shape, output_shape):
        """Normalize shapes to same rank by padding with 1s.

        Pads the shorter shape with leading 1s to match ranks.
        Then validates that dimensions can only differ if one side is 1.

        Returns:
            tuple: (normalized_input_shape, normalized_output_shape, error_message)
        """
        input_shape = list(input_shape)
        output_shape = list(output_shape)

        # Pad shorter shape with leading 1s
        if len(input_shape) > len(output_shape):
            output_shape = [1] * (len(input_shape) - len(output_shape)) + output_shape
        elif len(output_shape) > len(input_shape):
            input_shape = [1] * (len(output_shape) - len(input_shape)) + input_shape

        return input_shape, output_shape

    def _validate_output_tensor_with_error(self, output_tensor, input_tensor):
        """Validate output tensor and return error message if invalid."""
        if output_tensor is None:
            return "output tensor is required but not provided"

        if output_tensor.numel() == 0:
            return "output tensor cannot be empty"

        if output_tensor.device.type != "neuron":
            return f"output tensor must be on neuron device, got {output_tensor.device}"

        output_shape = list(output_tensor.shape)
        input_shape = list(input_tensor.shape)
        input_shape, output_shape = self._normalize_shapes(input_shape, output_shape)
        if len(output_shape) != len(input_shape):
            return (
                f"output tensor rank ({len(output_shape)}) must match "
                f"input tensor rank ({len(input_shape)})"
            )

        # Validate input and output tensors have the same dtype
        if output_tensor.dtype != input_tensor.dtype:
            return (
                f"output tensor dtype ({output_tensor.dtype}) must match "
                f"input tensor dtype ({input_tensor.dtype})"
            )

        return None

    def _validate_replica_groups(self, replica_groups):
        """Validate replica_groups structure."""
        return replica_groups and len(replica_groups) > 0

    def _get_replica_group(self, replica_groups):
        """Get first replica group."""
        return replica_groups[0]

    def _validate_single_tensor_scatter(self, input_shape, output_shape, replica_groups):
        """Validate scatter operation for single tensor case."""
        return (
            self._validate_single_tensor_scatter_with_error(
                input_shape, output_shape, replica_groups
            )
            is None
        )

    def _validate_single_tensor_scatter_with_error(self, input_shape, output_shape, replica_groups):
        """Validate single tensor scatter and return error message if invalid."""
        # Normalize shapes to handle rank mismatches
        input_shape, output_shape = self._normalize_shapes(input_shape, output_shape)
        replica_group = self._get_replica_group(replica_groups)
        diff_indices = [
            i for i, (a, b) in enumerate(zip(input_shape, output_shape, strict=False)) if a != b
        ]

        if len(diff_indices) > 1:
            diff_info = ", ".join(
                [f"dim[{i}]: {input_shape[i]} -> {output_shape[i]}" for i in diff_indices]
            )
            return (
                f"single tensor scatter can only differ in one dimension, "
                f"found differences in: {diff_info}"
            )

        if len(diff_indices) == 0:
            if len(replica_group) != 1:
                return (
                    f"when input and output shapes are identical, "
                    f"replica group must have size 1, got {len(replica_group)}"
                )
            return None

        scatter_dimension = diff_indices[0]
        len_replica_group = len(replica_group)
        expected_ratio = input_shape[scatter_dimension] / output_shape[scatter_dimension]

        if len_replica_group != expected_ratio:
            return (
                f"scatter dimension {scatter_dimension}: input size "
                f"({input_shape[scatter_dimension]}) must be evenly divisible by "
                f"replica group size ({len_replica_group}), but "
                f"{input_shape[scatter_dimension]} / {len_replica_group} = "
                f"{input_shape[scatter_dimension] / len_replica_group}. Expected output size: "
                f"{input_shape[scatter_dimension] // len_replica_group}, actual output size: "
                f"{output_shape[scatter_dimension]}"
            )

        return None

    def _validate_multi_tensor_scatter(self, input_tensor_list, replica_groups):
        """Validate scatter operation for multiple tensor case."""
        return (
            self._validate_multi_tensor_scatter_with_error(input_tensor_list, replica_groups)
            is None
        )

    def _validate_multi_tensor_scatter_with_error(self, input_tensor_list, replica_groups):
        """Validate multi tensor scatter and return error message if invalid."""
        replica_group = self._get_replica_group(replica_groups)
        len_replica_group = len(replica_group)
        if len_replica_group != len(input_tensor_list):
            return (
                f"multi-tensor scatter: replica group size ({len_replica_group}) "
                f"must equal number of input tensors ({len(input_tensor_list)})"
            )
        return None

    def can_handle(self, *args, **kwargs):
        """Check if inputs can be handled by this implementation.

        Args:
            *args: Positional arguments where:
                args[0]: list of Input tensors
                args[1]: Replica group
                args[2]: Reduction operation
            **kwargs: Additional keyword arguments

        Returns:
            tuple[bool, str | None]: (can_handle, error_reason)
        """
        # Base class validation
        if not super().can_handle(*args, **kwargs):
            return False, None

        # Extract arguments
        input_tensor_list = args[0]
        replica_groups = args[1]
        reduce_op = args[2]
        output_tensor = kwargs["out"]
        first_input_tensor = input_tensor_list[0]

        # 1. Validate replica groups
        if not self._validate_replica_groups(replica_groups):
            return False, "invalid or empty replica groups"

        # 2. Validate reduce function
        if not self._validate_reduce_op(reduce_op):
            from .reduce_functions import SUPPORTED_REDUCE_OPS

            return (
                False,
                (
                    f"unsupported reduce operation '{reduce_op}', "
                    f"supported ops: {list(SUPPORTED_REDUCE_OPS)}"
                ),
            )

        # 3. Validate input tensor list
        input_validation_error = self._validate_input_tensor_list_with_error(input_tensor_list)
        if input_validation_error:
            return False, input_validation_error

        # 4. Validate output tensor
        output_validation_error = self._validate_output_tensor_with_error(
            output_tensor, first_input_tensor
        )
        if output_validation_error:
            return False, output_validation_error

        # 5. Validate Scatter dimensions
        input_shape = list(first_input_tensor.shape)
        output_shape = list(output_tensor.shape)
        if len(input_tensor_list) == 1:
            scatter_error = self._validate_single_tensor_scatter_with_error(
                input_shape, output_shape, replica_groups
            )
            if scatter_error:
                return False, scatter_error
        else:
            scatter_error = self._validate_multi_tensor_scatter_with_error(
                input_tensor_list, replica_groups
            )
            if scatter_error:
                return False, scatter_error

        return True, None

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
