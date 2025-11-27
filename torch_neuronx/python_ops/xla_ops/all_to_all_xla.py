"""XLA implementation of the AllToAll collective operation."""

from torch_neuronx.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


class AllToAllXlaOp(XLABuilderOpImpl):
    """XLA implementation of the AllToAll collective operation."""

    def __init__(self):
        """Initialize AllToAllXlaOp with XLA kernel configuration.

        Sets up the XLA kernel with static arguments for replica groups.
        Enables collective operations for distributed computing.
        """
        super().__init__(
            op_name="all_to_all", static_argnames=["replica_groups"], has_collectives=True
        )

    def hlo_fn(self, input_tensors, **kwargs):
        """Generate XLA computation for all-to-all operation.

        Args:
            input_tensors: List of input tensors to send to each rank
            **kwargs: Keyword arguments including:
                replica_groups: Groups of replicas for all-to-all communication

        Returns:
            XLA computation for all-to-all operation
        """
        replica_groups = kwargs["replica_groups"]
        dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(input_tensors[0].dtype)
        scribe = HloScribe()

        def _all_to_all(scribe):
            dtype_shape = HloShape(scribe, dtype)
            parameters = []

            # Create parameters for each input tensor
            for i, tensor in enumerate(input_tensors):
                parameters.append(dtype_shape[tensor.shape].Parameter(parameter_number=i))

            # Calculate shapes
            group_size = len(replica_groups[0])
            input_shape = input_tensors[0].shape

            # Step 1: Concatenate all input tensors along dimension 0
            # Since we don't support group_size = 1, len(parameters) is always > 1
            concat_shape = (group_size * input_shape[0], *input_shape[1:])
            concatenated = dtype_shape[concat_shape].Concatenate(*parameters, dimensions=[0])

            # Step 2: Perform native XLA AllToAll operation using StableHLO parameters
            # AllToAll splits along split_dimension, scatters between processes,
            # and concatenates along concat_dimension
            # Return the concatenated result - slicing will be done at backend level
            return dtype_shape[concat_shape].AllToAll(
                concatenated, replica_groups=replica_groups, dimensions={0}
            )

        return scribe(_all_to_all)

    def can_handle(self, *args, **kwargs):
        """Check if inputs can be handled by this implementation.

        Args:
            *args: Positional arguments where:
                args[0]: Input tensor list
                args[1]: Replica groups
            **kwargs: Additional keyword arguments
                out: Output tensor (single concatenated tensor, slicing done at backend level)

        Returns:
            tuple[bool, str | None]: (can_handle, error_reason)

        Checks:
            - Parent class compatibility
            - All input tensors should be on device
            - All input tensors should have the same dtype and shape
            - Input list should have same length as world size
            - Output should be a single concatenated tensor with correct shape
            - Hardware supports AllToAll with Mesh algorithm for the given world size
        """
        if not super().can_handle(*args, **kwargs):
            return False, None

        input_tensors = args[0]
        replica_groups = args[1]
        output_tensors = kwargs.get("out")
        world_size = len(replica_groups[0])

        # Hardware capability check: AllToAll requires Mesh algorithm support
        if world_size not in {4, 8, 16} and world_size % 32 != 0:
            return (
                False,
                (
                    f"unsupported world size {world_size}, "
                    f"supported sizes: 4, 8, 16, or multiples of 32"
                ),
            )

        # Each replica group start rank should be multiply of group size
        invalid_groups = [
            (i, group[0], group)
            for i, group in enumerate(replica_groups)
            if group[0] % world_size != 0
        ]
        if invalid_groups:
            group_info = ", ".join(
                [f"group[{i}] {group} starts at rank {start}" for i, start, group in invalid_groups]
            )
            return (
                False,
                (
                    f"replica group start ranks must be multiples of "
                    f"world size ({world_size}), found: {group_info}"
                ),
            )

        # Check input tensor list length matches world size
        if len(input_tensors) != world_size:
            return (
                False,
                (
                    f"number of input tensors ({len(input_tensors)}) "
                    f"must equal world size ({world_size})"
                ),
            )

        if not output_tensors:
            return False, "output tensors are required but not provided"

        # Now expecting a single output tensor (concatenated result)
        if len(output_tensors) != 1:
            return (
                False,
                (f"expected 1 output tensor (concatenated result), got {len(output_tensors)}"),
            )

        first_dtype = input_tensors[0].dtype
        first_shape = input_tensors[0].shape
        expected_concat_shape = (world_size * first_shape[0], *first_shape[1:])

        # Validate all input tensors
        device_issues = [
            (i, tensor.device)
            for i, tensor in enumerate(input_tensors)
            if tensor.device.type != "neuron"
        ]
        if device_issues:
            device_info = ", ".join([f"tensor[{i}]: {device}" for i, device in device_issues])
            return (
                False,
                (f"all input tensors must be on neuron device, found: {device_info}"),
            )

        empty_tensors = [
            (i, tensor.numel()) for i, tensor in enumerate(input_tensors) if tensor.numel() == 0
        ]
        if empty_tensors:
            return (
                False,
                (
                    f"input tensors cannot be empty, "
                    f"found empty tensors at indices: {[i for i, _ in empty_tensors]}"
                ),
            )

        dtype_mismatches = [
            (i, tensor.dtype)
            for i, tensor in enumerate(input_tensors)
            if tensor.dtype != first_dtype
        ]
        if dtype_mismatches:
            dtype_info = ", ".join([f"tensor[{i}]: {dtype}" for i, dtype in dtype_mismatches])
            return (
                False,
                (
                    f"all input tensors must have same dtype, "
                    f"expected {first_dtype} but found: {dtype_info}"
                ),
            )

        shape_mismatches = [
            (i, tensor.shape)
            for i, tensor in enumerate(input_tensors)
            if tensor.shape != first_shape
        ]
        if shape_mismatches:
            shape_info = ", ".join([f"tensor[{i}]: {shape}" for i, shape in shape_mismatches])
            return (
                False,
                (
                    f"all input tensors must have same shape, "
                    f"expected {first_shape} but found: {shape_info}"
                ),
            )

        # Validate the single output tensor (concatenated shape)
        output_tensor = output_tensors[0]
        if output_tensor.device.type != "neuron":
            return (
                False,
                (f"output tensor must be on neuron device, got {output_tensor.device}"),
            )

        if output_tensor.dtype != first_dtype:
            return (
                False,
                (
                    f"output tensor dtype mismatch, "
                    f"expected {first_dtype}, got {output_tensor.dtype}"
                ),
            )

        if output_tensor.shape != expected_concat_shape:
            return (
                False,
                (
                    f"output tensor shape mismatch, "
                    f"expected {expected_concat_shape}, got {output_tensor.shape}"
                ),
            )

        return True, None

    def _execute_impl(self, input_tensors, replica_groups, out=None):
        """Execute the all-to-all operation.

        Args:
            input_tensors: List of input tensors to send to each rank
            replica_groups: Groups of replicas for all-to-all communication
            out: Tuple of output tensors (required for all-to-all)

        Returns:
            ExecutionResult: Object containing execution results
        """
        return self.kernel(
            *input_tensors,
            replica_groups=replica_groups,
            out=out,
        )
