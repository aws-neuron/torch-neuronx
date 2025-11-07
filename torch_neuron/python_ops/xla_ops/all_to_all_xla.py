"""XLA implementation of the AllToAll collective operation."""

from torch_neuron.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuron.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuron.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


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
            bool: True if implementation can handle inputs, False otherwise

        Checks:
            - Parent class compatibility
            - All input tensors should be on device
            - All input tensors should have the same dtype and shape
            - Input list should have same length as world size
            - Output should be a single concatenated tensor with correct shape
            - Hardware supports AllToAll with Mesh algorithm for the given world size
        """
        if not super().can_handle(*args, **kwargs):
            return False

        input_tensors = args[0]
        replica_groups = args[1]
        output_tensors = kwargs.get("out")
        world_size = len(replica_groups[0])

        # Hardware capability check: AllToAll requires Mesh algorithm support
        if world_size not in {4, 8, 16} and world_size % 32 != 0:
            return False

        # Each replica group start rank should be multiply of group size
        if not all(replica_group[0] % world_size == 0 for replica_group in replica_groups):
            return False

        # Check input tensor list length matches world size
        if len(input_tensors) != world_size:
            return False

        if not output_tensors:
            return False

        # Now expecting a single output tensor (concatenated result)
        # The backend will handle slicing this into individual tensors
        if len(output_tensors) != 1:
            return False

        first_dtype = input_tensors[0].dtype
        first_shape = input_tensors[0].shape
        expected_concat_shape = (world_size * first_shape[0], *first_shape[1:])

        # Validate all input tensors
        if not all(
            tensor.device.type == "neuron"  # device check
            and tensor.numel() != 0  # size check
            and tensor.dtype == first_dtype  # dtype check
            and tensor.shape == first_shape  # shape check
            for tensor in input_tensors
        ):
            return False

        # Validate the single output tensor (concatenated shape)
        output_tensor = output_tensors[0]
        return (
            output_tensor.device.type == "neuron"
            and output_tensor.dtype == first_dtype
            and output_tensor.shape == expected_concat_shape
        )

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
