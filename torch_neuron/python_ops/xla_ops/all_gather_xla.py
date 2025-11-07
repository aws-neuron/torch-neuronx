"""XLA implementation of the AllGather collective operation."""

from torch_neuron.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuron.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuron.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


class AllGatherXlaOp(XLABuilderOpImpl):
    """XLA implementation of the AllGather collective operation."""

    def __init__(self):
        """Initialize AllGatherXLAOp with XLA kernel configuration.

        Sets up the XLA kernel with static arguments for replica groups.
        Enables collective operations for distributed computing.
        """
        super().__init__(
            op_name="all_gather", static_argnames=["replica_groups"], has_collectives=True
        )

    def hlo_fn(self, input_tensors, **kwargs):
        """Generate XLA computation for all-gather operation.

        Args:
            args: Tuple containing (output_tensors_list, input_tensors)
            **kwargs: Keyword arguments including:
                replica_groups: Groups of replicas for gathering
        Returns:
            XLA computation for all-gather operation
        """
        replica_groups = kwargs["replica_groups"]
        slice_output = kwargs.get("slice_output", False)
        dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(input_tensors[0].dtype)
        scribe = HloScribe()

        def _all_gather(scribe):
            dtype_shape = HloShape(scribe, dtype)
            parameters, out_shapes = [], []
            world_size = len(replica_groups[0])

            for i, tensor in enumerate(input_tensors):
                out_shapes.append((world_size * tensor.shape[0], *tensor.shape[1:]))
                parameters.append(dtype_shape[tensor.shape].Parameter(parameter_number=i))

            output_list = [dtype_shape[out_shape] for out_shape in out_shapes]
            output_shape = scribe.tuple(*output_list) if len(output_list) > 1 else output_list[0]
            # Calculate shapes

            # Step 1: Perform AllGather to get concatenated result
            gathered = output_shape.AllGather(
                *parameters, replica_groups=replica_groups, dimensions={0}
            )

            def _create_slice_dimensions(input_shape, rank):
                """Create slice dimensions for extracting rank's portion from gathered tensor."""
                slice_dimensions = []
                for dim, size in enumerate(input_shape):
                    if dim == 0:
                        slice_dimensions.append(
                            {"start": rank * size, "limit": (rank + 1) * size, "stride": 1}
                        )
                    else:
                        # For other dimensions, take the entire dimension
                        slice_dimensions.append({"start": 0, "limit": size, "stride": 1})
                return slice_dimensions

            def _extract_rank_slices(dtype_shape, input_shape, gathered_tensor, world_size):
                """Extract slices for each rank from gathered tensors."""
                output_pieces = []
                out_shapes = []

                for rank in range(world_size):
                    # Create slice specifications for this rank
                    slice_dimensions = _create_slice_dimensions(input_shape, rank)

                    # Extract the slice for this rank
                    # Example: From gathered_tensor[6,4], extract tensor[3,4] for each rank
                    slice_piece = dtype_shape[input_shape].Slice(
                        gathered_tensor, slice_dimensions=slice_dimensions
                    )

                    output_pieces.append(slice_piece)
                    out_shapes.append(dtype_shape[input_shape])

                return output_pieces, out_shapes

            if slice_output:
                # Extract slices for each rank from the gathered tensors
                slice_pieces, slice_shapes = _extract_rank_slices(
                    dtype_shape, input_tensors[0].shape, gathered, world_size
                )
                # Create a tuple from all slices
                return scribe.tuple(*slice_shapes).Tuple(*slice_pieces)
            else:
                return gathered

        return scribe(_all_gather)

    def can_handle(self, *args, **kwargs):
        """Check if inputs can be handled by this implementation.

        Args:
            *args: Positional arguments where:
                args[0]: Input tensors
            **kwargs: Additional keyword arguments
                out: Output tensors (required for all-gather)
                slice_output: Boolean indicating if output should be sliced
                world_size: Number of processes in the group

        Returns:
            bool: True if implementation can handle inputs, False otherwise

        Checks:
            - Parent class compatibility
            - All input and output tensors should be on device
            - All input and output tensors should have the same dtype
            - Output tensor shapes should match expected dimensions based on slice_output
        """
        if not super().can_handle(*args, **kwargs):
            return False

        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        replica_groups = args[1]
        output_tensors = kwargs.get("out")
        slice_output = kwargs.get("slice_output", False)

        if not tensors or output_tensors is None:
            return False

        if not isinstance(output_tensors, (tuple | list)):
            output_tensors = [output_tensors]

        world_size = len(replica_groups[0])
        first_dtype = tensors[0].dtype

        # Check all tensors are on neuron device with same dtype
        for tensor in tensors + list(output_tensors):
            if tensor.device.type != "neuron" or tensor.size(0) == 0 or tensor.dtype != first_dtype:
                return False

        # Validate shapes based on operation type
        if slice_output:
            return len(tensors) == 1 and all(
                out.shape == tensors[0].shape for out in output_tensors
            )

        if len(tensors) == 1:
            if len(output_tensors) != 1:
                return False
            expected_shape = (world_size * tensors[0].shape[0], *tensors[0].shape[1:])
            return tuple(output_tensors[0].shape) == expected_shape

        # Multiple tensors case
        if len(output_tensors) != len(tensors):
            return False

        return all(
            out.shape == (world_size * inp.shape[0], *inp.shape[1:])
            for inp, out in zip(tensors, output_tensors, strict=False)
        )

    def _execute_impl(self, input_tensors, replica_groups, opts=None, out=None, slice_output=False):
        """Execute the all-gather operation.

        Args:
            input_tensors: Input tensors to gather
            out: Tuple of output tensors (required for all-gather)
            opts: Additional options including replica groups

        Returns:
            ExecutionResult: Object containing execution results
        """
        return self.kernel(
            *(input_tensors if isinstance(input_tensors, list) else [input_tensors]),
            replica_groups=replica_groups,
            opts=None,
            out=out,
            slice_output=slice_output,
        )
