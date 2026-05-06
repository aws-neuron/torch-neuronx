"""XLA implementation of the AllGather collective operation."""

from torch_neuronx.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter


class AllGatherXlaOp(XLABuilderOpImpl):
    """XLA implementation of the AllGather collective operation."""

    def __init__(self):
        """Initialize AllGatherXLAOp with XLA kernel configuration.

        Sets up the XLA kernel with static arguments for replica groups and slice_output.
        Enables collective operations for distributed computing.
        """
        super().__init__(
            op_name="all_gather",
            static_argnames=["replica_groups", "slice_output"],
            has_collectives=True,
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
                if len(input_tensors) == 1:
                    # Single tensor case
                    slice_pieces, slice_shapes = _extract_rank_slices(
                        dtype_shape, input_tensors[0].shape, gathered, world_size
                    )
                    return scribe.tuple(*slice_shapes).Tuple(*slice_pieces)
                else:
                    # Multiple tensors case: slice each gathered tensor
                    all_slices = []
                    all_shapes = []
                    for i, input_tensor in enumerate(input_tensors):
                        gathered_i = dtype_shape[out_shapes[i]].GetTupleElement(
                            gathered, tuple_index=i
                        )
                        slice_pieces, slice_shapes = _extract_rank_slices(
                            dtype_shape, input_tensor.shape, gathered_i, world_size
                        )
                        all_slices.extend(slice_pieces)
                        all_shapes.extend(slice_shapes)
                    return scribe.tuple(*all_shapes).Tuple(*all_slices)
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
            tuple[bool, str | None]: (can_handle, error_reason)

        Checks:
            - Parent class compatibility
            - All input and output tensors should be on device
            - All input and output tensors should have the same dtype
            - Output tensor shapes should match expected dimensions based on slice_output
        """
        if not super().can_handle(*args, **kwargs):
            return False, None

        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        replica_groups = args[1]
        output_tensors = kwargs.get("out")
        slice_output = kwargs.get("slice_output", False)

        if not tensors:
            return False, "no input tensors provided"

        if output_tensors is None:
            return False, "output tensors are required but not provided"

        if not isinstance(output_tensors, (tuple | list)):
            output_tensors = [output_tensors]

        world_size = len(replica_groups[0])
        first_dtype = tensors[0].dtype

        # Check all tensors are on neuron device
        all_tensors = tensors + list(output_tensors)
        non_neuron_tensors = [
            (i, tensor.device)
            for i, tensor in enumerate(all_tensors)
            if tensor.device.type != "neuron"
        ]
        if non_neuron_tensors:
            tensor_info = ", ".join([f"tensor[{i}]: {device}" for i, device in non_neuron_tensors])
            return (
                False,
                f"all tensors must be on neuron device, found: {tensor_info}",
            )

        # Check for empty tensors
        empty_tensors = [
            (i, tensor.shape) for i, tensor in enumerate(all_tensors) if tensor.size(0) == 0
        ]
        if empty_tensors:
            tensor_info = ", ".join([f"tensor[{i}]: {shape}" for i, shape in empty_tensors])
            return (
                False,
                (f"tensors cannot have zero size in dimension 0, found: {tensor_info}"),
            )

        # Check dtype consistency
        dtype_mismatches = [
            (i, tensor.dtype) for i, tensor in enumerate(all_tensors) if tensor.dtype != first_dtype
        ]
        if dtype_mismatches:
            dtype_info = ", ".join([f"tensor[{i}]: {dtype}" for i, dtype in dtype_mismatches])
            return (
                False,
                (
                    f"all tensors must have same dtype, expected {first_dtype} "
                    f"but found: {dtype_info}"
                ),
            )

        # Validate shapes based on operation type
        if slice_output:
            if len(tensors) == 1:
                # Single tensor case
                shape_mismatches = [
                    (i, out.shape)
                    for i, out in enumerate(output_tensors)
                    if out.shape != tensors[0].shape
                ]
                if shape_mismatches:
                    shape_info = ", ".join(
                        [f"output[{i}]: {shape}" for i, shape in shape_mismatches]
                    )
                    return (
                        False,
                        (
                            f"with slice_output=True, all output tensors must match "
                            f"input shape {tensors[0].shape}, found: {shape_info}"
                        ),
                    )
            else:
                # Multiple tensors case: output should be nested structure
                if len(output_tensors) != len(tensors) * world_size:
                    return (
                        False,
                        (
                            f"with slice_output=True and {len(tensors)} input tensors, "
                            f"expected {len(tensors) * world_size} output tensors, "
                            f"got {len(output_tensors)}"
                        ),
                    )
                for i, input_tensor in enumerate(tensors):
                    for j in range(world_size):
                        out_idx = i * world_size + j
                        if output_tensors[out_idx].shape != input_tensor.shape:
                            return (
                                False,
                                (
                                    f"output tensor at index {out_idx} has shape "
                                    f"{output_tensors[out_idx].shape}, "
                                    f"expected {input_tensor.shape} to match input tensor {i}"
                                ),
                            )
            return True, None

        if len(tensors) == 1:
            if len(output_tensors) != 1:
                return (
                    False,
                    (
                        f"single input tensor requires single output tensor, "
                        f"got {len(output_tensors)} output tensors"
                    ),
                )
            expected_shape = (world_size * tensors[0].shape[0], *tensors[0].shape[1:])
            if tuple(output_tensors[0].shape) != expected_shape:
                return (
                    False,
                    (
                        f"output shape mismatch, expected {expected_shape}, "
                        f"got {output_tensors[0].shape}"
                    ),
                )
            return True, None

        # Multiple tensors case
        if len(output_tensors) != len(tensors):
            return (
                False,
                (
                    f"number of output tensors ({len(output_tensors)}) "
                    f"must match number of input tensors ({len(tensors)})"
                ),
            )

        shape_mismatches = []
        for i, (inp, out) in enumerate(zip(tensors, output_tensors, strict=False)):
            expected_shape = (world_size * inp.shape[0], *inp.shape[1:])
            if out.shape != expected_shape:
                shape_mismatches.append(f"pair[{i}]: expected {expected_shape}, got {out.shape}")

        if shape_mismatches:
            return (
                False,
                f"output shape mismatches: {'; '.join(shape_mismatches)}",
            )

        return True, None

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
