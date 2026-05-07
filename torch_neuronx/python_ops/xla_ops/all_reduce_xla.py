"""XLA implementation of the AllReduce collective operation."""

import torch

from torch_neuronx.python_ops.xla_builder.op_impl import XLABuilderOpImpl
from torch_neuronx.python_ops.xla_builder.scribe import HloScribe, HloShape
from torch_neuronx.python_ops.xla_builder.type_converter import XLABuilderTypeConverter

from .avg_utils import apply_avg_division_single, apply_avg_division_tuple, prepare_avg_inputs


class AllReduceXLAOp(XLABuilderOpImpl):
    """XLA implementation of the AllReduce collective operation."""

    def __init__(self):
        """Initialize AllReduceXLAOp with XLA kernel configuration.
        Sets up the XLA kernel with static arguments for replica groups
        and reduce operation. Enables collective operations for
        distributed computing.
        """
        super().__init__(
            op_name="all_reduce", static_argnames=["replica_groups", "reduce"], has_collectives=True
        )

    def hlo_fn(self, tensors, **kwargs):
        """Generate XLA computation for all-reduce operation.

        Args:
            args: Tuple of input arguments where args[0] contains input tensors
            **kwargs: Keyword arguments including:
                replica_groups: Groups of replicas for reduction
                reduce: Reduction operation type

        Returns:
            XLA computation for all-reduce operation

        Raises:
            NotImplementedError: If reduction operation is not torch.distributed.ReduceOp.SUM
        """
        replica_groups = kwargs["replica_groups"]
        reduce = kwargs["reduce"]
        dtype = XLABuilderTypeConverter.torch_to_primitive_dtype(tensors[0].dtype)
        scribe = HloScribe()

        from .reduce_functions import get_reduce_function

        reduce_fn = get_reduce_function(reduce, dtype)

        def _all_reduce(scribe):
            dtype_shape = HloShape(scribe, dtype)
            parameters, out_shapes = [], []

            # Determine tensor count (exclude divisor for AVG)
            original_tensor_count = len(tensors) - (1 if reduce == "AVG" else 0)

            # Add tensor parameters (excluding divisor)
            for i in range(original_tensor_count):
                tensor = tensors[i]
                out_shape = kwargs["out"][i].shape if kwargs.get("out") else tensor.shape
                out_shapes.append(out_shape)
                parameters.append(dtype_shape[tensor.shape].Parameter(parameter_number=i))

            output_list = [dtype_shape[out_shape] for out_shape in out_shapes]
            output_shape = scribe.tuple(*output_list) if len(output_list) > 1 else output_list[0]

            # Perform AllReduce only on tensor parameters, not divisor
            all_reduce_result = output_shape.AllReduce(
                *parameters, replica_groups=replica_groups, to_apply=reduce_fn
            )

            # For AVG, divide by number of replicas in HLO
            if reduce == "AVG":
                divisor_param = dtype_shape.Parameter(parameter_number=original_tensor_count)
                if len(output_list) > 1:
                    return apply_avg_division_tuple(
                        scribe, all_reduce_result, divisor_param, output_list
                    )
                else:
                    return apply_avg_division_single(
                        scribe, all_reduce_result, divisor_param, output_list[0]
                    )

            return all_reduce_result

        return scribe(_all_reduce)

    def can_handle(self, *args, **kwargs):
        """Check if inputs can be handled by this implementation.

        Args:
            *args: Positional arguments where:
                args[0]: Input tensors
                args[2]: Reduction operation
            **kwargs: Additional keyword arguments

        Returns:
            tuple[bool, str | None]: (can_handle, error_reason)

        Checks:
            - Parent class compatibility
            - Reduction operation is supported
            - All input tensors should be on device
            - All input tensors have the same dtype
        """
        if not super().can_handle(*args, **kwargs):
            return False, None

        from .reduce_functions import SUPPORTED_REDUCE_OPS, is_supported_reduce_op

        reduce_op = args[2]
        if not is_supported_reduce_op(reduce_op):
            return (
                False,
                (
                    f"unsupported reduce operation '{reduce_op}', "
                    f"supported ops: {list(SUPPORTED_REDUCE_OPS)}"
                ),
            )

        tensors = args[0]

        # Check all tensors should be on device
        non_neuron_devices = [tensor.device for tensor in tensors if tensor.device.type != "neuron"]
        if non_neuron_devices:
            return (
                False,
                (f"all tensors must be on neuron device, found tensors on: {non_neuron_devices}"),
            )

        # No tensor should be zero sized
        empty_tensor_indices = [i for i, tensor in enumerate(tensors) if tensor.numel() == 0]
        if empty_tensor_indices:
            return (
                False,
                (
                    f"tensors cannot be empty, "
                    f"found empty tensors at indices: {empty_tensor_indices}"
                ),
            )

        # Get dtype of first tensor and check consistency
        first_dtype = tensors[0].dtype
        mismatched_dtypes = [
            (i, tensor.dtype) for i, tensor in enumerate(tensors) if tensor.dtype != first_dtype
        ]
        if mismatched_dtypes:
            dtype_info = ", ".join([f"tensor[{i}]: {dtype}" for i, dtype in mismatched_dtypes])
            return (
                False,
                (
                    f"all tensors must have same dtype, expected {first_dtype} "
                    f"but found: {dtype_info}"
                ),
            )

        return True, None

    def _execute_impl(self, inputs, replica_groups, reduce, out=None):
        """Execute the all-reduce operation.

        Args:
            inputs (tuple[torch.Tensor]): Input tensors to be reduced
            replica_groups (List[List[int]]): Groups of replicas for reduction
            reduce (torch.distributed.ReduceOp): Reduction operation
            out (tuple[torch.Tensor], optional): Output tensors. If None, new tensors are created.

        Returns:
            ExecutionResult: Object containing:
                - success (bool): Whether execution was successful
                - output (tuple[torch.Tensor]): Reduced tensors if successful
                - error_msg (str): Error message if unsuccessful

        Note:
            Output tensors will have the same shape and dtype as input tensors.
        """
        output = tuple(
            [torch.empty(input.shape, dtype=input.dtype, device=input.device) for input in inputs]
            if out is None
            else out
        )

        # For AVG operation, pass divisor as additional input
        if reduce == "AVG":
            kernel_inputs = prepare_avg_inputs(inputs, reduce, replica_groups)
        else:
            kernel_inputs = inputs

        result = self.kernel(
            *kernel_inputs, replica_groups=replica_groups, reduce=reduce, out=output
        )
        return result
