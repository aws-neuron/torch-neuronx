"""Shared reduction functions for XLA collective operations."""

from torch_neuronx.python_ops.xla_builder.scribe import HloShape

# Supported operations
SUPPORTED_REDUCE_OPS = {"SUM", "AVG", "MIN", "MAX"}


def is_supported_reduce_op(reduce_op):
    """Check if the reduce operation is supported.

    Args:
        reduce_op (str): The reduction operation to check

    Returns:
        bool: True if supported, False otherwise
    """
    return reduce_op in SUPPORTED_REDUCE_OPS


def get_reduce_function(reduce_op, dtype):
    """Get the appropriate XLA reduction function for the given operation.

    Args:
        reduce_op (str): The reduction operation ("SUM", "AVG", "MIN", "MAX")
        dtype: The data type for the HLO computation

    Returns:
        function: XLA reduction function that takes a scribe parameter

    Raises:
        NotImplementedError: If the reduction operation is not supported
    """
    # Map operations to their HLO methods
    # for Avg, we do sum first, and then apply division on the result. This follows
    # other frameworks (PyTorch, JAX)
    op_map = {"SUM": "Add", "AVG": "Add", "MIN": "Minimum", "MAX": "Maximum"}

    if reduce_op not in op_map:
        raise NotImplementedError(f"Reduce operation {reduce_op} is not supported")

    hlo_method = op_map[reduce_op]

    def reduce_fn(scribe):
        dtype_shape = HloShape(scribe, dtype)
        p0 = dtype_shape.Parameter(parameter_number=0)
        p1 = dtype_shape.Parameter(parameter_number=1)
        return getattr(dtype_shape, hlo_method)(p0, p1)

    return reduce_fn
