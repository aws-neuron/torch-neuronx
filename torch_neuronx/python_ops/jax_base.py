"""JAX-based operation implementation base class.

This module provides backward compatibility by importing from the refactored
JAX operations package. The original monolithic implementation has been
refactored into a clean, modular architecture with proper separation of concerns.
"""

# Import refactored components for backward compatibility
from .jax import JaxKernel, JaxOpImpl

# Also import helper functions that were in the original file
from .jax.config import ReductionOps


# Create backward-compatible helper functions
def _infer_identity_value(aten_op_name: str):
    """Infer identity value for reduction operations.

    This function is kept for backward compatibility.
    New code should use ReductionOps.get_identity_value() directly.
    """
    return ReductionOps.get_identity_value(aten_op_name)


def _is_reduction_operation(aten_op_name: str) -> bool:
    """Check if an operation is a reduction operation.

    This function is kept for backward compatibility.
    New code should use ReductionOps.is_reduction() directly.
    """
    return ReductionOps.is_reduction(aten_op_name)


# Export the main classes and helper functions
__all__ = [
    "JaxKernel",
    "JaxOpImpl",
    "_infer_identity_value",
    "_is_reduction_operation",
]
