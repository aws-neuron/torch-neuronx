"""
Type promotion utilities for torch-neuronx JAX operations.

This module provides centralized type promotion functions to avoid repeating
similar code across operations.
"""

from typing import Any

import jax.numpy as jnp


def promote_types(*values: Any) -> tuple[Any, ...]:
    """
    Promote multiple values to a common dtype using JAX's type promotion rules.

    Args:
        *values: Variable number of JAX arrays or scalar values to promote

    Returns:
        Tuple of promoted values with matching dtypes

    Example:
        >>> a = jnp.array([1, 2, 3], dtype=jnp.int32)
        >>> b = 2.5  # float scalar
        >>> a_promoted, b_promoted = promote_types(a, b)
        >>> # Both will have dtype float32
    """
    if len(values) == 0:
        return ()

    if len(values) == 1:
        return values

    # Determine common dtype using JAX's type promotion rules
    common_dtype = jnp.result_type(*values)

    # Convert all values to the common dtype
    promoted_values = []
    for value in values:
        if hasattr(value, "astype"):
            # JAX array - use astype
            promoted_values.append(value.astype(common_dtype))
        else:
            # Scalar - convert to JAX array with the common dtype
            promoted_values.append(jnp.array(value, dtype=common_dtype))

    return tuple(promoted_values)


def promote_binary_op(lhs: Any, rhs: Any) -> tuple[Any, Any]:
    """
    Promote two values for binary operations.

    This is a convenience function for the common case of binary operations.
    Equivalent to torch_xla's std::tie(lhs, rhs) = XlaHelpers::Promote(lhs, rhs).

    Args:
        lhs: Left operand (JAX array or scalar)
        rhs: Right operand (JAX array or scalar)

    Returns:
        Tuple of (promoted_lhs, promoted_rhs) with matching dtypes
    """
    return promote_types(lhs, rhs)


def promote_ternary_op(a: Any, b: Any, c: Any) -> tuple[Any, Any, Any]:
    """
    Promote three values for ternary operations.

    This is a convenience function for operations with three inputs like addmm.

    Args:
        a: First operand (JAX array or scalar)
        b: Second operand (JAX array or scalar)
        c: Third operand (JAX array or scalar)

    Returns:
        Tuple of (promoted_a, promoted_b, promoted_c) with matching dtypes
    """
    return promote_types(a, b, c)


def get_common_dtype(*values: Any) -> jnp.dtype:
    """
    Get the common dtype for multiple values without converting them.

    This is useful when you need to know the promoted dtype without
    actually performing the conversion.

    Args:
        *values: Variable number of JAX arrays or scalar values

    Returns:
        The common dtype that would result from promotion
    """
    return jnp.result_type(*values)
