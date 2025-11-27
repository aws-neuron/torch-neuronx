"""
JAX operation registry for torch-neuronx.

This module provides utilities to access JAX function implementations
for PyTorch ATen operations.
"""

from collections.abc import Callable

# Local registry of available JAX operations
_LOCAL_JAX_OPS = {}


def get_jax_function(aten_op_name: str) -> Callable | None:
    """
    Returns a callable JAX function that implements the given PyTorch ATen operation.

    Args:
        aten_op_name: String name of the ATen operation (e.g., "aten::add.Tensor")

    Returns:
        A callable JAX function that implements the ATen operation, or None if not found.
        The returned function can be called directly with JAX arrays.

    Example:
        >>> jax_add = get_jax_function("aten::add.Tensor")
        >>> if jax_add:
        >>>     import jax.numpy as jnp
        >>>     x = jnp.array([1, 2, 3])
        >>>     y = jnp.array([4, 5, 6])
        >>>     result = jax_add(x, y)  # JAX implementation of addition
    """
    # Look up by string
    if aten_op_name in _LOCAL_JAX_OPS:
        return _LOCAL_JAX_OPS[aten_op_name]

    print(f"Warning: Operation {aten_op_name} not found in local JAX registry")
    return None


def add_jax_operation(aten_op_name: str, jax_func: Callable):
    """
    Add a custom JAX implementation for an ATen operation.

    Args:
        aten_op_name: String name of the ATen operation
        jax_func: JAX function implementation
    """
    _LOCAL_JAX_OPS[aten_op_name] = jax_func
