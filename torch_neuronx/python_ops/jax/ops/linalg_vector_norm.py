"""JAX implementations for linalg_vector_norm using function registration."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten
from torch_neuronx.python_ops.jax.type_converter import convert_dtype_with_default


@register_aten(
    ["aten::linalg_vector_norm", "aten::linalg_vector_norm.out"],
    static_argnums=(1, 2, 3),
    static_argnames=("dtype",),
)
def aten_linalg_vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
    # Convert dtype and cast input tensor if needed
    if dtype is not None:
        dtype = convert_dtype_with_default(dtype)
        self = self.astype(dtype)

    # Special case for negative ord values (except -inf)
    if ord < 0 and ord != float("-inf"):
        # Check for zeros along reduction dimensions
        has_zero = jnp.any(jnp.equal(self, 0), axis=dim, keepdims=keepdim)

        norm_result = jnp.linalg.vector_norm(self, ord=ord, axis=dim, keepdims=keepdim)

        # Replace with 0.0 where zeros exist in input along reduction dimensions
        result = jnp.where(has_zero, jnp.zeros_like(norm_result), norm_result)
    else:
        result = jnp.linalg.vector_norm(self, ord=ord, axis=dim, keepdims=keepdim)

    # Handle NaNs: eplace with NaN where input has NaN along reduction dimensions
    has_nan = jnp.any(jnp.isnan(self), axis=dim, keepdims=keepdim)
    result = jnp.where(has_nan, jnp.nan, result)

    return result
