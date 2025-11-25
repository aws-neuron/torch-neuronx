"""JAX implementation of index_add operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    ["aten::index_add", "aten::index_add.out"], static_argnums=(1,), static_argnames=("alpha",)
)
def _aten_index_add(input, dim, index, source, *, alpha=1):
    """JAX implementation of torch.index_add and index_add_."""

    if alpha != 1:
        source = alpha * source
    if dim < 0:
        dim += input.ndim
    # Handle 0-d tensors
    zero_dim = input.ndim == 0
    if zero_dim:
        input = jnp.expand_dims(input, axis=0)
        dim = 0
    indices = tuple(index if i == dim else slice(None) for i in range(input.ndim))
    result = input.at[indices].add(source)
    return jnp.squeeze(result, axis=0) if zero_dim else result
