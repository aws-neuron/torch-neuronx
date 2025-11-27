import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::index_select_backward", static_argnums=(1, 2))
def _aten_index_select_backward(grad_output, input_sizes, dim, index):
    grad_input = jnp.zeros(input_sizes, dtype=grad_output.dtype)

    indices = [slice(None)] * len(input_sizes)
    indices[dim] = index

    return grad_input.at[tuple(indices)].add(grad_output)
