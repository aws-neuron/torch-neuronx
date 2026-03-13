import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::slice_backward", static_argnums=(1, 2, 3, 4, 5))
def _aten_slice_backward(grad_output, input_sizes, dim, start, end, step):
    grad_input = jnp.zeros(input_sizes, dtype=grad_output.dtype)

    indices = [slice(None)] * len(input_sizes)
    indices[dim] = slice(start, end, step)

    return grad_input.at[tuple(indices)].set(grad_output)
