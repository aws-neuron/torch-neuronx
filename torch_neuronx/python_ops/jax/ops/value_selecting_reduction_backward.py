import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::value_selecting_reduction_backward", static_argnums=(1, 3, 4))
def _aten_value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    """
    Backward function for value selecting reduction operations (eg: max, min, topk, mode).
    Grad is propagated to specific locations based on indices.
    """
    # recreate original input shape for grad
    grad_input = jnp.zeros(sizes, dtype=grad.dtype)

    if dim < 0:
        dim = dim + len(sizes)
    # expand grad if needed
    if not keepdim:
        grad = jnp.expand_dims(grad, axis=dim)
        indices = jnp.expand_dims(indices, axis=dim)
    coords = []
    for d in range(len(sizes)):
        if d == dim:
            coord = indices.flatten()
        else:
            shape_d = [1] * len(grad.shape)
            shape_d[d] = grad.shape[d]
            coord_d = jnp.arange(grad.shape[d]).reshape(shape_d)
            coord_d = jnp.broadcast_to(coord_d, grad.shape)
            coord = coord_d.flatten()
        coords.append(coord)

    coords = jnp.stack(coords, axis=0)
    # Scatter grads to selected indices
    grad_input = grad_input.at[tuple(coords)].add(grad.flatten())

    return grad_input
