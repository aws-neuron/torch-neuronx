import jax
import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    ["aten::_fused_rms_norm"],
    static_argnums=(1, 3),
)
def _aten_fused_rms_norm(input, normalized_shape, weight=None, eps=1e-5):
    original_dtype = input.dtype
    input = input.astype(jnp.float32)
    dims_to_reduce = tuple(input.ndim - i - 1 for i in range(len(normalized_shape)))
    variance = jnp.mean(jnp.square(input), axis=dims_to_reduce, keepdims=True)
    rstd = jax.lax.rsqrt(variance + eps)
    result = input * rstd

    if weight is not None:
        if weight.dtype in (jnp.float16, jnp.bfloat16):
            weight = weight.astype(jnp.float32)
        result = result * weight
    return result.astype(original_dtype), rstd


@register_aten(["aten::_fused_rms_norm_backward"], static_argnums=(2, 5))
def _aten_fused_rms_norm_backward(
    grad_out,
    input,
    normalized_shape,
    rstd,
    weight,
    output_mask,
):
    original_input_dtype = input.dtype
    original_weight_dtype = weight.dtype if weight is not None else None
    # Upcast to fp32 for computation
    grad_out = grad_out.astype(jnp.float32)
    input = input.astype(jnp.float32)
    if weight is not None:
        weight = weight.astype(jnp.float32)

    # Compute dimensions
    axis = input.ndim - len(normalized_shape)
    inner_dim_indices = tuple(range(axis, input.ndim))
    outer_dim_indices = tuple(range(axis))
    prod_dim = jnp.prod(jnp.array(input.shape[axis:]))

    # Compute x_hat and grad_x_hat
    x_hat = input * rstd
    grad_x_hat = grad_out * weight if weight is not None else grad_out

    # Gradient w.r.t. input
    grad_input = None
    if output_mask[0]:
        sum_val = jnp.sum(x_hat * grad_x_hat, axis=inner_dim_indices, keepdims=True)
        grad_input = (grad_x_hat - (x_hat / prod_dim) * sum_val) * rstd
        grad_input = grad_input.astype(original_input_dtype)

    # Gradient w.r.t. weight
    grad_weight = None
    if output_mask[1] and weight is not None:
        grad_weight_full = grad_out * x_hat
        if len(outer_dim_indices) > 0:
            grad_weight = jnp.sum(grad_weight_full, axis=outer_dim_indices)
        else:
            grad_weight = grad_weight_full
        grad_weight = grad_weight.astype(original_weight_dtype)

    return grad_input, grad_weight
