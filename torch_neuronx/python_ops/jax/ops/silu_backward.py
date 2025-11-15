import jax

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::silu_backward")
def _aten_silu_backward(grad_output, x):
    sigmoid_x = jax.nn.sigmoid(x)
    silu_derivative = sigmoid_x * (1 + x * (1 - sigmoid_x))
    result = grad_output * silu_derivative

    return result
