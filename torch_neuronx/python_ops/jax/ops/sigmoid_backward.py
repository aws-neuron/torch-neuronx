from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::sigmoid_backward")
def _aten_sigmoid_backward(grad_output, output):
    # output = sigmoid(x) i.e. output of forward
    result = grad_output * output * (1 - output)

    return result
