"""Activation function operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::relu", "aten::relu.out", "aten::relu_"])
def torch_relu(x, out=None):
    """ReLU activation."""
    return torch.relu(x)


@register_aten(["aten::gelu", "aten::gelu_", "aten::gelu.out"])
def torch_gelu(x, approximate="none", out=None):
    return torch.nn.functional.gelu(x, approximate=approximate)


@register_aten(["aten::silu", "aten::silu.out"])
def torch_silu(x, out=None):
    return torch.nn.functional.silu(x)


@register_aten(["aten::sigmoid", "aten::sigmoid.out"])
def torch_sigmoid(x, out=None):
    return torch.sigmoid(x)


@register_aten(["aten::tanh", "aten::tanh.out"])
def torch_tanh(self, out=None):
    return torch.tanh(self)


@register_aten(["aten::softplus", "aten::softplus.out"], static_argnums=(1, 2))
def torch_softplus(self, beta=1, threshold=20):
    return torch.nn.functional.softplus(self, beta=beta, threshold=threshold)


@register_aten(["aten::sigmoid_backward", "aten::sigmoid_backward.grad_input"])
def torch_sigmoid_backward(grad_output, output):
    return torch.ops.aten.sigmoid_backward.default(grad_output, output)


@register_aten(["aten::silu_backward"])
def torch_silu_backward(grad_output, x):
    return torch.ops.aten.silu_backward.default(grad_output, x)


@register_aten(["aten::tanh_backward"])
def torch_tanh_backward(grad_output, output):
    return torch.ops.aten.tanh_backward(grad_output, output)


@register_aten(["aten::gelu_backward", "aten::gelu_backward.grad_input"])
def torch_gelu_backward(grad_output, self, approximate="none"):
    return torch.ops.aten.gelu_backward(grad_output, self, approximate=approximate)


@register_aten(
    ["aten::softplus_backward", "aten::softplus_backward.grad_input"], static_argnums=(2, 3)
)
def torch_softplus_backward(out_grad, x, beta, threshold):
    return torch.ops.aten.softplus_backward(out_grad, x, beta=beta, threshold=threshold)
