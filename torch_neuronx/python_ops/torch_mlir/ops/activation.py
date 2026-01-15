"""Activation function operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::relu", "aten::relu.out", "aten::relu_"])
def torch_relu(x, out=None):
    """ReLU activation."""
    return torch.relu(x)


@register_aten(["aten::gelu", "aten::gelu_", "aten::gelu.out"])
def torch_gelu(x, approximate="none"):
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


@register_aten(["aten::sigmoid_backward", "aten::sigmoid_backward.grad_input"])
def torch_sigmoid_backward(grad_output, output):
    return torch.ops.aten.sigmoid_backward.default(grad_output, output)


@register_aten(["aten::silu_backward"])
def torch_silu_backward(grad_output, x):
    return grad_output * torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))


@register_aten(["aten::tanh_backward"])
def torch_tanh_backward(grad_output, output):
    return torch.ops.aten.tanh_backward(grad_output, output)
