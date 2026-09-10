"""Activation function operations."""

import torch

from ..operation_registry import register_aten
from .unary import _validate_float_promoting_out


@register_aten(["aten::relu", "aten::relu.out", "aten::relu_"])
def torch_relu(x, out=None):
    """ReLU activation."""
    return torch.relu(x)


@register_aten(["aten::gelu", "aten::gelu_", "aten::gelu.out"], static_argnames=("approximate",))
def torch_gelu(x, approximate="none", out=None):
    return torch.nn.functional.gelu(x, approximate=approximate)


@register_aten(["aten::silu", "aten::silu.out"])
def torch_silu(x, out=None):
    return torch.nn.functional.silu(x)


@register_aten(
    ["aten::sigmoid", "aten::sigmoid.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sigmoid(x, out=None):
    return torch.sigmoid(x)


@register_aten(
    ["aten::tanh", "aten::tanh.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_tanh(self, out=None):
    return torch.tanh(self)


@register_aten(["aten::softplus", "aten::softplus.out"], static_argnums=(1, 2))
def torch_softplus(self, beta=1, threshold=20, out=None):
    return torch.nn.functional.softplus(self, beta=beta, threshold=threshold)


@register_aten(
    ["aten::sigmoid_backward", "aten::sigmoid_backward.grad_input"], output_params=("grad_input",)
)
def torch_sigmoid_backward(grad_output, output, grad_input=None):
    return torch.ops.aten.sigmoid_backward.default(grad_output, output)


@register_aten(
    ["aten::silu_backward", "aten::silu_backward.grad_input"], output_params=("grad_input",)
)
def torch_silu_backward(grad_output, x, grad_input=None):
    return torch.ops.aten.silu_backward.default(grad_output, x)


@register_aten(
    ["aten::tanh_backward", "aten::tanh_backward.grad_input"], output_params=("grad_input",)
)
def torch_tanh_backward(grad_output, output, grad_input=None):
    return torch.ops.aten.tanh_backward(grad_output, output)


@register_aten(
    ["aten::gelu_backward", "aten::gelu_backward.grad_input"],
    static_argnames=("approximate",),
    output_params=("grad_input",),
)
def torch_gelu_backward(grad_output, self, approximate="none", grad_input=None):
    return torch.ops.aten.gelu_backward(grad_output, self, approximate=approximate)


@register_aten(
    ["aten::softplus_backward", "aten::softplus_backward.grad_input"],
    static_argnums=(2, 3),
    output_params=("grad_input",),
)
def torch_softplus_backward(out_grad, x, beta, threshold, grad_input=None):
    return torch.ops.aten.softplus_backward(out_grad, x, beta=beta, threshold=threshold)


@register_aten(
    ["aten::log_sigmoid_forward", "aten::log_sigmoid_forward.output"],
    output_params=("output", "buffer"),
)
def torch_log_sigmoid_forward(self, output=None, buffer=None):
    return torch.ops.aten.log_sigmoid_forward(self)


@register_aten(
    ["aten::log_sigmoid_backward", "aten::log_sigmoid_backward.grad_input"],
    output_params=("grad_input",),
)
def torch_log_sigmoid_backward(grad_output, self, buffer, grad_input=None):
    return torch.ops.aten.log_sigmoid_backward(grad_output, self, buffer)


@register_aten(["aten::celu", "aten::celu.out"], static_argnames=("alpha",))
def torch_celu(self, alpha=1.0, out=None):
    return torch.ops.aten.celu(self, alpha)


@register_aten(["aten::leaky_relu", "aten::leaky_relu.out"], static_argnames=("negative_slope",))
def torch_leaky_relu(self, negative_slope=0.01, out=None):
    return torch.ops.aten.leaky_relu(self, negative_slope)


@register_aten(["aten::leaky_relu_"], static_argnames=("negative_slope",))
def torch_leaky_relu_(self, negative_slope=0.01):
    return torch.ops.aten.leaky_relu(self, negative_slope)


@register_aten(["aten::hardsigmoid", "aten::hardsigmoid.out"])
def torch_hardsigmoid(self, out=None):
    return torch.ops.aten.hardsigmoid(self)


@register_aten(["aten::hardsigmoid_"])
def torch_hardsigmoid_(self):
    return torch.ops.aten.hardsigmoid(self)


@register_aten(
    ["aten::hardsigmoid_backward", "aten::hardsigmoid_backward.grad_input"],
    output_params=("grad_input",),
)
def torch_hardsigmoid_backward(grad_output, self, grad_input=None):
    return torch.ops.aten.hardsigmoid_backward.default(grad_output, self)
