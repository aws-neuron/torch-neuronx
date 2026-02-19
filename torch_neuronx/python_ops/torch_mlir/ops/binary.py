"""Binary arithmetic operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::add.Tensor", "aten::add.out", "aten::add_.Tensor"], static_argnames=("alpha",)
)
def torch_add(x, y, alpha=1, out=None):
    """Add operation."""
    return torch.add(x, y * alpha)


@register_aten(
    ["aten::mul.Scalar", "aten::mul.Tensor", "aten::mul.out", "aten::mul.Scalar_out", "aten::mul_"]
)
def torch_mul(x, y, out=None):
    """Multiply operation."""
    return torch.mul(x, y)


@register_aten(
    ["aten::sub.Tensor", "aten::sub.out", "aten::sub_.Tensor"], static_argnames=("alpha",)
)
def torch_sub(x, y, alpha=1, out=None):
    return torch.sub(x, y, alpha=alpha)


@register_aten(
    ["aten::div", "aten::div.out", "aten::div.out_mode", "aten::div_.Tensor"],
    static_argnames=("rounding_mode",),
)
def torch_div(x, y, rounding_mode=None, out=None):
    if rounding_mode is None:
        return x / y
    elif rounding_mode == "floor":
        result = x // y
    else:
        result = torch.trunc(x / y)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device=x.device, dtype=x.dtype)

    return torch.where(y == 0, torch.sign(x) * float("inf"), result)


@register_aten(
    ["aten::pow", "aten::pow.Tensor_Scalar_out", "aten::pow.Tensor_Tensor_out"], static_argnums=(1,)
)
def torch_pow(x, exponent, out=None):
    return torch.pow(x, exponent)


@register_aten(["aten::remainder", "aten::remainder.Tensor_out"])
def torch_remainder(x, y, out=None):
    return torch.remainder(x, y)


@register_aten(["aten::floor_divide", "aten::floor_divide.out"])
def torch_floor_divide(x, y, out=None):
    return torch.floor_divide(x, y)


@register_aten(["aten::addcmul", "aten::addcmul.out"])
def torch_addcmul(input, tensor1, tensor2, value=1, out=None):
    torch.broadcast_tensors(input, tensor1, tensor2)
    return input + value * tensor1 * tensor2


@register_aten(["aten::addcdiv", "aten::addcdiv.out"])
def torch_addcdiv(input, tensor1, tensor2, value=1, out=None):
    torch.broadcast_tensors(input, tensor1, tensor2)
    return input + value * tensor1 / tensor2
