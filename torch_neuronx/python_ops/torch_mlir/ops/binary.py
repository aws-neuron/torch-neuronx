"""Binary arithmetic operations."""

import torch

from ..operation_registry import register_aten
from ..validation import validate_inplace_broadcast


@register_aten(["aten::add.Tensor", "aten::add.out", "aten::add_"], static_argnames=("alpha",))
def torch_add(x, y, alpha=1, out=None):
    """Add operation."""
    return torch.add(x, y, alpha=alpha)


@register_aten(
    ["aten::add.Scalar", "aten::add_.Scalar"],
    static_argnames=("alpha",),
)
def torch_add_scalar(x, y, alpha=1):
    """Add with scalar. Delegates to torch.add which handles type promotion
    checks.
    """
    return torch.add(x, y, alpha=alpha)


@register_aten(
    [
        "aten::mul.Scalar",
        "aten::mul.Tensor",
        "aten::mul.out",
        "aten::mul.Scalar_out",
        "aten::mul_",
    ]
)
def torch_mul(x, y, out=None):
    """Multiply operation."""
    return torch.mul(x, y)


@register_aten(["aten::sub.Tensor", "aten::sub.out"], static_argnames=("alpha",))
@register_aten(
    ["aten::sub_.Tensor"],
    static_argnames=("alpha",),
    validate_static_inputs=validate_inplace_broadcast,
)
def torch_sub(x, y, alpha=1, out=None):
    return torch.sub(x, y, alpha=alpha)


@register_aten(
    ["aten::div", "aten::div.out", "aten::div.out_mode"],
    static_argnames=("rounding_mode",),
)
@register_aten(
    ["aten::div_.Tensor"],
    static_argnames=("rounding_mode",),
    validate_static_inputs=validate_inplace_broadcast,
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
@register_aten(["aten::addcmul_"], validate_static_inputs=validate_inplace_broadcast)
def torch_addcmul(input, tensor1, tensor2, value=1, out=None):
    if input.dtype == torch.bool:
        raise RuntimeError("\"addcmul\" not implemented for 'Bool'")
    torch.broadcast_tensors(input, tensor1, tensor2)
    return input + value * tensor1 * tensor2


@register_aten(["aten::addcdiv", "aten::addcdiv.out"])
@register_aten(["aten::addcdiv_"], validate_static_inputs=validate_inplace_broadcast)
def torch_addcdiv(input, tensor1, tensor2, value=1, out=None):
    if not input.dtype.is_floating_point:
        raise RuntimeError(
            "Integer division with addcdiv is no longer supported, and in a future "
            "release addcdiv will perform a true division of tensor1 and tensor2. "
            "The historic addcdiv behavior can be implemented as "
            "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for "
            "integer inputs and as (input + value * tensor1 / tensor2) for float inputs. "
            "The future addcdiv behavior is just the latter implementation: "
            "(input + value * tensor1 / tensor2), for all dtypes."
        )
    torch.broadcast_tensors(input, tensor1, tensor2)
    return input + value * tensor1 / tensor2
