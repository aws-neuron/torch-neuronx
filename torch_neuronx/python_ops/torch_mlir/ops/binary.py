"""Binary arithmetic operations."""

import torch

from ..operation_registry import register_aten
from ..validation import validate_inplace_broadcast


def _inplace_result(result, self_tensor):
    """Cast an in-place op's (promoted) result back to self's dtype.

    In-place ops do not type-promote and cannot resize self by broadcasting the
    other operand up. Mirror PyTorch's two rejections and the demotion:

      * broadcast that expands self (result shape != self shape) -> RuntimeError;
      * promoted result not castable to self (e.g. float -> int) -> RuntimeError;
      * otherwise cast the result to self's dtype (e.g. int64 -> int32).

    The final cast also keeps the graph output at self's dtype, so a wider
    (int64) operand doesn't leave a packed int32[..., 2] output whose logical
    shape then leaks (the inferred output dtype is self's, so the unpack never
    fires).
    """
    if isinstance(self_tensor, torch.Tensor):
        if tuple(result.shape) != tuple(self_tensor.shape):
            raise RuntimeError(
                f"output with shape {list(self_tensor.shape)} doesn't match "
                f"the broadcast shape {list(result.shape)}"
            )
        if not torch.can_cast(result.dtype, self_tensor.dtype):
            raise RuntimeError(
                f"result type {result.dtype} can't be cast to the desired "
                f"output type {self_tensor.dtype}"
            )
        return result.to(self_tensor.dtype)
    return result


def _div_compute(x, y, rounding_mode=None):
    """Core division shared by the out-of-place and in-place div handlers."""
    # Explicit casting for fp16 and bf16 to avoid accuracy mismatch
    up = isinstance(x, torch.Tensor) and x.dtype in (torch.float16, torch.bfloat16)
    xx = x.float() if up else x
    yy = y.float() if (up and isinstance(y, torch.Tensor)) else y

    if rounding_mode is None:
        # Lower bool true division directly. Converting an expanded bool input
        # during dispatch can otherwise lose the distinction between 0/0 and 1/0.
        if (
            isinstance(x, torch.Tensor)
            and isinstance(y, torch.Tensor)
            and x.dtype == torch.bool
            and y.dtype == torch.bool
        ):
            x_float = x.to(torch.get_default_dtype())
            inf = torch.full_like(x_float, float("inf"))
            nan = torch.full_like(x_float, float("nan"))
            return torch.where(y, x_float, torch.where(x, inf, nan))
        result = xx / yy
        return result.to(x.dtype) if up else result

    result = xx // yy if rounding_mode == "floor" else torch.trunc(xx / yy)

    if not isinstance(yy, torch.Tensor):
        yy = torch.tensor(yy, device=xx.device, dtype=xx.dtype)

    result = torch.where(yy == 0, torch.sign(xx) * float("inf"), result)
    return result.to(x.dtype) if up else result


@register_aten(["aten::add.Tensor", "aten::add.out"], static_argnames=("alpha",))
def torch_add(x, y, alpha=1, out=None):
    """Add operation."""
    return torch.add(x, y, alpha=alpha)


@register_aten(["aten::add_.Tensor"], static_argnames=("alpha",))
def torch_add_inplace(x, y, alpha=1, out=None):
    """In-place add — no type promotion; result follows self's dtype."""
    return _inplace_result(torch.add(x, y, alpha=alpha), x)


@register_aten(
    ["aten::add.Scalar", "aten::add_.Scalar"],
    static_argnames=("alpha",),
)
def torch_add_scalar(x, y, alpha=1):
    """Add with scalar. Delegates to torch.add which handles type promotion
    checks.
    """
    return torch.add(x, y, alpha=alpha)


@register_aten(["aten::mul.Scalar", "aten::mul.Tensor", "aten::mul.out", "aten::mul.Scalar_out"])
def torch_mul(x, y, out=None):
    """Multiply operation."""
    return torch.mul(x, y)


@register_aten(["aten::mul_.Tensor"])
def torch_mul_inplace(x, y, out=None):
    """In-place multiply — no type promotion; result follows self's dtype."""
    return _inplace_result(torch.mul(x, y), x)


@register_aten(["aten::sub.Tensor", "aten::sub.out"], static_argnames=("alpha",))
def torch_sub(x, y, alpha=1, out=None):
    return torch.sub(x, y, alpha=alpha)


@register_aten(
    ["aten::sub_.Tensor"],
    static_argnames=("alpha",),
    validate_static_inputs=validate_inplace_broadcast,
)
def torch_sub_inplace(x, y, alpha=1, out=None):
    """In-place sub — no type promotion; result follows self's dtype."""
    return _inplace_result(torch.sub(x, y, alpha=alpha), x)


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
    return _div_compute(x, y, rounding_mode)


@register_aten(["aten::div_.Tensor"], static_argnames=("rounding_mode",))
def torch_div_inplace(x, y, rounding_mode=None, out=None):
    """In-place divide — no type promotion; result follows self's dtype.

    Without this, a broadcasting divisor (e.g. [5, 1].div_([5]) -> [5, 5]) would
    try to write the broadcast result into the smaller self, tripping NRT
    "output0 too small". _inplace_result rejects broadcast-expand-self and
    enforces PyTorch's cast rule (e.g. int.div_(int) -> float is not castable
    to int, so it raises, matching PyTorch).
    """
    return _inplace_result(_div_compute(x, y, rounding_mode), x)


@register_aten(
    ["aten::pow", "aten::pow.Tensor_Scalar_out", "aten::pow.Tensor_Tensor_out"], static_argnums=(1,)
)
def torch_pow(x, exponent, out=None):
    if x.dtype == torch.bool and isinstance(exponent, bool):
        return x.clone() if exponent else torch.ones_like(x)

    # neuronx-cc has no 64-bit-integer `power` opcode, so an int64-promoted result
    # (e.g. `bool ** int` / `int ** int`) is computed in int32 and widened back.
    if torch.result_type(x, exponent) == torch.int64:
        exp = exponent.to(torch.int32) if isinstance(exponent, torch.Tensor) else exponent
        return torch.pow(x.to(torch.int32), exp).to(torch.int64)
    return torch.pow(x, exponent)


@register_aten(["aten::pow.Scalar", "aten::pow.Scalar_out"])
def torch_pow_scalar(self, exponent, out=None):
    """Scalar base raised to a tensor exponent (``scalar ** tensor``).

    Distinct from pow.Tensor_Scalar (tensor ** scalar): here the first arg is a
    Python scalar base and the second is the exponent tensor.
    """
    # neuronx-cc has no 64-bit-integer `power` opcode, so an int64-promoted result
    # (e.g. `int ** int`) is computed in int32 and widened back.
    if torch.result_type(self, exponent) == torch.int64:
        return torch.pow(self, exponent.to(torch.int32)).to(torch.int64)
    return torch.pow(self, exponent)


@register_aten(
    [
        "aten::remainder",
        "aten::remainder.Tensor_out",
        "aten::remainder.Scalar",
        "aten::remainder.Scalar_out",
        "aten::remainder_.Scalar",
    ]
)
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
