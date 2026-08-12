"""Logical operations."""

import torch

from ..operation_registry import register_aten


def maybe_cast_dtype(result, dtype):
    if dtype is not None:
        return result.to(dtype)
    else:
        return result


@register_aten(["aten::logical_and", "aten::logical_and.out"], static_argnames=("dtype",))
def torch_logical_and(x, y, out=None, dtype=None):
    result = torch.logical_and(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_or", "aten::logical_or.out"], static_argnames=("dtype",))
def torch_logical_or(x, y, out=None, dtype=None):
    result = torch.logical_or(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_not", "aten::logical_not.out"], static_argnames=("dtype",))
def torch_logical_not(x, out=None, dtype=None):
    result = torch.logical_not(x)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_xor", "aten::logical_xor.out"], static_argnames=("dtype",))
def torch_logical_xor(x, y, out=None, dtype=None):
    result = torch.logical_xor(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(
    [
        "aten::bitwise_and",
        "aten::bitwise_and.out",
        "aten::bitwise_and.Tensor_out",
        "aten::bitwise_and.Scalar",
        "aten::bitwise_and.Scalar_out",
        "aten::bitwise_and_.Scalar",
    ],
    static_argnames=("dtype",),
)
def torch_bitwise_and(x, y, out=None, dtype=None):
    return torch.bitwise_and(x, y)


@register_aten(
    [
        "aten::bitwise_or",
        "aten::bitwise_or.out",
        "aten::bitwise_or.Tensor_out",
        "aten::bitwise_or.Scalar",
        "aten::bitwise_or.Scalar_out",
        "aten::bitwise_or_.Scalar",
    ],
    static_argnames=("dtype",),
)
def torch_bitwise_or(x, y, out=None, dtype=None):
    return torch.bitwise_or(x, y)


@register_aten(
    [
        "aten::bitwise_xor",
        "aten::bitwise_xor.out",
        "aten::bitwise_xor.Tensor_out",
        "aten::bitwise_xor.Scalar",
        "aten::bitwise_xor.Scalar_out",
        "aten::bitwise_xor_.Scalar",
    ],
    static_argnames=("dtype",),
)
def torch_bitwise_xor(x, y, out=None, dtype=None):
    result = torch.bitwise_xor(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(
    ["aten::bitwise_not", "aten::bitwise_not.out", "aten::bitwise_not.Tensor_out"],
    static_argnames=("dtype",),
)
def torch_bitwise_not(x, out=None, dtype=None):
    return torch.bitwise_not(x)


def run_bitwise_shift_op(op_name, x, y):
    """
    1. Tensor_Scalar and Scalar_Tensor variants are not registered in torch_mlir
       need to convert scalar to Tensor
       https://github.com/llvm/torch-mlir/blob/cb0f5dcc0dfe0527bb2718066f96d6bbaa38245f/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td#L3221
    2. neuronxcc requires operand to be 32 bit for bitwise shift op
    3. When x is a scalar, mlir fails to broadcast the shape which causes
       LLVM ERROR: Failed to infer result type(s)
    """
    op = getattr(torch, op_name)

    # Shifts type-promote like any other binary op, so the result dtype comes
    # from both operands -- e.g. int8 >> int64 is int64 and uint8 >> int8 is
    # int16 on CPU. Using x.dtype here would silently narrow the result.
    #
    # This must be computed from the ORIGINAL arguments, before the scalar ->
    # tensor conversions below: result_type treats a Python int as a weak scalar
    # (int32 >> 3 stays int32) but a 0-dim int64 tensor as a real operand
    # (int32 >> tensor(3) promotes to int64). Computing it after the conversion
    # widened 0-dim scalar shifts to int64, which broke the packed-int64 paths
    # that build their high word with bitwise_right_shift(low, 31).
    result_dtype = torch.result_type(x, y)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    if not isinstance(x, torch.Tensor):
        x = torch.full_like(y, x)

    # Handle scalar tensor broadcasting
    if x.dim() == 0 and y.dim() > 0:
        x = x.expand_as(y)

    x = x.to(torch.int32)
    if isinstance(y, torch.Tensor):
        y = y.to(torch.int32)
    result = op(x, y)
    return result.to(result_dtype)


@register_aten(
    [
        "aten::bitwise_left_shift.Tensor",
        "aten::bitwise_left_shift.Tensor_out",
        "aten::bitwise_left_shift.Tensor_Scalar",
        "aten::bitwise_left_shift.Tensor_Scalar_out",
        "aten::bitwise_left_shift.Scalar_Tensor",
        "aten::bitwise_left_shift.Scalar_Tensor_out",
    ],
)
def torch_bitwise_left_shift(x, y, out=None):
    return run_bitwise_shift_op("bitwise_left_shift", x, y)


@register_aten(
    [
        "aten::bitwise_right_shift.Tensor",
        "aten::bitwise_right_shift.Tensor_out",
        "aten::bitwise_right_shift.Tensor_Scalar",
        "aten::bitwise_right_shift.Tensor_Scalar_out",
        "aten::bitwise_right_shift.Scalar_Tensor",
        "aten::bitwise_right_shift.Scalar_Tensor_out",
    ],
)
def torch_bitwise_right_shift(x, y, out=None):
    return run_bitwise_shift_op("bitwise_right_shift", x, y)


# aten::__lshift__ and aten::__rshift__ are the Python operator-form overloads
# of the bitwise shifts (`x << y` and `x >> y`). They are semantically
# identical to bitwise_left_shift / bitwise_right_shift for integer tensors,
# so we delegate to the same implementation. All four schema variants are
# registered -- leaving the `_out` ones unregistered sends them through the
# fallback path, which aliases `self` to `out` and corrupts `self`.
@register_aten(
    [
        "aten::__lshift__.Tensor",
        "aten::__lshift__.Scalar",
        "aten::__lshift__.Tensor_out",
        "aten::__lshift__.Scalar_out",
    ],
)
def torch___lshift__(x, y, out=None):
    return run_bitwise_shift_op("bitwise_left_shift", x, y)


@register_aten(
    [
        "aten::__rshift__.Tensor",
        "aten::__rshift__.Scalar",
        "aten::__rshift__.Tensor_out",
        "aten::__rshift__.Scalar_out",
    ],
)
def torch___rshift__(x, y, out=None):
    return run_bitwise_shift_op("bitwise_right_shift", x, y)


@register_aten(["aten::signbit", "aten::signbit.out"])
def torch_signbit(x, out=None):
    return torch.signbit(x)


@register_aten(["aten::sgn", "aten::sgn.out"])
def torch_sgn(x, out=None):
    return torch.sgn(x)
