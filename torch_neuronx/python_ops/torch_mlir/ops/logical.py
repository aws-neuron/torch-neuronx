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
    ["aten::bitwise_and", "aten::bitwise_and.out", "aten::bitwise_and.Tensor_out"],
    static_argnames=("dtype",),
)
def torch_bitwise_and(x, y, out=None, dtype=None):
    return torch.bitwise_and(x, y)


@register_aten(
    ["aten::bitwise_or", "aten::bitwise_or.out", "aten::bitwise_or.Tensor_out"],
    static_argnames=("dtype",),
)
def torch_bitwise_or(x, y, out=None, dtype=None):
    return torch.bitwise_or(x, y)


@register_aten(
    ["aten::bitwise_xor", "aten::bitwise_xor.out", "aten::bitwise_xor.Tensor_out"],
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

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    if not isinstance(x, torch.Tensor):
        x = torch.full_like(y, x)

    # Handle scalar tensor broadcasting
    if x.dim() == 0 and y.dim() > 0:
        x = x.expand_as(y)

    result_dtype = x.dtype
    x = x.to(torch.int32)
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


@register_aten(["aten::signbit", "aten::signbit.out"])
def torch_signbit(x, out=None):
    return torch.signbit(x)


@register_aten(["aten::sgn", "aten::sgn.out"])
def torch_sgn(x, out=None):
    return torch.sgn(x)
