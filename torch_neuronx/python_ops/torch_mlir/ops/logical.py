"""Logical operations."""

import torch

from ..operation_registry import register_aten
from .unary import _validate_float_promoting_out


def maybe_cast_dtype(result, dtype):
    """Cast a result when a target dtype is provided.

    Args:
        result (torch.Tensor): Value to return or cast.
        dtype (torch.dtype or None): Target dtype, or ``None`` to return ``result`` unchanged.

    Returns:
        torch.Tensor: Original tensor or its dtype-converted value.
    """
    if dtype is not None:
        return result.to(dtype)
    else:
        return result


@register_aten(["aten::logical_and", "aten::logical_and.out"], static_argnames=("dtype",))
def torch_logical_and(x, y, out=None, dtype=None):
    """Compute the elementwise logical AND of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise logical AND result, optionally cast to ``dtype``.
    """
    result = torch.logical_and(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_or", "aten::logical_or.out"], static_argnames=("dtype",))
def torch_logical_or(x, y, out=None, dtype=None):
    """Compute the elementwise logical OR of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise logical OR result, optionally cast to ``dtype``.
    """
    result = torch.logical_or(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_not", "aten::logical_not.out"], static_argnames=("dtype",))
def torch_logical_not(x, out=None, dtype=None):
    """Compute the elementwise logical NOT of an input.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise logical NOT result, optionally cast to ``dtype``.
    """
    result = torch.logical_not(x)
    return maybe_cast_dtype(result, dtype)


@register_aten(["aten::logical_xor", "aten::logical_xor.out"], static_argnames=("dtype",))
def torch_logical_xor(x, y, out=None, dtype=None):
    """Compute the elementwise logical XOR of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise logical XOR result, optionally cast to ``dtype``.
    """
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
    """Compute the elementwise bitwise AND of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor or scalar): Second input.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Unused dtype placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise bitwise AND result.

    Note:
        Registered ``_`` overloads update the first input in place.
    """
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
    """Compute the elementwise bitwise OR of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor or scalar): Second input.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Unused dtype placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise bitwise OR result.

    Note:
        Registered ``_`` overloads update the first input in place.
    """
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
    """Compute the elementwise bitwise XOR of two inputs.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor or scalar): Second input.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise bitwise XOR result, optionally cast to ``dtype``.

    Note:
        Registered ``_`` overloads update the first input in place.
    """
    result = torch.bitwise_xor(x, y)
    return maybe_cast_dtype(result, dtype)


@register_aten(
    ["aten::bitwise_not", "aten::bitwise_not.out", "aten::bitwise_not.Tensor_out"],
    static_argnames=("dtype",),
)
def torch_bitwise_not(x, out=None, dtype=None):
    """Compute the elementwise bitwise NOT of an input.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.
        dtype (torch.dtype, optional): Unused dtype placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise bitwise NOT result.
    """
    return torch.bitwise_not(x)


def run_bitwise_shift_op(op_name, x, y):
    """Run a bitwise shift through the 32-bit lowering path.

    Args:
        op_name (str): Name of the ``torch`` bitwise shift function.
        x (torch.Tensor or scalar): Value to shift.
        y (torch.Tensor or scalar): Shift amount.

    Returns:
        torch.Tensor: Shift result cast to the dtype promoted from the original inputs.

    Note:
        Scalar operands become tensors because Torch-MLIR lacks the scalar
        variants. The lowering requires 32-bit operands, and scalar ``x`` is
        expanded to avoid MLIR result-shape inference failures. Promotion is
        computed before conversion so weak Python scalars do not widen results.
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
    """Shift input bits left by the specified amounts.

    Args:
        x (torch.Tensor or scalar): Value to shift.
        y (torch.Tensor or scalar): Shift amount.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Left-shifted result with the promoted input dtype.
    """
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
    """Shift input bits right by the specified amounts.

    Args:
        x (torch.Tensor or scalar): Value to shift.
        y (torch.Tensor or scalar): Shift amount.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Right-shifted result with the promoted input dtype.
    """
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
    """Apply the operator-form bitwise left shift.

    Args:
        x (torch.Tensor): Value to shift.
        y (torch.Tensor or scalar): Shift amount.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Left-shifted result with the promoted input dtype.
    """
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
    """Apply the operator-form bitwise right shift.

    Args:
        x (torch.Tensor): Value to shift.
        y (torch.Tensor or scalar): Shift amount.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Right-shifted result with the promoted input dtype.
    """
    return run_bitwise_shift_op("bitwise_right_shift", x, y)


@register_aten(["aten::signbit", "aten::signbit.out"])
def torch_signbit(x, out=None):
    """Test whether each input element has its sign bit set.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Boolean tensor indicating elements with a set sign bit.
    """
    return torch.signbit(x)


@register_aten(
    ["aten::sgn", "aten::sgn.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sgn(x, out=None):
    """Compute the elementwise sign of an input.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output buffer validated by registration but not
            forwarded by this implementation. Defaults to ``None``.

    Returns:
        torch.Tensor: Elementwise sign values.

    Raises:
        RuntimeError: If floating-point ``x`` has a non-floating-point ``out``.
    """
    return torch.sgn(x)
