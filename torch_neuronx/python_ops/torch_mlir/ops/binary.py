"""Binary arithmetic operations."""

import torch

from ..operation_registry import register_aten
from ..validation import (
    check_bool_sub,
    validate_inplace_broadcast,
    validate_true_divide_out,
)
from .unary import _validate_float_promoting_out


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

    # Integer floor/trunc division must be exact, so compute it in the integer
    # domain. A float divide cannot represent an int64 operand: the value loses
    # its high word on the way to fp32 and the 24-bit mantissa cannot hold it,
    # so div(2**63-2, 1) came back as -2. ``//`` lowers to an exact integer
    # floor division (TorchToStablehlo emits divide + remainder), and trunc is
    # derived from it. Floats keep the float path.
    _x_int = isinstance(x, torch.Tensor) and not x.dtype.is_floating_point and x.dtype != torch.bool
    _y_int = (not isinstance(y, torch.Tensor)) or (
        not y.dtype.is_floating_point and y.dtype != torch.bool
    )
    if _x_int and _y_int:
        yt = (
            yy
            if isinstance(yy, torch.Tensor)
            else torch.tensor(yy, device=xx.device, dtype=xx.dtype)
        )
        quot = xx // yt
        if rounding_mode == "floor":
            return quot
        # floor rounds toward -inf and trunc toward zero, so step the quotient
        # back up by one when it is negative and the division is not exact.
        rem = xx - quot * yt
        negative_quot = (xx < 0) != (yt < 0)
        return quot + ((rem != 0) & negative_quot).to(quot.dtype)
    else:
        result = xx // yy if rounding_mode == "floor" else torch.trunc(xx / yy)

    if not isinstance(yy, torch.Tensor):
        yy = torch.tensor(yy, device=xx.device, dtype=xx.dtype)

    result = torch.where(yy == 0, torch.sign(xx) * float("inf"), result)
    return result.to(x.dtype) if up else result


@register_aten(
    ["aten::add.Tensor", "aten::add.out"],
    static_argnames=("alpha",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_add(x, y, alpha=1, out=None):
    """Add two operands with an optional scale factor.

    Args:
        x (torch.Tensor): First operand.
        y (torch.Tensor): Second operand.
        alpha (Number, optional): Scale factor applied to ``y``. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Sum of ``x`` and ``alpha * y``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If floating-point ``x`` is paired with a non-floating ``out``.
    """
    return torch.add(x, y, alpha=alpha)


@register_aten(["aten::add_.Tensor"], static_argnames=("alpha",))
def torch_add_inplace(x, y, alpha=1, out=None):
    """Compute an in-place-style add result in the input dtype.

    Args:
        x (torch.Tensor): Tensor whose shape and dtype constrain the result.
        y (torch.Tensor): Value to add.
        alpha (Number, optional): Scale factor applied to ``y``. Defaults to ``1``.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Add result matching the shape and dtype of ``x``.

    Raises:
        RuntimeError: If broadcasting expands ``x`` or the result cannot be cast to its dtype.

    Note:
        Registered dispatch aliases the result to ``x`` and does not type-promote.
    """
    return _inplace_result(torch.add(x, y, alpha=alpha), x)


@register_aten(
    ["aten::add.Scalar", "aten::add_.Scalar"],
    static_argnames=("alpha",),
)
def torch_add_scalar(x, y, alpha=1):
    """Add a scaled scalar to a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        y (Number): Scalar to add.
        alpha (Number, optional): Scale factor applied to ``y``. Defaults to ``1``.

    Returns:
        torch.Tensor: Element-wise sum of ``x`` and ``alpha * y``.

    Note:
        ``torch.add`` performs type-promotion checks; registered ``aten::add_.Scalar``
        dispatch aliases the result to ``x``.
    """
    return torch.add(x, y, alpha=alpha)


@register_aten(
    ["aten::mul.Scalar", "aten::mul.Tensor", "aten::mul.out", "aten::mul.Scalar_out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_mul(x, y, out=None):
    """Multiply two operands.

    Args:
        x (torch.Tensor): First operand.
        y (torch.Tensor or Number): Second operand.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Element-wise product of ``x`` and ``y``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If floating-point ``x`` is paired with a non-floating ``out``.
    """
    return torch.mul(x, y)


@register_aten(["aten::mul_.Tensor", "aten::mul_.Scalar"])
def torch_mul_inplace(x, y, out=None):
    """Compute an in-place-style product in the input dtype.

    Args:
        x (torch.Tensor): Tensor whose shape and dtype constrain the result.
        y (torch.Tensor or Number): Value to multiply.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Product matching the shape and dtype of ``x``.

    Raises:
        RuntimeError: If broadcasting expands ``x`` or the result cannot be cast to its dtype.

    Note:
        Registered dispatch aliases the result to ``x`` and does not type-promote.
    """
    return _inplace_result(torch.mul(x, y), x)


@register_aten(
    ["aten::sub.Tensor", "aten::sub.out"],
    static_argnames=("alpha",),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_sub(x, y, alpha=1, out=None):
    """Subtract a scaled operand from a tensor.

    Args:
        x (torch.Tensor): Minuend tensor.
        y (torch.Tensor or Number): Subtrahend.
        alpha (Number, optional): Scale factor applied to ``y``. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Difference ``x - alpha * y``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If either operand is Boolean or floating-point ``x`` is paired with a
            non-floating ``out``.
    """
    check_bool_sub(x, y)
    return torch.sub(x, y, alpha=alpha)


@register_aten(
    ["aten::sub_.Tensor"],
    static_argnames=("alpha",),
    validate_static_inputs=validate_inplace_broadcast,
)
def torch_sub_inplace(x, y, alpha=1, out=None):
    """Compute an in-place-style difference in the input dtype.

    Args:
        x (torch.Tensor): Tensor whose shape and dtype constrain the result.
        y (torch.Tensor or Number): Value to subtract.
        alpha (Number, optional): Scale factor applied to ``y``. Defaults to ``1``.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Difference matching the shape and dtype of ``x``.

    Raises:
        RuntimeError: If either operand is Boolean, broadcasting expands ``x``, or
            the result cannot be cast to its dtype.

    Note:
        Registered dispatch aliases the result to ``x`` and does not type-promote.
    """
    check_bool_sub(x, y)
    return _inplace_result(torch.sub(x, y, alpha=alpha), x)


def _validate_div_out(x, y=None, rounding_mode=None, out=None, **kwargs):
    """Apply both out= rules for div: the float-result rule and the cast rule."""
    _validate_float_promoting_out(x, y, rounding_mode=rounding_mode, out=out, **kwargs)
    validate_true_divide_out(x, y, rounding_mode=rounding_mode, out=out)


@register_aten(
    ["aten::div", "aten::div.out", "aten::div.out_mode"],
    static_argnames=("rounding_mode",),
    validate_static_inputs=_validate_div_out,
)
@register_aten(
    ["aten::div_.Tensor"],
    static_argnames=("rounding_mode",),
    validate_static_inputs=validate_inplace_broadcast,
)
def torch_div(x, y, rounding_mode=None, out=None):
    """Divide two operands with optional rounding.

    Args:
        x (torch.Tensor): Dividend tensor.
        y (torch.Tensor or Number): Divisor.
        rounding_mode (str, optional): ``None`` for true division, or
            ``"floor"``/``"trunc"``. Defaults to ``None``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Element-wise quotient; aliases ``out`` when provided.

    Raises:
        RuntimeError: If ``out`` cannot hold the validated result dtype.

    Note:
        ``torch.float16`` and ``torch.bfloat16`` ``x`` values are computed in ``torch.float32``
        and cast back. Registered ``aten::div_.Tensor`` dispatch aliases the result to ``x``.
    """
    return _div_compute(x, y, rounding_mode)


@register_aten(["aten::div_.Tensor"], static_argnames=("rounding_mode",))
def torch_div_inplace(x, y, rounding_mode=None, out=None):
    """Compute an in-place-style quotient in the input dtype.

    Args:
        x (torch.Tensor): Tensor whose shape and dtype constrain the result.
        y (torch.Tensor or Number): Divisor.
        rounding_mode (str, optional): ``None`` for true division, or
            ``"floor"``/``"trunc"``. Defaults to ``None``.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Quotient matching the shape and dtype of ``x``.

    Raises:
        RuntimeError: If broadcasting expands ``x`` or the quotient cannot be cast to its dtype.

    Note:
        Registered dispatch aliases the result to ``x`` without type promotion. Rejecting
        broadcast expansion avoids an undersized NRT output buffer.
    """
    return _inplace_result(_div_compute(x, y, rounding_mode), x)


@register_aten(
    ["aten::pow", "aten::pow.Tensor_Scalar_out", "aten::pow.Tensor_Tensor_out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_pow(x, exponent, out=None):
    """Raise a tensor to a scalar or tensor exponent.

    Args:
        x (torch.Tensor): Base tensor.
        exponent (torch.Tensor or Number): Exponent.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Element-wise power result; aliases ``out`` when provided.

    Note:
        Boolean bases receive explicit Boolean-exponent handling. ``torch.int64`` results are
        computed in ``torch.int32`` and widened because neuronx-cc has no 64-bit ``power`` opcode.
    """
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
    """Raise a scalar base to a tensor exponent.

    Args:
        self (Number): Scalar base.
        exponent (torch.Tensor): Exponent tensor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Element-wise ``self ** exponent`` result; aliases ``out`` when provided.

    Note:
        Unlike ``pow.Tensor_Scalar``, this handler receives a scalar base and tensor exponent.
        ``torch.int64`` results are computed in ``torch.int32`` and widened.
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
    ],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_remainder(x, y, out=None):
    """Compute the element-wise remainder.

    Args:
        x (torch.Tensor): Dividend tensor.
        y (torch.Tensor or Number): Divisor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Remainder of division by ``y``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If floating-point ``x`` is paired with a non-floating ``out``.

    Note:
        Registered ``aten::remainder_.Scalar`` dispatch aliases the result to ``x``.
    """
    return torch.remainder(x, y)


@register_aten(
    ["aten::floor_divide", "aten::floor_divide.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_floor_divide(x, y, out=None):
    """Floor-divide two operands.

    Args:
        x (torch.Tensor): Dividend tensor.
        y (torch.Tensor or Number): Divisor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Element-wise floor quotient; aliases ``out`` when provided.

    Raises:
        RuntimeError: If floating-point ``x`` is paired with a non-floating ``out``.
    """
    return torch.floor_divide(x, y)


@register_aten(
    ["aten::addcmul", "aten::addcmul.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
@register_aten(["aten::addcmul_"], validate_static_inputs=validate_inplace_broadcast)
def torch_addcmul(input, tensor1, tensor2, value=1, out=None):
    """Add a scaled element-wise product to an input tensor.

    Args:
        input (torch.Tensor): Tensor to which the product is added.
        tensor1 (torch.Tensor): First multiplicand.
        tensor2 (torch.Tensor): Second multiplicand.
        value (Number, optional): Scale converted to ``input.dtype`` before multiplication.
            Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Scaled-product sum after converting ``value``; aliases ``out`` when
            provided.

    Raises:
        RuntimeError: If ``input`` is Boolean, operands cannot broadcast, or floating-point
            ``input`` is paired with a non-floating ``out``.

    Note:
        Registered ``aten::addcmul_`` dispatch aliases the result to ``input``.
    """
    if input.dtype == torch.bool:
        raise RuntimeError("\"addcmul\" not implemented for 'Bool'")
    torch.broadcast_tensors(input, tensor1, tensor2)
    # Cast value to input dtype so a float scalar doesn't promote integer tensors.
    value_t = torch.tensor(value, dtype=input.dtype, device=input.device)
    return input + value_t * tensor1 * tensor2


@register_aten(
    ["aten::addcdiv", "aten::addcdiv.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
@register_aten(["aten::addcdiv_"], validate_static_inputs=validate_inplace_broadcast)
def torch_addcdiv(input, tensor1, tensor2, value=1, out=None):
    """Add a scaled element-wise quotient to an input tensor.

    Args:
        input (torch.Tensor): Tensor to which the quotient is added.
        tensor1 (torch.Tensor): Dividend tensor.
        tensor2 (torch.Tensor): Divisor tensor.
        value (Number, optional): Scale factor for the quotient. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Result of ``input + value * tensor1 / tensor2``; aliases ``out``
            when provided.

    Raises:
        RuntimeError: If ``input`` is not floating point or the operands cannot be broadcast.

    Note:
        Registered ``aten::addcdiv_`` dispatch aliases the result to ``input``.
    """
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


@register_aten(
    ["aten::ldexp.Tensor", "aten::ldexp.out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_ldexp(self, other, out=None):
    """Multiply a tensor by a power of two.

    Args:
        self (torch.Tensor): Mantissa tensor.
        other (torch.Tensor): Exponent tensor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Result of ``self * 2**other``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If floating-point ``self`` is paired with a non-floating ``out``.
    """
    return torch.ops.aten.ldexp(self, other)


@register_aten(["aten::rsub.Tensor", "aten::rsub.Tensor_out"], static_argnames=("alpha",))
def torch_rsub(self, other, alpha=1, out=None):
    """Subtract a scaled tensor from another operand.

    Args:
        self (torch.Tensor): Tensor to subtract.
        other (torch.Tensor or Number): Minuend.
        alpha (Number, optional): Scale factor applied to ``self``. Defaults to ``1``.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Difference ``other - alpha * self``; aliases ``out`` when provided.

    Raises:
        RuntimeError: If either operand is Boolean.
    """
    check_bool_sub(self, other)
    return torch.ops.aten.rsub(self, other, alpha=alpha)


_POLAR_RESULT_DTYPE = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def _validate_polar_out(abs, angle, out=None):
    """Reject an out= tensor whose dtype cannot hold the complex result.

    Mirrors PyTorch's structured-kernel unsafe-cast guard, which the Neuron
    dispatch path skips. polar makes a complex result from a real magnitude, so
    the out dtype must be the complex type that matches the input width.
    """
    if out is None:
        return
    expected = _POLAR_RESULT_DTYPE.get(abs.dtype)
    if expected is not None and out.dtype != expected:
        raise RuntimeError(
            f"Expected out tensor to have dtype {expected}, but got {out.dtype} instead"
        )


@register_aten(
    ["aten::polar", "aten::polar.out"],
    output_params=("out",),
    validate_static_inputs=_validate_polar_out,
)
def torch_polar(abs, angle, out=None):
    """Build a complex tensor from a magnitude and a phase angle.

    Args:
        abs (torch.Tensor): Magnitude tensor.
        angle (torch.Tensor): Phase-angle tensor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Complex tensor represented by ``abs * exp(1j * angle)``;
            aliases ``out`` when provided.

    Raises:
        RuntimeError: If ``out`` does not have the complex dtype corresponding to ``abs``.

    Note:
        The shared ``polar_decomp`` rewrites this op into real cosine and sine math
        plus ``view_as_complex`` on eager and ``torch.compile`` paths because
        torch-mlir has no StableHLO legalization for ``aten::polar``.
    """
    return torch.polar(abs, angle)


def _validate_logaddexp2(self, other, out=None):
    """Reject non-float inputs and mismatched out= dtypes.

    The Neuron dispatch path skips aten's float-only and unsafe-cast guards, so
    replicate them here to match CPU.
    """
    for t in (self, other):
        if not t.is_floating_point():
            raise RuntimeError(f'"logaddexp2" not implemented for {t.dtype}')
    if out is not None:
        expected = torch.result_type(self, other)
        if out.dtype != expected:
            raise RuntimeError(
                f"result type {expected} can't be cast to the desired output type {out.dtype}"
            )


@register_aten(
    ["aten::logaddexp2", "aten::logaddexp2.out"],
    validate_static_inputs=_validate_logaddexp2,
)
def torch_logaddexp2(self, other, out=None):
    """Compute the base-two log-sum-exp of two tensors.

    Args:
        self (torch.Tensor): First input tensor.
        other (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Destination populated with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: Stable element-wise ``log2(2**self + 2**other)`` result;
            aliases ``out`` when provided.

    Note:
        Neuron has no native ``float64``, so double inputs run in ``float32`` and
        cast back instead of using CPU fallback.
    """
    return torch.ops.aten.logaddexp2(self, other)
