"""Comparison operations."""

import torch

from ..operation_registry import register_aten
from ..validation import check_broadcast_shapes, make_non_boolean_out_validator
from .constants import INT_DTYPES
from .unary import _validate_float_promoting_out


def _validate_broadcast_and_out(x, other=None, *args, out=None, **kwargs):
    """Reject a non-broadcastable pair, then check the out= dtype.

    minimum/maximum and the clamp_min/clamp_max tensor overloads report a
    broadcast failure from the lowering, with different wording than ATen. The
    foreach list overloads dispatch here for each element, so this also gives
    _foreach_clamp_max and its family the ATen message.
    """
    if isinstance(x, torch.Tensor) and isinstance(other, torch.Tensor):
        check_broadcast_shapes(x, other)
    _validate_float_promoting_out(x, other, *args, out=out, **kwargs)


@register_aten(["aten::eq", "aten::eq.Scalar_out", "aten::eq.Tensor_out"])
def torch_eq(x, y, out=None):
    """Compare two values for elementwise equality.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean equality mask; aliases ``out`` when provided.
    """
    return torch.eq(x, y)


@register_aten(["aten::ne", "aten::ne.Scalar_out", "aten::ne.Tensor_out"])
def torch_ne(x, y, out=None):
    """Compare two values for elementwise inequality.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean inequality mask; aliases ``out`` when provided.
    """
    return torch.ne(x, y)


@register_aten(["aten::gt", "aten::gt.Scalar_out", "aten::gt.Tensor_out"])
def torch_gt(x, y, out=None):
    """Compare whether elements of one value exceed another.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean greater-than mask; aliases ``out`` when provided.
    """
    return torch.gt(x, y)


@register_aten(["aten::ge", "aten::ge.Scalar_out", "aten::ge.Tensor_out"])
def torch_ge(x, y, out=None):
    """Compare whether elements of one value meet or exceed another.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean greater-than-or-equal mask; aliases ``out`` when provided.
    """
    return torch.ge(x, y)


@register_aten(["aten::lt", "aten::lt.Scalar_out", "aten::lt.Tensor_out"])
def torch_lt(x, y, out=None):
    """Compare whether elements of one value are less than another.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean less-than mask; aliases ``out`` when provided.
    """
    return torch.lt(x, y)


@register_aten(["aten::le", "aten::le.Tensor_out", "aten::le.Scalar_out"])
def torch_le(x, y, out=None):
    """Compare whether elements of one value are at most another.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor or Number): Tensor or scalar to compare with ``x``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean less-than-or-equal mask; aliases ``out`` when provided.
    """
    return torch.le(x, y)


@register_aten(
    ["aten::clamp", "aten::clamp.out", "aten::clamp.Tensor", "aten::clamp.Tensor_out"],
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_clamp(x, min=None, max=None, out=None):
    """Clamp tensor elements to optional lower and upper bounds.

    Args:
        x (torch.Tensor): Input tensor.
        min (torch.Tensor or Number, optional): Lower bound. Defaults to ``None``.
        max (torch.Tensor or Number, optional): Upper bound. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Clamped tensor, promoted with tensor bounds; aliases ``out`` when provided.

    Raises:
        RuntimeError: If both ``min`` and ``max`` are ``None``.
    """
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")

    # Handle tensor bounds with torch.maximum/minimum to avoid StableHLO
    # broadcast_dimensions errors on rank mismatch.
    # Promote to common dtype to match PyTorch's dtype promotion rules.
    if isinstance(min, torch.Tensor):
        common_dtype = torch.result_type(x, min)
        x, min = torch.maximum(x.to(common_dtype), min.to(common_dtype)), None
    if isinstance(max, torch.Tensor):
        common_dtype = torch.result_type(x, max)
        x, max = torch.minimum(x.to(common_dtype), max.to(common_dtype)), None

    return torch.clamp(x, min=min, max=max) if min is not None or max is not None else x


@register_aten(
    [
        "aten::clamp_max",
        "aten::clamp_max.out",
        "aten::clamp_max.Tensor",
        "aten::clamp_max.Tensor_out",
    ],
    validate_static_inputs=_validate_broadcast_and_out,
)
def torch_clamp_max(x, max, out=None):
    # Ensure tensor max has same dtype as input
    """Clamp tensor elements to an upper bound.

    Args:
        x (torch.Tensor): Input tensor.
        max (torch.Tensor or Number): Upper bound.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Upper-bounded tensor, promoted with a tensor bound; aliases ``out`` when
            provided.

    Raises:
        RuntimeError: If a tensor bound cannot broadcast with ``x`` or floating-point ``x``
            is paired with a non-floating ``out``.

    Note:
        With integer ``x``, floating tensor ``max``, and integer ``out``, registration
        currently copy-casts the floating result into ``out``.
    """
    if isinstance(max, torch.Tensor):
        # Use torch.minimum instead of torch.clamp_max: it handles broadcasting
        # natively, avoiding StableHLO broadcast_dimensions errors on rank mismatch
        common_dtype = torch.result_type(x, max)
        return torch.minimum(x.to(common_dtype), max.to(common_dtype))
    return torch.clamp_max(x, max)


@register_aten(
    [
        "aten::clamp_min",
        "aten::clamp_min.out",
        "aten::clamp_min.Tensor",
        "aten::clamp_min.Tensor_out",
    ],
    validate_static_inputs=_validate_broadcast_and_out,
)
def torch_clamp_min(x, min, out=None):
    # Ensure tensor min has same dtype as input
    """Clamp tensor elements to a lower bound.

    Args:
        x (torch.Tensor): Input tensor.
        min (torch.Tensor or Number): Lower bound.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Lower-bounded tensor, promoted with a tensor bound; aliases ``out`` when
            provided.
    """
    if isinstance(min, torch.Tensor):
        # Use torch.maximum instead of torch.clamp_min: it handles broadcasting
        # natively, avoiding StableHLO broadcast_dimensions errors on rank mismatch
        common_dtype = torch.result_type(x, min)
        return torch.maximum(x.to(common_dtype), min.to(common_dtype))
    return torch.clamp_min(x, min)


@register_aten(["aten::isnan", "aten::isnan.out"], output_params=("out",))
def torch_isnan(x, out=None):
    """Identify NaN elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean mask true for NaN elements; aliases ``out`` when provided.
    """
    return torch.isnan(x)


@register_aten(["aten::isinf", "aten::isinf.out"], output_params=("out",))
def torch_isinf(x, out=None):
    """Identify infinite elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean mask marking infinite elements; aliases ``out`` when provided and
            is always false for integer inputs.

    Note:
        Integer tensors bypass ``stablehlo.abs`` because it does not support
        their dtypes; integers cannot represent infinity.
    """
    if x.dtype.is_floating_point or x.dtype.is_complex:
        return torch.isinf(x)
    else:
        return torch.zeros_like(x, dtype=torch.bool)


@register_aten(
    ["aten::isneginf", "aten::isneginf.out"],
    validate_static_inputs=make_non_boolean_out_validator("isneginf"),
)
def torch_isneginf(x, out=None):
    """Identify negative-infinity elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean mask true for negative infinity; aliases ``out`` when provided.

    Raises:
        RuntimeError: If ``out`` does not have Boolean dtype.
    """
    return torch.isneginf(x)


@register_aten(["aten::isfinite"])
def torch_isfinite(x):
    """Identify finite elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Boolean mask true for finite elements.

    Note:
        ``torch.isfinite`` is decomposed before MLIR, so no
        ``stablehlo.isfinite`` reaches HLO. Integers and booleans return true;
        floating-point and complex values are checked for NaN and infinity.
    """
    if x.dtype in (*INT_DTYPES, torch.bool):
        # Integers and bools are always finite
        return torch.ones_like(x, dtype=torch.bool)
    else:
        # For floating point: finite = not (nan or inf)
        return ~(torch.isnan(x) | torch.isinf(x))


@register_aten(
    [
        "aten::isin.Tensor_Tensor_out",
        "aten::isin.Tensor_Scalar_out",
        "aten::isin.Scalar_Tensor_out",
    ],
    static_argnames=("assume_unique", "invert"),
)
def torch_isin_out(elements, test_elements, assume_unique=False, invert=False, out=None):
    """Compute membership for an ``out=`` variant of ``torch.isin``.

    Args:
        elements (torch.Tensor or Number): Values to test for membership.
        test_elements (torch.Tensor or Number): Values against which to test.
        assume_unique (bool, optional): Whether both inputs contain unique elements.
            Defaults to ``False``.
        invert (bool, optional): Whether to invert the membership result. Defaults to ``False``.
        out (torch.Tensor, optional): Destination populated by registration. Defaults to ``None``.

    Returns:
        torch.Tensor: Boolean membership mask; aliases ``out`` when provided.
    """
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


@register_aten(
    ["aten::minimum", "aten::minimum.out"],
    validate_static_inputs=_validate_broadcast_and_out,
)
def torch_minimum(self, other, out=None):
    """Select the elementwise minimum of two tensors.

    Args:
        self (torch.Tensor): First input tensor.
        other (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Elementwise minimum values; aliases ``out`` when provided.

    Raises:
        RuntimeError: If the inputs cannot broadcast or floating-point ``self`` is paired
            with a non-floating ``out``.

    Note:
        With integer ``self``, floating ``other``, and integer ``out``, registration
        currently copy-casts the floating result into ``out``.
    """
    return torch.minimum(self, other)


@register_aten(
    ["aten::maximum", "aten::maximum.out"],
    validate_static_inputs=_validate_broadcast_and_out,
)
def torch_maximum(self, other, out=None):
    """Select the elementwise maximum of two tensors.

    Args:
        self (torch.Tensor): First input tensor.
        other (torch.Tensor): Second input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Elementwise maximum values; aliases ``out`` when provided.

    Raises:
        RuntimeError: If the inputs cannot broadcast or floating-point ``self`` is paired
            with a non-floating ``out``.

    Note:
        With integer ``self``, floating ``other``, and integer ``out``, registration
        currently copy-casts the floating result into ``out``.
    """
    return torch.maximum(self, other)


@register_aten(
    ["aten::isposinf", "aten::isposinf.out"],
    validate_static_inputs=make_non_boolean_out_validator("isposinf"),
)
def torch_isposinf(self, out=None):
    """Identify positive-infinity elements in a tensor.

    Args:
        self (torch.Tensor): Input tensor.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: Boolean mask true for positive infinity; aliases ``out`` when provided.

    Raises:
        RuntimeError: If ``out`` does not have Boolean dtype.
    """
    return torch.ops.aten.isposinf(self)
