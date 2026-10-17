import functools
import logging
from collections.abc import Callable
from typing import Any

import nki.language as nl
import torch

from torch_neuronx.neuron_dynamo_backend._nki.ast import (
    NISAOpValue,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten

lowerings: dict[Callable[..., Any] | str, Callable[..., Any]] = {}

# Map from point-wise core aten ops to nki ops.
# TODO: add decompositions for None-mapped aten ops
# TODO: are there any other ops omitted?

binary_op_map = {
    aten.add: nl.add,
    aten.atan2: None,
    aten.bitwise_and: nl.bitwise_and,
    aten.bitwise_or: nl.bitwise_or,
    aten.bitwise_xor: nl.bitwise_xor,
    aten.div: None,
    aten.eq: nl.equal,
    aten.fill: None,
    aten.fmod: None,
    aten.ge: nl.greater_equal,
    aten.gt: nl.greater,
    aten.le: nl.less_equal,
    aten.logical_and: nl.logical_and,
    aten.logical_or: nl.logical_or,
    aten.logical_xor: nl.logical_xor,
    aten.lt: nl.less,
    aten.maximum: nl.maximum,
    aten.minimum: nl.minimum,
    aten.mul: nl.multiply,
    aten.ne: nl.not_equal,
    aten.pow: nl.power,
    aten.remainder: None,
    aten.sub: nl.subtract,
}


def is_scalar(value):
    """Check whether a value is a supported Python scalar.

    Args:
        value (Any): Value to inspect.

    Returns:
        bool: ``True`` for integer, floating-point, Boolean, or complex values.
    """
    return isinstance(value, int | float | bool | complex)


def _scalar_expr(value):
    return repr(value)


def register_all_overload(lowering_dict, aten_fn, lowering_fn):
    """Register a lowering for an ATen operator and its eligible overloads.

    Args:
        lowering_dict (dict): Registry to update; overload eligibility is checked
            against the module-level ``lowerings`` registry.
        aten_fn (Any): ATen operator, overload packet, or list or tuple of operators.
        lowering_fn (Callable): Lowering function to register.
    """
    aten_fn = [aten_fn] if not isinstance(aten_fn, list | tuple) else list(aten_fn)
    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    for fn in aten_fn:
        lowering_dict[fn] = lowering_fn


def make_binary_op(aten_op, nl_op, ctx, *args, **kwargs):
    """Lower a binary ATen operation to an NISA tensor operation.

    Args:
        aten_op (Callable): ATen operation used to infer result metadata.
        nl_op (Callable): NKI language operation to emit.
        ctx (NKIContext): Lowering context to update.
        *args (Any): Left operation value and right operation value or scalar.
        **kwargs (Any): Options including ``reverse`` for scalar operations.

    Returns:
        NISAOpValue: Emitted binary operation.
    """
    nl_op_name = "nl." + nl_op._name
    reverse = kwargs.get("reverse", False)
    if is_scalar(args[1]):
        meta_tensor = aten_op(args[0].dst_arr.meta_tensor, args[1])
        binary_op = NISAOpValue(ctx, "nisa.tensor_scalar", meta_tensor)
        nisa_args = [binary_op.dst_arr, args[0], nl_op_name, _scalar_expr(args[1])]
        if reverse:
            nisa_args.append("reverse0=True")
    else:
        lhs_shape = args[0].dst_arr.meta_tensor.shape
        rhs_shape = args[1].dst_arr.meta_tensor.shape
        meta_tensor = aten_op(args[0].dst_arr.meta_tensor, args[1].dst_arr.meta_tensor)
        # If one operand is (1,1), broadcast it to match the other's shape
        # nisa.tensor_tensor requires matching partition dimensions
        if rhs_shape == (1, 1) and lhs_shape != (1, 1):
            broadcast_op = NISAOpValue(
                ctx,
                "nl.broadcast_to",
                torch.empty(lhs_shape, dtype=meta_tensor.dtype, device="meta"),
            )
            broadcast_op.name = broadcast_op.dst_arr.name
            broadcast_op._custom_codegen = (
                f"{broadcast_op.name} = nl.broadcast_to({args[1].name}, {tuple(lhs_shape)})"
            )
            ctx.code_body.append(broadcast_op)
            binary_op = NISAOpValue(ctx, "nisa.tensor_tensor", meta_tensor)
            nisa_args = [binary_op.dst_arr, args[0], broadcast_op, nl_op_name]
        elif lhs_shape == (1, 1) and rhs_shape != (1, 1):
            broadcast_op = NISAOpValue(
                ctx,
                "nl.broadcast_to",
                torch.empty(rhs_shape, dtype=meta_tensor.dtype, device="meta"),
            )
            broadcast_op.name = broadcast_op.dst_arr.name
            broadcast_op._custom_codegen = (
                f"{broadcast_op.name} = nl.broadcast_to({args[0].name}, {tuple(rhs_shape)})"
            )
            ctx.code_body.append(broadcast_op)
            binary_op = NISAOpValue(ctx, "nisa.tensor_tensor", meta_tensor)
            nisa_args = [binary_op.dst_arr, broadcast_op, args[1], nl_op_name]
        else:
            binary_op = NISAOpValue(ctx, "nisa.tensor_tensor", meta_tensor)
            nisa_args = [binary_op.dst_arr, args[0], args[1], nl_op_name]
    nisa_op = binary_op(*nisa_args)
    ctx.code_body.append(nisa_op)
    return nisa_op


for aten_op in binary_op_map:
    nl_op = binary_op_map[aten_op]
    if nl_op is not None:
        lowering_fn = functools.partial(make_binary_op, aten_op, nl_op)
        register_all_overload(lowerings, aten_op, lowering_fn)


activation_op_map = {
    aten.abs: nl.abs,
    aten.acos: None,
    aten.acosh: None,
    aten.asin: None,
    aten.asinh: None,
    aten.atan: nl.arctan,
    aten.atanh: None,
    aten.ceil: None,
    aten.cos: None,
    aten.cosh: None,
    aten.elu: None,
    aten.erf: nl.erf,
    aten.exp: nl.exp,
    aten.expm1: None,
    aten.floor: None,
    aten.gelu: nl.gelu,
    aten.hardtanh: None,
    aten.leaky_relu: None,
    aten.log: nl.log,
    aten.log10: None,
    aten.log1p: None,
    aten.log2: None,
    aten.neg: None,
    aten.relu: nl.relu,
    aten.round: None,
    aten.rsqrt: nl.rsqrt,
    aten.sigmoid: nl.sigmoid,
    aten.sign: nl.sign,
    aten.sin: nl.sin,
    aten.sinh: None,
    aten.sqrt: nl.sqrt,
    aten.tan: None,
    aten.tanh: nl.tanh,
    aten.trunc: None,
    aten.reciprocal: nl.reciprocal,
}


def make_activation(aten_op, nl_op, ctx, *args, **kwargs):
    """Lower a unary ATen operation to an NISA activation operation.

    Args:
        aten_op (Callable): ATen operation used to infer result metadata.
        nl_op (Callable): NKI language activation to emit.
        ctx (NKIContext): Lowering context to update.
        *args (NISAOpValue): Input operation value.
        **kwargs (Any): Additional lowering options, which are ignored.

    Returns:
        NISAOpValue: Emitted activation operation.
    """
    nl_op_name = "nl." + nl_op._name
    meta_tensor = aten_op(args[0].dst_arr.meta_tensor)
    activation_op = NISAOpValue(ctx, "nisa.activation", meta_tensor)
    nisa_args = [activation_op.dst_arr, nl_op_name, args[0]]
    nisa_op = activation_op(*nisa_args)
    ctx.code_body.append(nisa_op)
    return nisa_op


for aten_op in activation_op_map:
    nl_op = activation_op_map[aten_op]
    if nl_op is not None:
        lowering_fn = functools.partial(make_activation, aten_op, nl_op)
        register_all_overload(lowerings, aten_op, lowering_fn)

misc_op_map = {
    aten.clone: None,
    aten.copy: None,
    aten.isinf: None,
    aten.isnan: None,
    aten.logical_not: None,
    aten.bitwise_not: nl.invert,
}


def make_new_array(ctx, shape, value, dtype):
    """Create an NKI array filled by an NISA memset operation.

    Args:
        ctx (NKIContext): Lowering context to update.
        shape (Sequence[int]): Shape of the array.
        value (Any): Scalar or template value used to fill the array.
        dtype (torch.dtype): Data type of the array.

    Returns:
        NISAOpValue: Emitted memset operation.
    """
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    add_op = NISAOpValue(ctx, "nisa.memset", meta_tensor)

    nisa_args = [add_op.dst_arr, value]
    nisa_op = add_op(*nisa_args)
    ctx.code_body.append(nisa_op)
    return nisa_op


def fill_lowering(ctx, *args, **kwargs):
    """Lower ``aten.fill`` to an NKI constant array.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value and fill value.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.
    """
    meta_tensor = args[0].dst_arr.meta_tensor
    value = args[1]
    return make_new_array(ctx, meta_tensor.shape, value, meta_tensor.dtype)


register_all_overload(lowerings, aten.fill, fill_lowering)


def full_lowering(ctx, *args, **kwargs):
    # full(size, value): a constant tile. Emitted by clamp_decomp as full((), c).
    """Lower ``aten.full`` to an NKI constant array.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Requested size and fill value.
        **kwargs (Any): Options including the optional ``dtype``; other options are
            ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.

    Note:
        Shapes with fewer than two dimensions become a ``(1, 1)`` tile.
    """
    size = list(args[0]) if args else []
    value = args[1]
    shape = tuple(size) if len(size) >= 2 else (1, 1)
    dtype = kwargs.get("dtype") or torch.get_default_dtype()
    return make_new_array(ctx, shape, value, dtype)


register_all_overload(lowerings, aten.full, full_lowering)


def new_ones_lowering(ctx, *args, **kwargs):
    """Lower ``aten.new_ones`` to an NKI array filled with ones.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value supplying the shape; remaining values are
            ignored.
        **kwargs (Any): Options supplying the required ``dtype``; other options are
            ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.

    Raises:
        KeyError: If ``dtype`` is not provided.
    """
    meta_tensor = args[0].dst_arr.meta_tensor.to(kwargs["dtype"])
    if "dtype" in kwargs:
        meta_tensor = meta_tensor.to(kwargs["dtype"])
    return make_new_array(ctx, meta_tensor.shape, 1, meta_tensor.dtype)


register_all_overload(lowerings, aten.new_ones, new_ones_lowering)


def new_zeros_lowering(ctx, *args, **kwargs):
    """Lower ``aten.new_zeros`` to an NKI array filled with zeros.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value supplying the shape; remaining values are
            ignored.
        **kwargs (Any): Options supplying the required ``dtype``; other options are
            ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.

    Raises:
        KeyError: If ``dtype`` is not provided.
    """
    meta_tensor = args[0].dst_arr.meta_tensor.to(kwargs["dtype"])
    if "dtype" in kwargs:
        meta_tensor = meta_tensor.to(kwargs["dtype"])
    return make_new_array(ctx, meta_tensor.shape, 0, meta_tensor.dtype)


register_all_overload(lowerings, aten.new_zeros, new_zeros_lowering)


def ones_like_lowering(ctx, *args, **kwargs):
    """Lower ``aten.ones_like`` to an NKI array filled with ones.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value supplying shape and default dtype;
            remaining values are ignored.
        **kwargs (Any): Options including an optional ``dtype`` override; other
            options are ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.
    """
    meta_tensor = args[0].dst_arr.meta_tensor
    dtype = kwargs.get("dtype")
    if dtype is not None:
        meta_tensor = meta_tensor.to(dtype)
    return make_new_array(ctx, meta_tensor.shape, 1, meta_tensor.dtype)


register_all_overload(lowerings, aten.ones_like, ones_like_lowering)


def zeros_like_lowering(ctx, *args, **kwargs):
    """Lower ``aten.zeros_like`` to an NKI array filled with zeros.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value supplying shape and default dtype;
            remaining values are ignored.
        **kwargs (Any): Options including an optional ``dtype`` override; other
            options are ignored.

    Returns:
        NISAOpValue: Emitted constant-array operation.
    """
    meta_tensor = args[0].dst_arr.meta_tensor
    dtype = kwargs.get("dtype")
    if dtype is not None:
        meta_tensor = meta_tensor.to(dtype)
    return make_new_array(ctx, meta_tensor.shape, 0, meta_tensor.dtype)


register_all_overload(lowerings, aten.zeros_like, zeros_like_lowering)

# Composite operations lowering


def div_lowering(ctx, *args, **kwargs):
    """Lower division to multiplication by a reciprocal.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Dividend and divisor operation values.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted multiplication operation.

    Raises:
        ZeroDivisionError: If a scalar divisor is zero.
    """
    if is_scalar(args[1]):
        result = make_binary_op(aten.mul, nl.multiply, ctx, args[0], 1 / args[1])
    else:
        reciprocal = make_activation(aten.reciprocal, nl.reciprocal, ctx, args[1])
        result = make_binary_op(aten.mul, nl.multiply, ctx, args[0], reciprocal)
    return result


register_all_overload(lowerings, aten.div, div_lowering)


def tanh_backward_lowering(ctx, *args, **kwargs):
    # tanh_backward(grad_output, output) = grad_output * (1 - output^2)
    """Lower the hyperbolic-tangent backward formula to NISA operations.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Gradient output and forward output operation values.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted gradient operation.
    """
    grad_output, output = args[0], args[1]
    # output^2
    output_sq = make_binary_op(aten.mul, nl.multiply, ctx, output, output)
    # ones tensor

    one_minus_sq = make_binary_op(aten.sub, nl.subtract, ctx, output_sq, 1, reverse=True)
    # grad_output * (1 - output^2)
    return make_binary_op(aten.mul, nl.multiply, ctx, grad_output, one_minus_sq)


register_all_overload(lowerings, aten.tanh_backward, tanh_backward_lowering)


def neg_lowering(ctx, *args, **kwargs):
    # neg(x) = -x = 0 - x
    # Emitted as `nisa.tensor_scalar(dst, x, nl.subtract, 0, reverse0=True)`,
    """Lower negation to a reversed scalar subtraction from zero.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Input operation value.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted subtraction operation.
    """
    return make_binary_op(aten.sub, nl.subtract, ctx, args[0], 0, reverse=True)


register_all_overload(lowerings, aten.neg, neg_lowering)


def exp2_lowering(ctx, *args, **kwargs):
    # exp2(x) = 2^x = exp(x * ln(2))
    # NKI doesn't expose nl.exp2; decompose into a scaled nl.exp.
    """Lower base-two exponentiation to a scaled natural exponential.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Input operation value.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted exponential operation.
    """
    import math

    scaled = make_binary_op(aten.mul, nl.multiply, ctx, args[0], math.log(2.0))
    return make_activation(aten.exp, nl.exp, ctx, scaled)


register_all_overload(lowerings, aten.exp2, exp2_lowering)


def rsub_lowering(ctx, *args, **kwargs):
    # rsub(x, scalar) = scalar - x.
    # Maps to `tensor_scalar(dst, x, nl.subtract, scalar, reverse0=True)`,
    """Lower reverse subtraction to a reversed scalar subtraction.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Input operation value and scalar minuend.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted subtraction operation.
    """
    return make_binary_op(aten.sub, nl.subtract, ctx, args[0], args[1], reverse=True)


register_all_overload(lowerings, aten.rsub, rsub_lowering)


def floor_lowering(ctx, *args, **kwargs):
    # nl.floor is an elementwise builtin, NOT a nisa.activation op, so call it
    # directly.
    """Lower floor directly to the NKI language elementwise operation.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Input operation value.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted floor operation.
    """
    src = args[0]
    meta = aten.floor(src.dst_arr.meta_tensor)
    op = NISAOpValue(ctx, "nl.floor", meta)
    op.name = op.dst_arr.name
    op._custom_codegen = f"{op.name} = nl.floor({src.name})"
    ctx.code_body.append(op)
    return op


register_all_overload(lowerings, aten.floor, floor_lowering)


def floor_divide_lowering(ctx, *args, **kwargs):
    # x // N. For a power-of-2 N use an exact right_shift;
    #  otherwise compute - floor(x * (1/N))
    # The result is cast back to x's dtype to match torch's
    # integer floor_divide semantics.
    """Lower floor division by a positive integer scalar.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Dividend operation value and positive integer divisor.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted floor-division operation.

    Raises:
        NotImplementedError: If the divisor is not a positive integer scalar.

    Note:
        Power-of-two divisors use an exact right shift; other divisors use
        ``floor(x * (1 / divisor))`` and cast back to the input dtype.
    """
    src, divisor = args[0], args[1]
    if not (isinstance(divisor, int) and divisor > 0):
        raise NotImplementedError(
            f"floor_divide on NKI only supports positive int scalar divisors, got {divisor!r}"
        )
    if divisor & (divisor - 1) == 0:
        shift = divisor.bit_length() - 1
        meta = torch.empty(
            src.dst_arr.meta_tensor.shape, dtype=src.dst_arr.meta_tensor.dtype, device="meta"
        )
        op = NISAOpValue(ctx, "nisa.tensor_scalar", meta)
        op.name = op.dst_arr.name
        op._custom_codegen = (
            f"nisa.tensor_scalar({op.dst_arr.name}, "
            f"nl.add({src.name}, 0, dtype=nl.int32), nl.right_shift, {shift})"
        )
        ctx.code_body.append(op)
        return op
    # non-power-of-2: floor(x * (1/N)), then cast back to x's dtype
    scaled = make_binary_op(aten.mul, nl.multiply, ctx, src, 1.0 / divisor)
    floored = floor_lowering(ctx, scaled)
    return to_copy_lowering(ctx, floored, dtype=src.dst_arr.meta_tensor.dtype)


register_all_overload(lowerings, aten.floor_divide, floor_divide_lowering)


def remainder_lowering(ctx, *args, **kwargs):
    # remainder(x, y) = x - floor_divide(x, y) * y.
    """Lower remainder through floor division and multiplication.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Dividend operation value and positive integer divisor.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted remainder operation.

    Raises:
        NotImplementedError: If the divisor is not a positive integer scalar.
    """
    fd = floor_divide_lowering(ctx, *args, **kwargs)
    product = make_binary_op(aten.mul, nl.multiply, ctx, fd, args[1])
    return make_binary_op(aten.sub, nl.subtract, ctx, args[0], product)


register_all_overload(lowerings, aten.remainder, remainder_lowering)


def clamp_lowering(ctx, *args, **kwargs):
    # clamp(x, lo, hi) = minimum(maximum(x, lo), hi).
    # Either lo or hi may be None (one-sided clamp);
    """Lower clamp to optional NKI minimum and maximum operations.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Input operation value and optional positional bounds.
        **kwargs (Any): Optional ``min`` and ``max`` bounds; other options are
            ignored.

    Returns:
        NISAOpValue: Constrained value, or the original input when both bounds are ``None``.
    """
    x = args[0]
    lo = args[1] if len(args) > 1 else kwargs.get("min")
    hi = args[2] if len(args) > 2 else kwargs.get("max")
    if lo is not None:
        x = make_binary_op(aten.clamp_min, nl.maximum, ctx, x, lo)
    if hi is not None:
        x = make_binary_op(aten.clamp_max, nl.minimum, ctx, x, hi)
    return x


register_all_overload(lowerings, aten.clamp, clamp_lowering)


def identity_lowering(ctx, *args, **kwargs):
    """Return the input operation value unchanged.

    Args:
        ctx (NKIContext): Unused lowering context.
        *args (Any): Input operation value.
        **kwargs (Any): Unused ATen arguments.

    Returns:
        NISAOpValue: First input operation value.
    """
    return args[0]


register_all_overload(lowerings, aten.detach, identity_lowering)
register_all_overload(lowerings, aten.alias, identity_lowering)


def to_copy_lowering(ctx, *args, **kwargs):
    """Lower a dtype conversion to an NISA tensor copy when needed.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Source operation value.
        **kwargs (Any): Options including the target ``dtype``; other options are
            ignored.

    Returns:
        NISAOpValue: Source value or emitted copy operation.
    """
    src = args[0]
    target_dtype = kwargs.get("dtype", src.dst_arr.meta_tensor.dtype)
    # If dtypes match, no copy needed
    if target_dtype == src.dst_arr.meta_tensor.dtype:
        return src
    meta_tensor = src.dst_arr.meta_tensor.to(target_dtype)
    copy_op = NISAOpValue(ctx, "nisa.tensor_copy", meta_tensor)
    nisa_args = [copy_op.dst_arr, src]
    nisa_op = copy_op(*nisa_args)
    ctx.code_body.append(nisa_op)
    return nisa_op


register_all_overload(lowerings, aten._to_copy, to_copy_lowering)


def index_lowering(ctx, *args, **kwargs):
    """Lower ``aten.index.Tensor`` to a per-partition NKI gather.

    Args:
        ctx (NKIContext): Lowering context to update.
        *args (Any): Preloaded source operation and one- or two-item index list.
        **kwargs (Any): Additional ATen arguments, which are ignored.

    Returns:
        NISAOpValue: Emitted flattened-gather operation.

    Raises:
        NotImplementedError: If indexing is not one- or two-dimensional or contains ``None``.

    Note:
        The source is a preloaded ``(1, N)`` SBUF row broadcast to ``(P, N)``.
        Two-dimensional indexing uses only the final index. Indices are converted to
        ``uint32``; index shapes below rank two produce a ``(128, 128)`` result.
    """
    source = args[0]
    indices = args[1]
    # 2-D source is pre-sliced to one row by the template; gather on the column
    # index. Both ranks resolve to a single index tile against a (1, N) source.
    if not 1 <= len(indices) <= 2 or any(i is None for i in indices):
        raise NotImplementedError(
            f"index lowering only supports 1D and 2D indexing, got {len(indices)} indices"
        )
    idx = indices[-1]
    idx_shape = idx.dst_arr.meta_tensor.shape
    result_shape = idx_shape if len(idx_shape) >= 2 else (128, 128)

    num_elems = 1
    for d in source.dst_arr.meta_tensor.shape:
        num_elems *= int(d)
    partition_dim = result_shape[0]

    bcast_meta = torch.empty((partition_dim, num_elems), dtype=source.dst_arr.dtype, device="meta")
    bcast_op = NISAOpValue(ctx, "nl.broadcast_to", bcast_meta)
    bcast_op.name = bcast_op.dst_arr.name
    bcast_op._custom_codegen = (
        f"{bcast_op.name} = nl.broadcast_to({source.name}, "
        f"({partition_dim}, {source.name}.shape[1]))"
    )
    ctx.code_body.append(bcast_op)

    # nc_n_gather requires uint32 indices.
    gather_meta = torch.empty(result_shape, dtype=source.dst_arr.dtype, device="meta")
    gather_op = NISAOpValue(ctx, "nl.gather_flattened", gather_meta)
    gather_op.name = gather_op.dst_arr.name
    gather_op._custom_codegen = (
        f"{gather_op.name} = nl.gather_flattened({bcast_op.name}, "
        f"nl.add({idx.name}, 0, dtype=nl.uint32))"
    )
    ctx.code_body.append(gather_op)
    return gather_op


register_all_overload(lowerings, aten.index, index_lowering)
