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


def _not_implemented(ctx, *args, **kwargs):
    raise NotImplementedError


def is_scalar(value):
    return isinstance(value, int | float | bool | complex)


def register_all_overload(lowering_dict, aten_fn, lowering_fn):
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
    nl_op_name = "nl." + nl_op._name
    reverse = kwargs.get("reverse", False)
    if is_scalar(args[1]):
        meta_tensor = aten_op(args[0].dst_arr.meta_tensor, args[1])
        binary_op = NISAOpValue(ctx, "nisa.tensor_scalar", meta_tensor)
        nisa_args = [binary_op.dst_arr, args[0], nl_op_name, args[1]]
        if reverse:
            nisa_args.append("reverse0=True")
    else:
        meta_tensor = aten_op(args[0].dst_arr.meta_tensor, args[1].dst_arr.meta_tensor)
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
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    add_op = NISAOpValue(ctx, "nisa.memset", meta_tensor)

    nisa_args = [add_op.dst_arr, value]
    nisa_op = add_op(*nisa_args)
    ctx.code_body.append(nisa_op)
    return nisa_op


def fill_lowering(ctx, *args, **kwargs):
    meta_tensor = args[0].dst_arr.meta_tensor
    value = args[1]
    return make_new_array(ctx, meta_tensor.shape, value, meta_tensor.dtype)


register_all_overload(lowerings, aten.fill, fill_lowering)


def new_ones_lowering(ctx, *args, **kwargs):
    meta_tensor = args[0].dst_arr.meta_tensor.to(kwargs["dtype"])
    if "dtype" in kwargs:
        meta_tensor = meta_tensor.to(kwargs["dtype"])
    return make_new_array(ctx, meta_tensor.shape, 1, meta_tensor.dtype)


register_all_overload(lowerings, aten.new_ones, new_ones_lowering)


def new_zeros_lowering(ctx, *args, **kwargs):
    meta_tensor = args[0].dst_arr.meta_tensor.to(kwargs["dtype"])
    if "dtype" in kwargs:
        meta_tensor = meta_tensor.to(kwargs["dtype"])
    return make_new_array(ctx, meta_tensor.shape, 0, meta_tensor.dtype)


register_all_overload(lowerings, aten.new_zeros, new_zeros_lowering)

# Composite operations lowering


def div_lowering(ctx, *args, **kwargs):
    if is_scalar(args[1]):
        result = make_binary_op(aten.mul, nl.multiply, ctx, args[0], 1 / args[1])
    else:
        reciprocal = make_activation(aten.reciprocal, nl.reciprocal, ctx, args[1])
        result = make_binary_op(aten.mul, nl.multiply, ctx, args[0], reciprocal)
    return result


register_all_overload(lowerings, aten.div, div_lowering)


def tanh_backward_lowering(ctx, *args, **kwargs):
    # tanh_backward(grad_output, output) = grad_output * (1 - output^2)
    grad_output, output = args[0], args[1]
    # output^2
    output_sq = make_binary_op(aten.mul, nl.multiply, ctx, output, output)
    # ones tensor

    one_minus_sq = make_binary_op(aten.sub, nl.subtract, ctx, output_sq, 1, reverse=True)
    # grad_output * (1 - output^2)
    return make_binary_op(aten.mul, nl.multiply, ctx, grad_output, one_minus_sq)


register_all_overload(lowerings, aten.tanh_backward, tanh_backward_lowering)


def identity_lowering(ctx, *args, **kwargs):
    return args[0]


register_all_overload(lowerings, aten.detach, identity_lowering)


def to_copy_lowering(ctx, *args, **kwargs):
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
