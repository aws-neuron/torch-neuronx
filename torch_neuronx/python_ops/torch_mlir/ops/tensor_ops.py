"""Tensor manipulation operations (shape, stacking, etc.)."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::cat", "aten::cat.out"], static_argnums=(1,))
def torch_cat(tensors, dim=0, out=None):
    return torch.cat(tensors, dim)


@register_aten(["aten::stack", "aten::stack.out"], static_argnums=(1,))
def torch_stack(tensors, dim=0, out=None):
    return torch.stack(tensors, dim)


# @register_aten(["aten::vstack", "aten::vstack.out"])
def torch_vstack(tensors):
    return torch.vstack(tensors)


@register_aten(["aten::atleast_2d"])
def torch_atleast_2d(inputs):
    return torch.atleast_2d(inputs)


@register_aten(["aten::triu", "aten::triu.out"])
def torch_triu(m, diagonal=0):
    return torch.triu(m, diagonal)


@register_aten(["aten::tril", "aten::tril.out"])
def torch_tril(x, diagonal=0):
    return torch.tril(x, diagonal)


@register_aten(["aten::flip", "aten::flip.out"])
def torch_flip(x, dims):
    return torch.flip(x, dims)


@register_aten(["aten::pad"], static_argnums=(1, 2, 3))
def torch_pad(input, pad, mode="constant", value=0):
    return torch.nn.functional.pad(input, pad, mode, value)
