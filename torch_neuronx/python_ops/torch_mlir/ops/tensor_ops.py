"""Tensor manipulation operations (shape, stacking, etc.)."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::cat", "aten::cat.out"], static_argnums=(1,))
def torch_cat(tensors, dim=0):
    # Filter out empty tensors for concatenation
    non_empty_tensors = [t for t in tensors if t.numel() > 0]

    # If no tensors left, return empty tensor with appropriate shape
    if not non_empty_tensors:
        # Create empty tensor with same dtype as first original tensor
        if tensors:
            return torch.empty(0, dtype=tensors[0].dtype, device=tensors[0].device)
        else:
            return torch.empty(0)

    # If only one tensor left, return it directly
    if len(non_empty_tensors) == 1:
        return non_empty_tensors[0]

    return torch.cat(non_empty_tensors, dim)


@register_aten(["aten::stack", "aten::stack.out"], static_argnums=(1,))
def torch_stack(tensors, dim=0, out=None):
    return torch.stack(tensors, dim)


# @register_aten(["aten::vstack", "aten::vstack.out"])
def torch_vstack(tensors):
    return torch.vstack(tensors)


@register_aten(["aten::atleast_2d"])
def torch_atleast_2d(inputs):
    return torch.atleast_2d(inputs)


@register_aten(["aten::triu", "aten::triu.out"], static_argnums=(1,))
def torch_triu(m, diagonal=0):
    return torch.triu(m, diagonal)


@register_aten(["aten::tril", "aten::tril.out"], static_argnums=(1,))
def torch_tril(x, diagonal=0):
    return torch.tril(x, diagonal)


@register_aten(["aten::flip", "aten::flip.out"], static_argnums=(1,))
def torch_flip(x, dims):
    return torch.flip(x, dims)


@register_aten(["aten::pad"], static_argnums=(1, 2, 3))
def torch_pad(input, pad, mode="constant", value=0):
    return torch.nn.functional.pad(input, pad, mode, value)
