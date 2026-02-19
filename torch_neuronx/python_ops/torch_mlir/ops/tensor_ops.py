"""Tensor manipulation operations (shape, stacking, etc.)."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::cat", "aten::cat.out"], static_argnums=(1,))
def torch_cat(tensors, dim=0, out=None):
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


@register_aten(["aten::triu", "aten::triu.out"], static_argnums=(1,))
def torch_triu(m, diagonal=0, out=None):
    return torch.triu(m, diagonal)


@register_aten(["aten::tril", "aten::tril.out"], static_argnums=(1,))
def torch_tril(x, diagonal=0, out=None):
    return torch.tril(x, diagonal)


@register_aten(["aten::flip", "aten::flip.out"], static_argnums=(1,))
def torch_flip(x, dims, out=None):
    return torch.flip(x, dims)


@register_aten(["aten::constant_pad_nd"], static_argnums=(1, 2))
def torch_constant_pad_nd(input, pad, value=0):
    return torch.nn.functional.pad(input, pad, mode="constant", value=value)


@register_aten(["aten::reflection_pad1d"], static_argnums=(1,))
def torch_reflection_pad1d(input, padding):
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(["aten::reflection_pad2d"], static_argnums=(1,))
def torch_reflection_pad2d(input, padding):
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(["aten::reflection_pad3d"], static_argnums=(1,))
def torch_reflection_pad3d(input, padding):
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(["aten::replication_pad1d"], static_argnums=(1,))
def torch_replication_pad1d(input, padding):
    return torch.nn.functional.pad(input, padding, mode="replicate")


@register_aten(["aten::replication_pad2d", "aten::replication_pad2d.out"], static_argnums=(1,))
def torch_replication_pad2d(input, padding, out=None):
    return torch.nn.functional.pad(input, padding, mode="replicate")


@register_aten(["aten::replication_pad3d"], static_argnums=(1,))
def torch_replication_pad3d(input, padding):
    return torch.nn.functional.pad(input, padding, mode="replicate")


# Note: aten::_pad_circular is not registered here because it's marked as
# CompositeImplicitAutograd in PyTorch. Registering it would intercept the operation
# before PyTorch can decompose it into CopySlices operations, breaking the autograd
# chain. By not registering it, PyTorch naturally decomposes circular padding and
# preserves grad_fn=<CopySlices>.


@register_aten(["aten::repeat"], static_argnums=(1,))
def torch_repeat(x, repeats):
    return x.repeat(repeats)


def get_repeat_interleave_output_size(repeats, input=None, dim=None, output_size=None):
    """Calculate and validate output size for repeat_interleave result."""
    if isinstance(repeats, int):
        # Calculate for .self_int variant
        if dim is None:
            calculated_output_size = repeats * input.numel()
        else:
            calculated_output_size = repeats * input.shape[dim]
    else:
        # Calculate for .Tensor and .self_Tensor variants
        calculated_output_size = torch.sum(repeats).item()

    if output_size is None:
        output_size = calculated_output_size

    if output_size != calculated_output_size:
        raise RuntimeError("allocated size does not match required size")

    return output_size


def compute_repeat_interleave(input, repeats, dim=None, output_size=None):
    """Decompose repeat_interleave computation using cumsum and index_select."""
    if dim is None:
        dim = 0
        input = input.flatten()

    # Get scatter indices: cumsum(repeats) shifted right with 0 at start
    scatter_indices = torch.cumsum(repeats, dim=0)
    scatter_indices = torch.roll(scatter_indices, shifts=1, dims=0)
    scatter_indices[0] = 0

    # Mask out-of-range indices (happens when repeats has zeros)
    # Use mask to zero out contributions from invalid indices
    scatter_values = (scatter_indices < output_size).float()
    scatter_indices = scatter_indices.clamp(max=output_size - 1)

    # Create block split indicators
    block_split_indicators = torch.zeros(output_size, dtype=torch.float32, device=input.device)
    block_split_indicators.scatter_add_(0, scatter_indices, scatter_values)

    # Cumsum to get gather indices
    gather_indices = torch.cumsum(block_split_indicators, dim=0).long() - 1

    # Use index_select to gather the repeated elements
    return torch.index_select(input, dim, gather_indices)


# NOTE: repeat_interleave.self_int and repeat_interleave.self_Tensor are NOT registered
# because they are CompositeImplicitAutograd ops that naturally decompose with autograd preserved.
# - self_int decomposes to: unsqueeze -> expand -> clone -> flatten (grad_fn=<ViewBackward0>)
# - self_Tensor decomposes to: index_select (grad_fn=<IndexSelectBackward0>)
# Registering them would break the autograd chain.
# Reference: https://github.com/pytorch/pytorch/blob/8e65dfa6f2c451dc3c71d76d68796d1271c72772/aten/src/ATen/native/Repeat.cpp#L54


@register_aten(
    [
        "aten::repeat_interleave.Tensor",
    ],
    static_argnames=("output_size",),
    uses_preprocessing=True,
)
def torch_repeat_interleave_tensor(repeats, output_size=None):
    """releat_interleave.Tensor variant is not supported in MLIR.

    Custom decomposition and preprocessing needed.
    """
    output_size = get_repeat_interleave_output_size(repeats, output_size=output_size)

    def repeat_interleave_fn(repeats, output_size=None):
        indices = torch.arange(len(repeats), device=repeats.device, dtype=torch.long)
        return compute_repeat_interleave(indices, repeats, output_size=output_size)

    return repeat_interleave_fn, (repeats,), {"output_size": output_size}
