"""Tensor manipulation operations (shape, stacking, etc.)."""

import torch

from ..operation_registry import Preprocessing, register_aten
from .unary import _validate_float_promoting_out


def _wrap_dim(dim, ndim):
    """Validate/normalize `dim` like ATen's maybe_wrap_dim so invalid-dimension cases
    raise the same IndexError messages native torch.roll does (see test_errors_roll)
    instead of "IndexError: tuple index out of range" from x.shape[dim].

    Note: these are TORCH_CHECK_INDEX failures in ATen, so they raise IndexError —
    unlike the shifts/dims alignment checks in torch_roll, which are RuntimeError."""
    if ndim == 0:
        # A 0-dim (scalar) tensor has no dimensions to index.
        raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _roll_one_dim(x, shift, dim):
    """Roll a single dim by `shift` using narrow+cat (on-device slice+concatenate)."""
    dim = _wrap_dim(dim, x.dim())
    n = x.shape[dim]
    if n == 0:
        return x
    shift = int(shift) % n
    if shift == 0:
        return x
    tail = x.narrow(dim, n - shift, shift)  # last `shift` elements wrap to front
    head = x.narrow(dim, 0, n - shift)
    return torch.cat([tail, head], dim=dim)


@register_aten(["aten::roll", "aten::roll.out"], static_argnums=(1, 2), output_params=("out",))
def torch_roll(self, shifts, dims=(), out=None):
    """On-device aten::roll (avoids the CPU fallback). Decomposes to narrow+cat, which
    lowers to slice + concatenate. shifts/dims are static (static_argnums) so the result
    is shape-stable and compiles once per shape/shift config (swin window shifts are
    constant per layer). Also covers roll's backward, which autograd expresses as
    roll(grad, -shifts, dims)."""
    # Validate inputs exactly like native torch.roll so error cases raise the correct
    # RuntimeError (matching the aten::roll TORCH_CHECK messages) rather than an
    # IndexError from shifts[0] during meta/dtype inference — see test_errors_roll.
    if len(shifts) == 0:
        raise RuntimeError("`shifts` required")
    if not dims:
        # dims empty -> native requires exactly one shift; flatten, roll dim 0, reshape back.
        if len(shifts) != 1:
            raise RuntimeError(f"shifts and dimensions must align. shifts: {len(shifts)}, dims:0")
        orig_shape = self.shape
        flat = self.reshape(-1)
        flat = _roll_one_dim(flat, shifts[0], 0)
        return flat.reshape(orig_shape)
    if len(shifts) != len(dims):
        raise RuntimeError(
            f"shifts and dimensions must align. shifts: {len(shifts)}, dims:{len(dims)}"
        )
    result = self
    for shift, dim in zip(shifts, dims, strict=False):
        result = _roll_one_dim(result, shift, dim)
    return result


@register_aten(["aten::cat", "aten::cat.out"], static_argnums=(1,))
def torch_cat(tensors, dim=0, out=None):
    non_empty = [t for t in tensors if t.numel() > 0]
    if not non_empty:
        return torch.cat(tensors, dim)
    if len(non_empty) == 1:
        return non_empty[0].clone()
    return torch.cat(non_empty, dim)


@register_aten(
    ["aten::stack", "aten::stack.out", "aten::_stack", "aten::_stack.out"],
    static_argnums=(1,),
)
def torch_stack(tensors, dim=0, out=None):
    return torch.stack(tensors, dim)


# @register_aten(["aten::vstack", "aten::vstack.out"])
def torch_vstack(tensors):
    return torch.vstack(tensors)


@register_aten(
    ["aten::triu", "aten::triu.out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_triu(m, diagonal=0, out=None):
    return torch.triu(m, diagonal)


@register_aten(
    ["aten::tril", "aten::tril.out"],
    static_argnums=(1,),
    validate_static_inputs=_validate_float_promoting_out,
)
def torch_tril(x, diagonal=0, out=None):
    return torch.tril(x, diagonal)


@register_aten(["aten::flip", "aten::flip.out"], static_argnums=(1,))
def torch_flip(x, dims, out=None):
    return torch.flip(x, dims)


@register_aten(["aten::constant_pad_nd"], static_argnums=(1, 2))
def torch_constant_pad_nd(input, pad, value=0):
    return torch.nn.functional.pad(input, pad, mode="constant", value=value)


def _check_reflection_pad_dims(input, valid_dims, padding, expected_pad_len, out=None):
    # A direct ``aten::reflection_padNd`` call must reject wrong-rank inputs and
    # wrong-length padding with the same messages as eager (the nn.* module path
    # is intercepted earlier by ``F.pad``'s own "Padding size N is not supported
    # for MD input tensor" check, so this only fires on the direct-aten path).
    if len(padding) != expected_pad_len:
        raise RuntimeError(
            f"padding size is expected to be {expected_pad_len}, but got: {len(padding)}"
        )
    if input.dim() not in valid_dims:
        lo, hi = valid_dims
        raise RuntimeError(
            f"Expected {lo}D or {hi}D (batch mode) tensor with possibly 0 batch "
            f"size and other non-zero dimensions for input, but got: {list(input.shape)}"
        )


@register_aten(
    ["aten::reflection_pad1d", "aten::reflection_pad1d.out"],
    static_argnums=(1,),
)
def torch_reflection_pad1d(input, padding, out=None):
    _check_reflection_pad_dims(input, (2, 3), padding, 2)
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(
    ["aten::reflection_pad2d", "aten::reflection_pad2d.out"],
    static_argnums=(1,),
)
def torch_reflection_pad2d(input, padding, out=None):
    _check_reflection_pad_dims(input, (3, 4), padding, 4)
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(
    ["aten::reflection_pad3d", "aten::reflection_pad3d.out"],
    static_argnums=(1,),
)
def torch_reflection_pad3d(input, padding, out=None):
    _check_reflection_pad_dims(input, (4, 5), padding, 6)
    return torch.nn.functional.pad(input, padding, mode="reflect")


@register_aten(["aten::replication_pad1d", "aten::replication_pad1d.out"], static_argnums=(1,))
def torch_replication_pad1d(input, padding, out=None):
    return torch.nn.functional.pad(input, padding, mode="replicate")


@register_aten(["aten::replication_pad2d", "aten::replication_pad2d.out"], static_argnums=(1,))
def torch_replication_pad2d(input, padding, out=None):
    return torch.nn.functional.pad(input, padding, mode="replicate")


@register_aten(
    ["aten::replication_pad3d", "aten::replication_pad3d.out"],
    static_argnums=(1,),
)
def torch_replication_pad3d(input, padding, out=None):
    return torch.nn.functional.pad(input, padding, mode="replicate")


# Note: aten::_pad_circular is not registered here because it's marked as
# CompositeImplicitAutograd in PyTorch. Registering it would intercept the operation
# before PyTorch can decompose it into CopySlices operations, breaking the autograd
# chain. By not registering it, PyTorch naturally decomposes circular padding and
# preserves grad_fn=<CopySlices>.


@register_aten(["aten::repeat", "aten::repeat.out"], static_argnums=(1,), output_params=("out",))
def torch_repeat(x, repeats, out=None):
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
        "aten::repeat_interleave.Tensor_out",
    ],
    static_argnames=("output_size",),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    # output_size = sum(repeats) is value-derived, so re-lower every call. The
    # arange index is built inside the traced compute fn, so it is a graph
    # constant, not an extra runtime input.
    relower_every_call=True,
)
def torch_repeat_interleave_tensor(repeats, output_size=None, out=None):
    """releat_interleave.Tensor variant is not supported in MLIR.

    Custom decomposition and preprocessing needed.
    """
    output_size = get_repeat_interleave_output_size(repeats, output_size=output_size)

    def repeat_interleave_fn(repeats, output_size=None):
        indices = torch.arange(len(repeats), device=repeats.device, dtype=torch.long)
        return compute_repeat_interleave(indices, repeats, output_size=output_size)

    return repeat_interleave_fn, (repeats,), {"output_size": output_size}
