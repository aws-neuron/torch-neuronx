"""Utilities for analyzing and transforming tensor layouts"""

from enum import Enum, auto

import torch


def match_layout(
    source: torch.Tensor, reference: torch.Tensor, out: torch.Tensor = None
) -> torch.Tensor:
    """Return the source when strides match, otherwise copy it to a destination.

    Args:
        source (torch.Tensor): Tensor containing the values to return.
        reference (torch.Tensor): Tensor whose strides determine whether a copy is needed.
        out (torch.Tensor, optional): Destination used when a copy is required;
            its layout is not validated. Defaults to ``None`` (allocate from
            ``reference``).

    Returns:
        torch.Tensor: ``source`` when strides match, otherwise the copied destination.
    """
    if source.stride() == reference.stride():
        return source
    if out is None:
        out = torch.empty_like(reference)
    out.copy_(source)
    return out


class KernelType(Enum):
    """Represent the available tensor-copy kernel strategies."""

    RANK2 = auto()  # One outer loop + memcpy
    RANK3 = auto()  # Two outer loops + memcpy
    GENERIC = auto()  # Element-wise copy


def collapse_for_bulk_copy(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Collapse broadcast dimensions and unit-stride dense suffixes.

    Args:
        t (torch.Tensor): Tensor whose shape and strides are collapsed.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: Collapsed shape and strides.

    Note:
        Dense suffixes collapse only when the resulting innermost stride is one.
    """
    shape = list(t.shape)
    stride = list(t.stride())

    new_shape, new_stride = [], []

    # walking right → left
    for dim in reversed(range(len(shape))):
        sz, st = shape[dim], stride[dim]

        if not new_shape:  # first (innermost) axis
            new_shape.append(sz)
            new_stride.append(st)
            continue

        inn_sz, inn_st = new_shape[-1], new_stride[-1]

        # Can we fuse this axis into the one we already hold?
        fuse = False
        if sz == 1:  # broadcast never moves the ptr
            fuse = True
        elif st == inn_sz * inn_st and inn_st == 1:
            # dense-suffix AND innermost block is truly contiguous
            fuse = True

        if fuse:
            new_shape[-1] *= sz  # extend the contiguous block
            # keep the stride (=inn_st, guaranteed 1 here except for broadcast)
        else:
            new_shape.append(sz)
            new_stride.append(st)

    new_shape.reverse()
    new_stride.reverse()
    return tuple(new_shape), tuple(new_stride)


def select_kernel(shape: tuple[int, ...], stride: tuple[int, ...]) -> KernelType:
    """Select a copy kernel for a collapsed shape and stride.

    Args:
        shape (tuple[int, ...]): Collapsed tensor shape.
        stride (tuple[int, ...]): Collapsed tensor strides.

    Returns:
        KernelType: Copy strategy for the collapsed layout.

    Raises:
        RuntimeError: If the layout is scalar or unexpectedly contiguous rank one.
    """
    # If there are no dimensions (scalar), this is an error
    # Scalars are always contiguous and should never reach kernel selection
    if not shape:
        raise RuntimeError(
            "Unexpected scalar tensor in select_kernel. "
            "Scalars should be handled by is_contiguous() check."
        )

    # CRITICAL: Check if the rightmost (innermost) stride is 1
    # Without this, we cannot do bulk copies
    if stride[-1] != 1:
        return KernelType.GENERIC  # Must use element-wise copy

    # Now we know we can do bulk copies, determine the kernel based on rank
    rank = len(shape)

    # Check for rank-1 case - if we collapsed to rank-1 with unit stride,
    # the tensor should already be contiguous and handled earlier
    if rank == 1 and stride[0] == 1:
        raise RuntimeError(
            f"Unexpected rank-1 contiguous tensor in select_kernel with shape {shape}. "
            "Should be handled by is_contiguous() check."
        )

    # Map rank to kernel type
    rank_to_kernel = {
        2: KernelType.RANK2,  # One outer loop + memcpy
        3: KernelType.RANK3,  # Two outer loops + memcpy (will use generic implementation)
    }

    return rank_to_kernel.get(rank, KernelType.GENERIC)  # Higher ranks → element-wise walk


def get_collapsed_rank(shape: tuple[int, ...]) -> int:
    """Return the effective rank of a collapsed shape.

    Args:
        shape (tuple[int, ...]): Collapsed tensor shape.

    Returns:
        int: Number of dimensions in ``shape``.
    """
    return len(shape)


def has_unit_stride(strides: tuple[int, ...]) -> bool:
    """Check whether any tensor stride is one.

    Args:
        strides (tuple[int, ...]): Tensor strides to inspect.

    Returns:
        bool: ``True`` when at least one stride is one.
    """
    return any(s == 1 for s in strides)


def get_contiguous_block_size(shape: list[int], strides: list[int]) -> int:
    """Calculate the element count in a tensor's contiguous suffix.

    Args:
        shape (list[int]): Tensor shape.
        strides (list[int]): Tensor strides.

    Returns:
        int: Number of trailing elements copyable together, or ``0`` when no
            contiguous suffix exists.
    """
    if not shape or not strides:
        return 0

    # Start from the last dimension
    if strides[-1] != 1:
        return 0

    block_size = shape[-1]

    # Work backwards to find the dense suffix
    for i in range(len(shape) - 2, -1, -1):
        expected_stride = shape[i + 1] * strides[i + 1]
        if strides[i] != expected_stride:
            break
        block_size *= shape[i]

    return block_size
