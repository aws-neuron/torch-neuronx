"""Provide packed physical storage helpers for ``torch.int64`` tensors.

The sole supported mapping represents each logical ``torch.int64`` value as
low and high ``torch.int32`` words in a trailing lane of two. Nonscalar shapes
map from ``S`` to ``(*S, 2)``, while a scalar maps from ``()`` to ``(2,)``.
"""

from __future__ import annotations

import torch

# Logical → physical dtype mapping
_PACKED_PHYSICAL_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.int64: torch.int32,
}


def packed_physical_dtype(dtype: torch.dtype) -> torch.dtype | None:
    """Return the physical dtype for a packed logical dtype.

    Args:
        dtype (torch.dtype): Logical dtype to query.

    Returns:
        torch.dtype or None: Packed physical dtype, or ``None`` when unmapped.
    """
    return _PACKED_PHYSICAL_DTYPE.get(dtype)


def is_packed_logical_dtype(dtype: torch.dtype) -> bool:
    """Check whether a dtype uses packed storage on a Neuron device.

    Args:
        dtype (torch.dtype): Logical dtype to query.

    Returns:
        bool: ``True`` when ``dtype`` has a packed physical mapping.
    """
    return dtype in _PACKED_PHYSICAL_DTYPE


def packed_physical_layout(
    dtype: torch.dtype,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Compute physical layout metadata for a packed logical tensor.

    Args:
        dtype (torch.dtype): Packed logical dtype.
        shape (tuple[int, ...]): Logical tensor shape.
        stride (tuple[int, ...]): Logical tensor strides.
        storage_offset (int, optional): Logical storage offset. Defaults to ``0``.

    Returns:
        tuple: Physical shape, stride, and storage offset.

    Raises:
        TypeError: If ``dtype`` has no supported packed layout.

    Note:
        For ``torch.int64``, this appends a two-word lane, doubles strides and
        offset, and maps a scalar shape to ``(2,)``.
    """
    if dtype == torch.int64:
        if len(shape) == 0:
            # Scalar: append the trailing lane dim of 2 to the empty logical
            # shape -> (2,). NOT (1, 2): that collides with a 1-D (1,) tensor
            # (which also maps to (1, 2)) and disagrees with the output-packing
            # (stack of two 0-dim words -> (2,)) and logical_view (drops last dim:
            # (2,) -> ()) conventions. (2,) keeps the scalar path consistent and
            # unambiguous.
            return (2,), (1,), storage_offset * 2
        return (
            (*tuple(shape), 2),
            (*tuple(s * 2 for s in stride), 1),
            storage_offset * 2,
        )
    raise TypeError(f"dtype {dtype} does not support packed physical layout")


def is_packed_physical_view(physical: torch.Tensor, logical: torch.Tensor) -> bool:
    """Check whether a tensor matches another tensor's packed physical layout.

    Args:
        physical (torch.Tensor): Candidate physical tensor.
        logical (torch.Tensor): Logical tensor whose layout is expected.

    Returns:
        bool: ``True`` when dtype and shape match the computed physical layout.

    Note:
        Only ``torch.int64`` is packed: nonscalar tensors map to ``torch.int32``
        with a trailing lane of two, while scalars map from ``()`` to ``(2,)``.
        This checks only dtype and shape, so it also works with meta tensors.
    """
    if not (isinstance(physical, torch.Tensor) and isinstance(logical, torch.Tensor)):
        return False
    phys_dtype = packed_physical_dtype(logical.dtype)
    if phys_dtype is None or physical.dtype != phys_dtype or logical.dtype == phys_dtype:
        return False
    phys_shape, _, _ = packed_physical_layout(
        logical.dtype, shape=tuple(logical.shape), stride=tuple(logical.stride())
    )
    return tuple(physical.shape) == tuple(phys_shape)


def packed_physical_view(t: torch.Tensor) -> torch.Tensor:
    """Create a zero-copy physical view of a packed logical tensor.

    Args:
        t (torch.Tensor): Packed logical tensor to reinterpret.

    Returns:
        torch.Tensor: Storage-sharing tensor with the physical dtype and layout.

    Raises:
        TypeError: If the tensor dtype has no packed physical mapping.

    Note:
        An ``int64`` input becomes ``int32`` with a trailing dimension of two.
    """
    physical_dtype = packed_physical_dtype(t.dtype)
    if physical_dtype is None:
        raise TypeError(f"dtype {t.dtype} does not have a packed physical representation")

    if t.dtype == physical_dtype:
        return t

    shape, stride, offset = packed_physical_layout(
        t.dtype,
        shape=tuple(int(d) for d in t.shape),
        stride=tuple(int(s) for s in t.stride()),
        storage_offset=int(t.storage_offset()),
    )

    # Create a new tensor sharing the same storage with reinterpreted metadata
    out = torch.empty([], dtype=physical_dtype, device=t.device)
    out.set_(t.untyped_storage(), offset, shape, stride)
    return out


def packed_logical_view(
    physical: torch.Tensor, logical_dtype: torch.dtype, logical_shape: tuple[int, ...]
) -> torch.Tensor:
    """Reinterpret a contiguous physical tensor as its logical dtype and shape.

    Args:
        physical (torch.Tensor): Contiguous tensor in the physical dtype.
        logical_dtype (torch.dtype): Logical dtype to restore.
        logical_shape (tuple[int, ...]): Logical shape to restore.

    Returns:
        torch.Tensor: Zero-copy logical view sharing ``physical`` storage.

    Note:
        For the supported packed mapping, viewing ``int32`` ``(*S, 2)`` storage
        as ``int64`` leaves a trailing singleton ``(*S, 1)``; the final reshape
        removes that lane and restores ``S``.
    """
    if physical.dtype == logical_dtype:
        return physical.reshape(tuple(logical_shape))
    return physical.view(logical_dtype).reshape(tuple(logical_shape))


def packed_int64_low_word_alias(t: torch.Tensor) -> torch.Tensor:
    """Return a strided view of the low words in packed ``int64`` storage.

    Args:
        t (torch.Tensor): Packed ``int64`` tensor.

    Returns:
        torch.Tensor: Storage-sharing ``int32`` view with the logical shape.

    Raises:
        TypeError: If ``t`` is not an ``int64`` tensor.

    Note:
        Each returned stride is twice the corresponding logical stride, so
        elements in non-size-one dimensions skip their high words.
    """
    if t.dtype != torch.int64:
        raise TypeError(f"expected int64 tensor, got {t.dtype}")

    physical = packed_physical_view(t)
    if t.ndim == 0:
        # Scalar: physical is (2,) with stride (1,); low word is lane 0.
        return torch.as_strided(physical, (), (), int(physical.storage_offset()))
    # Non-scalar: take slice [..., 0] of the physical view
    # physical shape: (*logical_shape, 2), stride: (*logical_stride*2, 1)
    # low word: shape=logical_shape, stride=physical.stride()[:-1], offset=physical.storage_offset()
    return torch.as_strided(
        physical,
        tuple(int(d) for d in t.shape),
        tuple(int(s) for s in physical.stride()[:-1]),
        int(physical.storage_offset()),
    )


def packed_int64_high_word_alias(t: torch.Tensor) -> torch.Tensor:
    """Return a strided view of the high words in packed ``int64`` storage.

    Args:
        t (torch.Tensor): Packed ``int64`` tensor.

    Returns:
        torch.Tensor: Storage-sharing ``int32`` view with the logical shape.

    Raises:
        TypeError: If ``t`` is not an ``int64`` tensor.
    """
    if t.dtype != torch.int64:
        raise TypeError(f"expected int64 tensor, got {t.dtype}")

    physical = packed_physical_view(t)
    if t.ndim == 0:
        return torch.as_strided(physical, (), (), int(physical.storage_offset()) + 1)
    return torch.as_strided(
        physical,
        tuple(int(d) for d in t.shape),
        tuple(int(s) for s in physical.stride()[:-1]),
        int(physical.storage_offset()) + 1,
    )


def packed_int64_execution_view(t: torch.Tensor, *, contiguous: bool = True) -> torch.Tensor:
    """Extract low words from packed ``int64`` storage for graph execution.

    Args:
        t (torch.Tensor): Packed ``int64`` tensor.
        contiguous (bool, optional): Copy into contiguous ``int32`` storage.
            Defaults to ``True``; ``False`` returns a zero-copy alias.

    Returns:
        torch.Tensor: ``int32`` tensor with the input's logical shape.

    Raises:
        TypeError: If ``t`` is not an ``int64`` tensor.

    Note:
        Contiguous Neuron-device inputs route through ``nki_strided_copy_``, which
        uses an NKI gather kernel for eligible uniform-stride aliases and otherwise
        falls back to an NRT copy. Non-contiguous inputs use ``copy_``; CPU and meta
        inputs call ``contiguous`` on the low-word alias.
    """
    if not contiguous:
        return packed_int64_low_word_alias(t)

    if t.device.type == "neuron":
        alias = packed_int64_low_word_alias(t)
        dst = torch.empty(tuple(int(d) for d in t.shape), dtype=torch.int32, device=t.device)
        if t.is_contiguous():
            # Fast path: dense row-major lane alias → NKI flat-stream kernel.
            # Routed through the op so the kernel is the single call site.
            torch.ops.neuron.nki_strided_copy_(alias, dst)
        else:
            # Non-contiguous int64 (broadcast / slice / permuted): the alias is
            # not a dense pair stream. A plain int32→int32 copy of the lane alias
            # follows its strides correctly (incl. stride-0 broadcast) via the
            # general copy path; no int64 packed logic, so no recursion.
            dst.copy_(alias)
        return dst

    # CPU/meta fallback
    return packed_int64_low_word_alias(t).contiguous()


def materialize_int64_from_i32(
    dst: torch.Tensor,
    low_words: torch.Tensor,
    *,
    zero_extend: bool = False,
) -> torch.Tensor:
    """Write an ``int32`` execution result into packed ``int64`` storage.

    Args:
        dst (torch.Tensor): Destination ``int64`` tensor modified in place.
        low_words (torch.Tensor): ``int32`` results matching ``dst`` shape.
        zero_extend (bool, optional): Fill high words with zero instead of
            sign-extending. Defaults to ``False``.

    Returns:
        torch.Tensor: The modified ``dst`` tensor.

    Raises:
        TypeError: If ``dst`` is not ``int64`` or ``low_words`` is not ``int32``.
        ValueError: If source and destination shapes differ.

    Note:
        Sign extension uses an arithmetic right shift by 31.
    """
    if dst.dtype != torch.int64:
        raise TypeError(f"expected int64 dst, got {dst.dtype}")
    if low_words.dtype != torch.int32:
        raise TypeError(f"expected int32 low_words, got {low_words.dtype}")

    logical_shape = tuple(int(d) for d in dst.shape)
    if tuple(int(d) for d in low_words.shape) != logical_shape:
        raise ValueError(
            f"shape mismatch: dst={logical_shape}, "
            f"low_words={tuple(int(d) for d in low_words.shape)}"
        )

    high_words = torch.zeros_like(low_words) if zero_extend else low_words >> 31

    # Stack into (*, 2) and copy into physical view (contiguous-to-contiguous)
    packed_words = torch.stack((low_words, high_words), dim=-1)
    physical = packed_physical_view(dst)
    physical.copy_(packed_words)

    return dst
