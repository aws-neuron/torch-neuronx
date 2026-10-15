"""Packed physical storage for 64-bit dtypes on Neuron.

Neuron HW doesn't natively support 64-bit tensor payloads. This module provides
the layout computation and view helpers to represent logical 64-bit tensors in
narrower 32-bit physical storage.

Layout rules:
  - float64 → float32: same shape, same strides (lossy truncation)
  - int64 → int32: appends trailing dimension of 2, strides doubled
    Each int64 element is stored as [low_word, high_word] in little-endian order.
  - complex128 → complex64: same shape (lossy truncation)
"""

from __future__ import annotations

import torch

# Logical → physical dtype mapping
_PACKED_PHYSICAL_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.int64: torch.int32,
}


def packed_physical_dtype(dtype: torch.dtype) -> torch.dtype | None:
    """Return the physical dtype for a packed logical dtype, or None if not packed."""
    return _PACKED_PHYSICAL_DTYPE.get(dtype)


def is_packed_logical_dtype(dtype: torch.dtype) -> bool:
    """Return True if dtype requires packed physical storage on Neuron."""
    return dtype in _PACKED_PHYSICAL_DTYPE


def packed_physical_layout(
    dtype: torch.dtype,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    storage_offset: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Compute the physical (shape, stride, storage_offset) for a packed tensor.

    For int64: appends trailing dim of 2, doubles all strides, doubles offset.
    For float64/complex128: shape and strides unchanged (same element count).

    Returns:
        (physical_shape, physical_stride, physical_storage_offset)
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
    """Return True if ``physical`` is the packed physical view of ``logical``.

    i.e. ``logical`` has a packed physical representation (int64/float64/
    complex128) and ``physical`` matches its physical dtype and shape per
    :func:`packed_physical_layout` (e.g. int64 ``S`` ↔ int32 ``(*S, 2)``, and
    scalar ``()`` ↔ ``(2,)``).

    Single source of truth for "is this int32 tensor the packed view of that
    logical tensor?", shared by the graph-boundary gather (``_gather_arg``), the
    output reinterpret (``_reinterpret_packed_int64_output``), and the
    provided/inplace-output preparation (``_prepare_execution_tensors``). Works
    on meta tensors (uses only dtype/shape).
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

    The returned tensor shares storage with the input but has the physical
    dtype and layout. For int64 tensors, this means dtype=int32 with an
    extra trailing dimension of 2.

    Raises:
        TypeError: if the tensor's dtype doesn't have a packed physical dtype.
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
    """Reinterpret a contiguous physical tensor back to its logical dtype and shape.

    Inverse of :func:`packed_physical_view`. ``torch.view(logical_dtype)`` alone
    is not enough for int64/complex128: viewing an int32 ``(*, 2)`` tensor as
    int64 collapses the trailing lane dim to ``(*, 1)`` rather than dropping it,
    so the result must be reshaped back to ``logical_shape``. For float64 (same
    element count) this reshape is a no-op.

    Args:
        physical: contiguous tensor in the physical dtype (e.g. int32 pairs).
        logical_dtype: target logical dtype (e.g. torch.int64).
        logical_shape: the logical tensor's shape to restore.

    Returns:
        Tensor with dtype=logical_dtype and shape=logical_shape, sharing
        ``physical``'s storage (zero-copy reinterpretation).
    """
    if physical.dtype == logical_dtype:
        return physical.reshape(tuple(logical_shape))
    return physical.view(logical_dtype).reshape(tuple(logical_shape))


def packed_int64_low_word_alias(t: torch.Tensor) -> torch.Tensor:
    """Return a strided int32 view selecting only the low words from packed int64 storage.

    The result has the same logical shape as the input but dtype=int32 and
    stride pattern that skips the high words (stride * 2, offset on lane 0).
    This view is non-contiguous when the original has >0 dimensions.
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
    """Return a strided int32 view selecting only the high words from packed int64 storage."""
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
    """Extract the low 32 bits from a packed int64 tensor for execution.

    When contiguous=True (default), returns a contiguous int32 tensor with
    the same logical shape, containing a copy of the low words. This is the
    tensor that gets passed to compiled graphs.

    When contiguous=False, returns the non-contiguous low-word alias (no copy).

    Args:
        t: int64 tensor in packed storage.
        contiguous: If True, perform device-side DMA gather to produce
            contiguous output.

    Returns:
        int32 tensor with same logical shape as input.
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
    """Write int32 execution result back into packed int64 storage.

    Takes the int32 result from a compiled graph and packs it into the
    (low, high) word pairs expected by the packed int64 layout.

    The high word is either sign-extended (default) or zero-extended from
    the low word.

    Args:
        dst: Target int64 tensor (will be modified in-place via its packed view).
        low_words: int32 tensor with execution results (same logical shape as dst).
        zero_extend: If True, high word is zero. If False, sign-extend (arithmetic
            right shift by 31).

    Returns:
        dst tensor (modified in-place).
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
