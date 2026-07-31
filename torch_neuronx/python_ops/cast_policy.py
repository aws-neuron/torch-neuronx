"""Centralized copy/cast policy helpers.

This module consolidates rules for when to bounce through CPU (only for 64-bit
types) and how to perform raw transfers without re-dispatch recursion. It
provides small, well-scoped helpers that ops like `aten::copy_` and
`aten::_to_copy` can use consistently.

Key policy:
- No CPU bounce for dtype casting except when casting to or from 64-bit dtypes
  (torch.float64 or torch.int64).
- Avoid calling `.to(...)` on device tensors for casting. Use device-side
  kernels for <=32-bit casts or explicit CPU bounce helpers for 64-bit.
- Prefer `_C._nrt_copy_*` helpers for raw transfers instead of `.copy_` or `.to`.
- Enforce memory format rules: allow contiguous and preserve only.

Imports of heavy or circular modules are local to functions.
"""

from __future__ import annotations

import torch

from torch_neuronx.python_ops import io_tensor
from torch_neuronx.utils import is_native_int64_enabled


def bounce_64bit(dtype: torch.dtype | None) -> bool:
    """Return True if dtype is a 64-bit type needing CPU bounce on Neuron.

    Args:
        dtype: torch dtype or None

    Returns:
        bool: True for torch.float64; True for torch.int64 only in cpu_bounce mode.
    """
    if is_native_int64_enabled():
        return dtype == torch.float64
    from torch_neuronx.int64_policy import is_packed_int64_mode

    if is_packed_int64_mode():
        return dtype == torch.float64
    return dtype in (torch.float64, torch.int64)


def validate_neuron_memory_format(memory_format: torch.memory_format | None, op_name: str) -> None:
    """Validate memory format constraints for Neuron tensors.

    Supported formats: contiguous, preserve, channels_last, channels_last_3d.
    Raise a clear error otherwise.
    """
    if memory_format is None:
        return
    if memory_format in (
        torch.contiguous_format,
        torch.preserve_format,
        torch.channels_last,
        torch.channels_last_3d,
    ):
        return
    raise RuntimeError(f"{op_name}: Neuron tensors do not support {memory_format} memory format")


def ensure_contiguous_on_device(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous, using device-appropriate path.

    For Neuron tensors, use the internal contiguous op to avoid redispatch
    recursion; for CPU tensors, use the standard `.contiguous()`.
    """
    if t.is_contiguous():
        return t
    if t.device.type == "neuron":
        # Local import to avoid cycles
        from torch_neuronx.python_ops.contiguous import contiguous_internal

        return contiguous_internal(t, torch.contiguous_format)
    return t.contiguous()


def copy_cpu_to_neuron(
    cpu_src: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy CPU tensor to Neuron, casting on CPU first if needed.

    Ensures the CPU tensor is contiguous and in the target dtype, then allocates
    a Neuron tensor and performs a raw transfer via NRT copy. If the source had
    non-contiguous strides (e.g. from transpose), the result is re-strided on
    device to match, preserving the same layout as CUDA's .to() behaviour.
    """
    import torch_neuronx._C as _C  # local import
    from torch_neuronx.int64_policy import is_packed_int64_mode

    # Packed int64: reinterpret bytes as int32 pairs, bulk upload, preserve strides
    if is_packed_int64_mode() and target_dtype == torch.int64:
        from torch_neuronx.packed_storage import packed_physical_view

        tmp = cpu_src if cpu_src.dtype == torch.int64 else cpu_src.to(torch.int64)
        src_strides = tmp.stride()
        src_is_contiguous = tmp.is_contiguous()
        if not src_is_contiguous:
            tmp = tmp.contiguous()
        # Reinterpret contiguous int64 -> int32 pairs. Use packed_physical_view
        # (not tmp.view(torch.int32)): a 0-dim scalar int64 () has physical shape
        # (1, 2), but tmp.view(torch.int32) raises on a 0-dim tensor ("self.dim()
        # cannot be 0 to view Long as Int"). packed_physical_view handles the
        # scalar case and matches dst_physical's layout.
        physical_cpu = packed_physical_view(tmp)
        # Allocate contiguous int64 device tensor, upload via physical view
        dst = io_tensor.empty(tmp.shape, dtype=torch.int64, device=target_device)
        dst_physical = packed_physical_view(dst)
        _C._nrt_copy_cpu_to_neuron_tensor(physical_cpu, dst_physical, non_blocking=non_blocking)
        if not src_is_contiguous:
            strided = torch.empty_strided(
                cpu_src.shape, src_strides, dtype=torch.int64, device=target_device
            )
            strided_phys = packed_physical_view(strided)
            _C._nrt_copy_strided_neuron_to_neuron(dst_physical, strided_phys)
            dst = strided
        return dst

    tmp = cpu_src
    if tmp.dtype != target_dtype:
        tmp = tmp.to(target_dtype)
    src_strides = tmp.stride()
    src_is_contiguous = tmp.is_contiguous()
    if not src_is_contiguous:
        tmp = tmp.contiguous()
    dst = io_tensor.empty(tmp.shape, dtype=tmp.dtype, device=target_device)
    _C._nrt_copy_cpu_to_neuron_tensor(tmp, dst, non_blocking=non_blocking)
    if not src_is_contiguous:
        # Re-apply original strides on device via a strided copy, matching CUDA semantics.
        strided = torch.empty_strided(
            cpu_src.shape, src_strides, dtype=tmp.dtype, device=target_device
        )
        _C._nrt_copy_strided_neuron_to_neuron(dst, strided)
        dst = strided
    return dst


def copy_neuron_to_cpu(
    neuron_src: torch.Tensor, target_dtype: torch.dtype | None = None, non_blocking: bool = False
) -> torch.Tensor:
    """Copy Neuron tensor to CPU; optionally cast on CPU after copy.

    The initial copy is a raw byte transfer preserving the source dtype.
    If `target_dtype` is provided and differs, the CPU cast is performed.
    """
    import torch_neuronx._C as _C  # local import
    from torch_neuronx.int64_policy import is_packed_int64_mode

    # Packed int64: download physical int32 pairs, reinterpret as int64
    if is_packed_int64_mode() and neuron_src.dtype == torch.int64:
        if not neuron_src.is_contiguous():
            neuron_src = ensure_contiguous_on_device(neuron_src)
        from torch_neuronx.packed_storage import packed_physical_view

        src_physical = packed_physical_view(neuron_src)
        cpu_physical = torch.empty(
            src_physical.shape, dtype=torch.int32, device="cpu", pin_memory=non_blocking
        )
        _C._nrt_copy_neuron_to_cpu_tensor(src_physical, cpu_physical, non_blocking=non_blocking)
        # Reinterpret int32 pairs back to int64, restoring the logical shape.
        # (view(int64) alone collapses the trailing lane dim to (*, 1) instead
        # of dropping it — packed_logical_view handles the reshape.)
        from torch_neuronx.packed_storage import packed_logical_view

        cpu_result = packed_logical_view(
            cpu_physical, torch.int64, tuple(int(d) for d in neuron_src.shape)
        )
        if target_dtype is not None and target_dtype != torch.int64:
            cpu_result = cpu_result.to(target_dtype)
        return cpu_result

    if not neuron_src.is_contiguous():
        neuron_src = ensure_contiguous_on_device(neuron_src)
    cpu_tmp = io_tensor.empty(
        neuron_src.shape, dtype=neuron_src.dtype, device="cpu", pin_memory=non_blocking
    )
    _C._nrt_copy_neuron_to_cpu_tensor(neuron_src, cpu_tmp, non_blocking=non_blocking)
    if target_dtype is not None and target_dtype != cpu_tmp.dtype:
        cpu_tmp = cpu_tmp.to(target_dtype)
    return cpu_tmp


def prepare_src_for_neuron_to_neuron_copy(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Prepare source for Neuron->Neuron copy following layout policy.

    - Shapes must match.
    - If both are contiguous: return src.
    - If dst is contiguous but src is not: make src contiguous on device.
    - Otherwise: raise a layout-change error.
    """
    if tuple(src.shape) != tuple(dst.shape):
        raise RuntimeError("Neuron->Neuron copy with size mismatch is not supported")
    if src.is_contiguous() and dst.is_contiguous():
        return src
    if dst.is_contiguous() and not src.is_contiguous():
        return ensure_contiguous_on_device(src)
    raise RuntimeError(
        "Neuron->Neuron copy with layout change is not supported yet. "
        "Please make tensors contiguous first."
    )


def write_neuron_to_neuron(dst: torch.Tensor, src: torch.Tensor) -> None:
    """In-place copy/cast from Neuron tensor `src` to Neuron tensor `dst`.

    - If dtypes match: direct NRT device copy.
    - If 64-bit is involved (src or dst): CPU bounce for cast (Neuron lacks 64-bit).
    - Else: device-side cast kernel for <=32-bit dtypes.
    """
    import torch_neuronx._C as _C  # local import

    if dst.device.type != "neuron" or src.device.type != "neuron":
        raise RuntimeError("write_neuron_to_neuron expects both tensors on Neuron device")

    # Handle non-contiguous destination
    if not dst.is_contiguous():
        src_prepared = ensure_contiguous_on_device(src)
        if dst.dtype != src_prepared.dtype:
            # Need dtype conversion first
            if bounce_64bit(dst.dtype) or bounce_64bit(src_prepared.dtype):
                cpu_tmp = copy_neuron_to_cpu(
                    src_prepared, target_dtype=dst.dtype, non_blocking=False
                )
                src_prepared = copy_cpu_to_neuron(
                    cpu_tmp, dst.device, dst.dtype, non_blocking=False
                )
            else:
                from torch_neuronx.python_ops.device_cast import device_cast_neuron_to_neuron

                temp = io_tensor.empty(src_prepared.shape, dtype=dst.dtype, device=dst.device)
                device_cast_neuron_to_neuron(src_prepared, temp, non_blocking=False)
                src_prepared = temp
        _write_to_noncontiguous_neuron(dst, src_prepared)
        return

    if dst.dtype == src.dtype:
        # Enforce contiguity/layout policy even for same-dtype copies
        src_prepared = prepare_src_for_neuron_to_neuron_copy(src, dst)
        _C._nrt_copy_neuron_to_neuron_tensor(src_prepared, dst)
        return

    if bounce_64bit(dst.dtype) or bounce_64bit(src.dtype):
        # Implicit synchronization
        cpu_tmp = copy_neuron_to_cpu(src, target_dtype=dst.dtype, non_blocking=False)
        _C._nrt_copy_cpu_to_neuron_tensor(cpu_tmp, dst, non_blocking=False)
        return

    # <=32-bit on-device cast
    src_prepared = prepare_src_for_neuron_to_neuron_copy(src, dst)
    # Local import to avoid cycles and heavy deps
    from torch_neuronx.python_ops.device_cast import device_cast_neuron_to_neuron

    device_cast_neuron_to_neuron(src_prepared, dst)


def _get_permutation_from_strides(shape: tuple, strides: tuple) -> list | None:
    """Compute permutation that would produce the given strides from contiguous layout.

    Returns None if strides match contiguous layout (no permutation needed).
    Returns permutation list if tensor is a simple transpose.
    Raises if strides indicate broadcast (stride=0) or other complex patterns.
    """
    ndim = len(shape)
    if ndim == 0:
        return None

    # Check for broadcast (stride = 0)
    if any(s == 0 for s in strides):
        raise RuntimeError("Cannot handle broadcast (stride=0) patterns")

    # Compute expected contiguous strides
    expected = []
    s = 1
    for dim in reversed(shape):
        expected.insert(0, s)
        s *= dim

    if list(strides) == expected:
        return None  # Already contiguous layout

    # Sort dimensions by stride (descending) to get permutation
    # This gives us which dimension of the contiguous tensor maps to which
    indexed_strides = [(strides[i], i) for i in range(ndim)]
    indexed_strides.sort(key=lambda x: x[0], reverse=True)
    perm = [idx for _, idx in indexed_strides]

    return perm


def _write_to_noncontiguous_neuron(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write contiguous src into non-contiguous dst using XLA transpose + flatten copy.

    This avoids CPU bounce by:
    1. Analyzing dst's strides to determine the permutation
    2. Applying inverse permutation to src via XLA
    3. Flattening both to contiguous 1D and doing direct D2D copy
    """
    import torch_neuronx._C as _C

    if not src.is_contiguous():
        raise RuntimeError("_write_to_noncontiguous_neuron expects contiguous src")

    if dst.is_contiguous():
        # Fast path
        _C._nrt_copy_neuron_to_neuron_tensor(src, dst)
        return

    # Try XLA-based approach for transpose patterns
    try:
        perm = _get_permutation_from_strides(tuple(dst.shape), tuple(dst.stride()))

        if perm is not None:
            # Apply permutation to src so its memory layout matches dst's
            src_permuted = _apply_permutation_xla(src, perm)

            # Now flatten both and do direct copy
            dst_storage_size = dst.untyped_storage().size() // dst.element_size()
            dst_flat = dst.as_strided((dst_storage_size,), (1,), 0)
            src_flat = src_permuted.view(-1)

            _C._nrt_copy_neuron_to_neuron_tensor(src_flat, dst_flat)
            return
    except RuntimeError:
        pass  # Fall through to CPU fallback

    # CPU fallback for complex patterns (broadcast, slices with gaps, etc.)
    _write_to_noncontiguous_neuron_cpu_fallback(dst, src)


def _apply_permutation_xla(src: torch.Tensor, perm: list) -> torch.Tensor:
    """Apply permutation to tensor using MLIR transpose."""
    from torch_neuronx.python_ops.kernel_cache import get_or_create_mlir_kernel

    def _mlir_transpose(x, perm, out=None):
        return x.permute(perm)

    kernel = get_or_create_mlir_kernel(
        mlir_fn=_mlir_transpose,
        op_name="noncontiguous_copy_transpose",
        static_argnames=("perm",),
    )

    # Compute output shape after permutation
    out_shape = tuple(src.shape[p] for p in perm)
    out = io_tensor.empty(out_shape, dtype=src.dtype, device=src.device)

    kernel(src, out=out, perm=tuple(perm))

    return out


def _write_to_noncontiguous_neuron_cpu_fallback(dst: torch.Tensor, src: torch.Tensor) -> None:
    """CPU fallback for writing to non-contiguous dst when XLA path doesn't work."""
    import torch_neuronx._C as _C

    # Copy src to CPU (handle both Neuron and CPU src)
    if src.device.type == "neuron":
        src_cpu = io_tensor.empty(src.shape, dtype=src.dtype, device="cpu")
        _C._nrt_copy_neuron_to_cpu_tensor(src, src_cpu, non_blocking=False)
    else:
        src_cpu = src.contiguous() if not src.is_contiguous() else src

    # Copy dst storage to CPU
    dst_storage_size = dst.untyped_storage().size() // dst.element_size()
    dst_flat = dst.as_strided((dst_storage_size,), (1,), 0)
    cpu_buffer = io_tensor.empty((dst_storage_size,), dtype=dst.dtype, device="cpu")
    _C._nrt_copy_neuron_to_cpu_tensor(dst_flat, cpu_buffer, non_blocking=False)

    # Create strided view and copy on CPU
    cpu_strided = cpu_buffer.as_strided(
        size=tuple(dst.shape),
        stride=tuple(dst.stride()),
        storage_offset=dst.storage_offset(),
    )
    cpu_strided.copy_(src_cpu)

    # Copy back to Neuron
    _C._nrt_copy_cpu_to_neuron_tensor(cpu_buffer, dst_flat, non_blocking=False)
