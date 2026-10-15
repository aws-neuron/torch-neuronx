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
    """Determine whether a dtype requires a CPU bounce.

    Args:
        dtype (torch.dtype | None): Dtype to evaluate.

    Returns:
        bool: Whether transfers or casts for the dtype must use CPU fallback.

    Note:
        ``torch.float64`` always bounces. ``torch.int64`` bounces only when neither
        native int64 nor packed int64 mode is enabled.
    """
    if is_native_int64_enabled():
        return dtype == torch.float64
    from torch_neuronx.int64_policy import is_packed_int64_mode

    if is_packed_int64_mode():
        return dtype == torch.float64
    return dtype in (torch.float64, torch.int64)


def validate_neuron_memory_format(memory_format: torch.memory_format | None, op_name: str) -> None:
    """Validate a requested Neuron tensor memory format.

    Args:
        memory_format (torch.memory_format | None): Memory format to validate.
        op_name (str): Operation name included in an error message.

    Raises:
        RuntimeError: If a non-``None`` format is not contiguous, preserve,
            channels-last, or channels-last-3d.
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
    """Make a tensor contiguous through the appropriate device path.

    Args:
        t (torch.Tensor): Tensor whose layout to inspect.

    Returns:
        torch.Tensor: The input tensor if contiguous, otherwise a contiguous tensor.

    Note:
        Neuron tensors use the internal contiguous operation to avoid redispatch
        recursion; other tensors use ``Tensor.contiguous``.
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
    """Copy a CPU tensor to a Neuron device with the requested dtype.

    Args:
        cpu_src (torch.Tensor): CPU tensor to transfer.
        target_device (torch.device): Neuron device that receives the tensor.
        target_dtype (torch.dtype): Dtype of the transferred tensor.
        non_blocking (bool, optional): Whether to request an asynchronous copy.
            Defaults to ``False``.

    Returns:
        torch.Tensor: Neuron tensor with the source shape, values, and logical strides.

    Raises:
        RuntimeError: If the source is not on CPU, the target is not a Neuron device,
            or a native transfer fails.

    Note:
        The source is cast and made contiguous on CPU before the raw NRT transfer.
        Non-contiguous strides are restored on device, and packed int64 uses physical
        int32 pairs so scalar and strided layouts are preserved.
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
            _C._nki_copy_strided_neuron_to_neuron(dst_physical, strided_phys)
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
        _C._nki_copy_strided_neuron_to_neuron(dst, strided)
        dst = strided
    return dst


def copy_neuron_to_cpu(
    neuron_src: torch.Tensor, target_dtype: torch.dtype | None = None, non_blocking: bool = False
) -> torch.Tensor:
    """Copy a Neuron tensor to CPU and optionally cast it.

    Args:
        neuron_src (torch.Tensor): Neuron tensor to transfer.
        target_dtype (torch.dtype, optional): CPU result dtype. Defaults to ``None``.
        non_blocking (bool, optional): Whether to use pinned memory for an asynchronous
            copy. Defaults to ``False``.

    Returns:
        torch.Tensor: Contiguous CPU tensor in the requested or source dtype.

    Raises:
        RuntimeError: If the source is not on a Neuron device or a native transfer fails.

    Note:
        The raw transfer preserves the source dtype before any CPU cast. Packed int64
        downloads physical int32 pairs and restores the logical shape.
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
    """Prepare a source tensor for a Neuron-to-Neuron copy.

    Args:
        src (torch.Tensor): Source tensor to prepare.
        dst (torch.Tensor): Destination tensor whose shape and layout are required.

    Returns:
        torch.Tensor: The source tensor or a contiguous copy of it.

    Raises:
        RuntimeError: If shapes differ or the destination layout is unsupported.
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
    """Copy or cast a source tensor into a Neuron destination in place.

    Args:
        dst (torch.Tensor): Neuron tensor to update.
        src (torch.Tensor): Neuron tensor providing values.

    Raises:
        RuntimeError: If either tensor is not on a Neuron device or shape, device, or
            native-copy constraints are not satisfied.
        TypeError: If an up-to-32-bit dtype cast targets a noncontiguous destination; that path
            currently passes an unsupported keyword to the device-cast helper.

    Note:
        Matching dtypes with a contiguous destination use a direct device copy.
        Policy-controlled 64-bit casts use CPU fallback; up-to-32-bit casts with contiguous
        destinations use a device kernel. Noncontiguous same-dtype and 64-bit-converted values
        use a strided device copy or CPU fallback. An up-to-32-bit dtype change to a
        noncontiguous destination currently fails as described above. Successful device-only
        paths are stream ordered and may return before completion.
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
