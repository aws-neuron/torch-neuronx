"""Implementation of aten::copy_.

This replaces the C++ registration for `copy_` with a Python implementation
registered via `@neuron_op`. The implementation follows the requested policy:

- CPU -> Neuron:
  - Perform any necessary dtype/layout conversion on CPU first.
  - Move to Neuron, then device-to-device copy into destination.
- Neuron -> CPU:
  - Move to CPU. If dtype/layout need conversion, do so on CPU, then copy
    into destination CPU tensor.
- Neuron -> Neuron:
  - If no layout change is needed, write directly into the destination.
  - Otherwise, raise an unsupported error for now.

Note: This environment may not have a Neuron device. The code paths are written
      so that CPU-only testing will still function (e.g., the Neuron->CPU and
      CPU->Neuron flows rely on standard PyTorch moves).
"""

from __future__ import annotations

import logging
import os

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


def _is_expandable_to(src_shape: torch.Size, dst_shape: torch.Size) -> bool:
    if len(src_shape) > len(dst_shape):
        return False
    return all(
        src_dim == 1 or src_dim == dst_dim
        for src_dim, dst_dim in zip(reversed(src_shape), reversed(dst_shape), strict=False)
    )


def _is_same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Cheap check for self-copy (same underlying storage and offset)."""
    if a.device != b.device:
        return False
    try:
        return (
            a.dtype == b.dtype
            and a.is_set_to(b)
            and a.is_conj() == b.is_conj()
            and a.is_neg() == b.is_neg()
        )
    except Exception:
        # Fallback if storage isn't available; compare data_ptr
        return (
            a.dtype == b.dtype
            and a.data_ptr() == b.data_ptr()
            and a.storage_offset() == b.storage_offset()
            and tuple(a.size()) == tuple(b.size())
            and tuple(a.stride()) == tuple(b.stride())
            and a.is_conj() == b.is_conj()
            and a.is_neg() == b.is_neg()
        )


def _can_use_direct_neuron_to_cpu_copy(
    logical_src: torch.Tensor, transfer_src: torch.Tensor, dst: torch.Tensor
) -> bool:
    """Return whether Neuron-to-CPU copy_ needs no temporary.

    NRT bypasses TensorIterator and cannot perform dtype/layout conversion or
    reconcile mismatched conjugate or negative metadata.
    """
    return (
        tuple(logical_src.shape) == tuple(dst.shape)
        and transfer_src.layout == torch.strided
        and dst.layout == torch.strided
        and not transfer_src.is_quantized
        and not dst.is_quantized
        and tuple(transfer_src.shape) == tuple(dst.shape)
        and transfer_src.dtype == dst.dtype
        and transfer_src.is_contiguous()
        and dst.is_contiguous()
        and transfer_src.is_conj() == dst.is_conj()
        and transfer_src.is_neg() == dst.is_neg()
    )


def _layout_change_needed(src: torch.Tensor | int | float | bool, dst: torch.Tensor) -> bool:
    """Decide if a layout change is needed.

    For now, we conservatively define layout change as non-equal contiguity state
    or non-contiguous tensors. We only support the fast path when both are
    contiguous and shapes match.
    """
    if tuple(src.size()) != tuple(dst.size()):
        return True
    return not (src.is_contiguous() and dst.is_contiguous())


@neuron_op("aten::copy_", disable_dtype_autocast=True, priority=50)
class CopyNeuronImpl(OperationImplementation):
    """Implements aten::copy_ for the Neuron backend."""

    def can_handle(
        self,
        self_tensor: torch.Tensor,
        src: torch.Tensor | int | float | bool,
        non_blocking: bool = False,
        *,
        out=None,
        **kwargs,
    ) -> bool:
        # Handle scalar src -> Neuron dst
        if isinstance(src, (int | float | bool)) and isinstance(self_tensor, torch.Tensor):
            return self_tensor.device.type == "neuron"

        # Handle tensor src where either src or dst is Neuron
        if not (isinstance(self_tensor, torch.Tensor) and isinstance(src, torch.Tensor)):
            return False

        # Handle all neuron-to-neuron and CPU<->Neuron transfers
        return self_tensor.device.type == "neuron" or src.device.type == "neuron"

    def _execute_impl(
        self,
        self_tensor: torch.Tensor,
        src: torch.Tensor | int | float | bool,
        non_blocking: bool = False,
        *,
        out=None,
        **kwargs,
    ) -> ExecutionResult:
        # 0) Trivial no-op: copying a tensor to itself
        if isinstance(src, (int | float | bool)):
            src = torch.tensor(src, dtype=self_tensor.dtype, device="cpu").expand(self_tensor.shape)

        # Check for self-copy
        if _is_same_storage(self_tensor, src):
            return ExecutionResult(success=True, output=self_tensor)

        # 1) Validate shapes - handle broadcasting
        if tuple(self_tensor.size()) != tuple(src.size()):
            if not _is_expandable_to(src.shape, self_tensor.shape):
                return ExecutionResult(
                    success=False,
                    error_msg=(
                        "copy_(): Source and destination sizes must match or be broadcastable. "
                        f"Got src: {tuple(src.size())} and dst: {tuple(self_tensor.size())}"
                    ),
                )
            src = src.expand_as(self_tensor).contiguous()

        dst_dev = self_tensor.device.type
        src_dev = src.device.type

        # Case A: CPU -> Neuron
        if src_dev == "cpu" and dst_dev == "neuron":
            try:
                # Perform dtype/layout conversion on CPU side first
                cpu_prepared = src
                if src.dtype != self_tensor.dtype:
                    cpu_prepared = cpu_prepared.to(self_tensor.dtype)
                if not cpu_prepared.is_contiguous():
                    cpu_prepared = cpu_prepared.contiguous()

                import torch_neuronx._C as _C

                if self_tensor.is_contiguous():
                    # Fast path: direct copy
                    _C._nrt_copy_cpu_to_neuron_tensor(
                        cpu_prepared, self_tensor, non_blocking=non_blocking
                    )
                else:
                    # Slow path: copy to temp, then scatter to non-contiguous dst
                    from torch_neuronx.python_ops import io_tensor

                    from .cast_policy import _write_to_noncontiguous_neuron

                    temp_neuron = io_tensor.empty(
                        cpu_prepared.shape, dtype=cpu_prepared.dtype, device=self_tensor.device
                    )
                    _C._nrt_copy_cpu_to_neuron_tensor(cpu_prepared, temp_neuron, non_blocking=False)
                    _write_to_noncontiguous_neuron(self_tensor, temp_neuron)

                return ExecutionResult(success=True, output=self_tensor)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=f"CPU->Neuron copy failed: {e}")

        # Case B: Neuron -> CPU
        if src_dev == "neuron" and dst_dev == "cpu":
            try:
                import torch_neuronx._C as _C

                from .cast_policy import ensure_contiguous_on_device

                # Make src contiguous on device if needed
                src_prepared = src
                if not src.is_contiguous():
                    src_prepared = ensure_contiguous_on_device(src)

                if _can_use_direct_neuron_to_cpu_copy(src, src_prepared, self_tensor):
                    _C._nrt_copy_neuron_to_cpu_tensor(
                        src_prepared, self_tensor, non_blocking=non_blocking
                    )
                else:
                    # Copy to contiguous CPU buffer first
                    tmp_cpu = torch.empty(
                        src_prepared.shape, dtype=src_prepared.dtype, device="cpu"
                    )
                    _C._nrt_copy_neuron_to_cpu_tensor(src_prepared, tmp_cpu, non_blocking=False)
                    # Materialize conj/neg bits that raw byte copy does not handle.
                    if src.is_conj():
                        tmp_cpu.conj_physical_()
                    if src.is_neg():
                        tmp_cpu.neg_()

                    # Handle dtype conversion on CPU
                    if tmp_cpu.dtype != self_tensor.dtype:
                        tmp_cpu = tmp_cpu.to(self_tensor.dtype)

                    # Copy to dst (CPU handles non-contiguous dst efficiently)
                    self_tensor.copy_(tmp_cpu, non_blocking=non_blocking)

                return ExecutionResult(success=True, output=self_tensor)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=f"Neuron->CPU copy failed: {e}")

        # Case C: Neuron -> Neuron
        if src_dev == "neuron" and dst_dev == "neuron":
            if _layout_change_needed(src, self_tensor):
                try:
                    # Make source contiguous first if needed
                    from .cast_policy import (
                        _write_to_noncontiguous_neuron,
                        ensure_contiguous_on_device,
                    )

                    src_prepared = src
                    if not src.is_contiguous():
                        src_prepared = ensure_contiguous_on_device(src)

                    # If destination is non-contiguous, use scatter path
                    if not self_tensor.is_contiguous():
                        _write_to_noncontiguous_neuron(self_tensor, src_prepared)
                        return ExecutionResult(success=True, output=self_tensor)

                    # Both contiguous now, do direct copy
                    return self._copy_neuron_to_neuron(self_tensor, src_prepared, non_blocking=True)
                except Exception as e:
                    if os.environ.get("TN_DEBUG_COPY") == "1":
                        print(
                            f"[copy_] layout_change failed: {e}, "
                            f"src_contig={src.is_contiguous()} "
                            f"dst_contig={self_tensor.is_contiguous()} "
                            f"src_stride={tuple(src.stride())} "
                            f"dst_stride={tuple(self_tensor.stride())}"
                        )
                    return ExecutionResult(
                        success=False,
                        error_msg=f"Neuron->Neuron copy with layout change failed: {e}",
                    )
            try:
                return self._copy_neuron_to_neuron(self_tensor, src, non_blocking=non_blocking)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=f"Neuron->Neuron copy failed: {e}")

        # Any other combination should not land here (handled by CPU backend)
        return ExecutionResult(
            success=False,
            error_msg=(
                "Unsupported device combination for copy_: "
                f"dst={self_tensor.device}, src={src.device}"
            ),
        )

    def _handle_empty_tensor(
        self,
        self_tensor: torch.Tensor,
        src: torch.Tensor | int | float | bool,
        non_blocking: bool = False,
        **kwargs,
    ) -> ExecutionResult:
        """Handle empty tensors: copy_ on zero-sized tensors is a no-op returning dst.

        Matches PyTorch semantics where copy_ with numel()==0 leaves destination unchanged.
        """
        if isinstance(src, torch.Tensor) and not _is_expandable_to(src.shape, self_tensor.shape):
            return ExecutionResult(
                success=False,
                error_msg=(
                    "copy_(): Source and destination sizes must match or be broadcastable. "
                    f"Got src: {tuple(src.size())} and dst: {tuple(self_tensor.size())}"
                ),
            )
        return ExecutionResult(success=True, output=self_tensor)

    # -------------------- Helpers --------------------
    def _copy_neuron_to_neuron(
        self, dst: torch.Tensor, src: torch.Tensor | int | float | bool, non_blocking: bool = False
    ) -> ExecutionResult:
        """Copy on Neuron between Neuron tensors without unnecessary CPU hops.

        - Same dtype: perform a direct device-to-device copy using NRT helper.
        - Dtype change targeting 64-bit (int64/float64): perform CPU bounce for
          the cast (device lacks 64-bit compute), then copy back to Neuron.
        - Other dtype changes (<=32-bit targets): cast and write on device.
        """
        logger.debug(
            f"Using NRT API based copy for neuron-to-neuron copy_: "
            f"dst_shape={tuple(dst.shape)}, dst_dtype={dst.dtype}, dst_device={dst.device}, "
            f"src_shape={tuple(src.shape)}, src_dtype={src.dtype}, src_device={src.device}, "
            f"non_blocking={non_blocking}"
        )
        try:
            from .cast_policy import write_neuron_to_neuron

            write_neuron_to_neuron(dst, src)
            return ExecutionResult(success=True, output=dst)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
