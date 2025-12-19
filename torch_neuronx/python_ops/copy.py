"""JAX-backed implementation of aten::copy_.

This replaces the C++ registration for `copy_` with a Python implementation
registered via `@neuron_op`. The implementation follows the requested policy:

- CPU -> Neuron:
  - Perform any necessary dtype/layout conversion on CPU first.
  - Move to Neuron, then device-to-device copy into destination.
- Neuron -> CPU:
  - Move to CPU. If dtype/layout need conversion, do so on CPU, then copy
    into destination CPU tensor.
- Neuron -> Neuron:
  - If no layout change is needed, use a small JAX kernel with astype (or
    identity when dtypes match) to write directly into the destination.
  - Otherwise, raise an unsupported error for now.

Note: This environment may not have a Neuron device. The code paths are written
      so that CPU-only testing will still function (e.g., the Neuron->CPU and
      CPU->Neuron flows rely on standard PyTorch moves and our JAX kernel is
      only used for Neuron->Neuron when both tensors are on Neuron).
"""

from __future__ import annotations

import os

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


def _is_same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Cheap check for self-copy (same underlying storage and offset)."""
    try:
        return (
            a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            and a.storage_offset() == b.storage_offset()
            and tuple(a.size()) == tuple(b.size())
        )
    except Exception:
        # Fallback if storage isn't available; compare data_ptr
        return a.data_ptr() == b.data_ptr() and tuple(a.size()) == tuple(b.size())


def _layout_change_needed(src: torch.Tensor, dst: torch.Tensor) -> bool:
    """Decide if a layout change is needed.

    For now, we conservatively define layout change as non-equal contiguity state
    or non-contiguous tensors. We only support the fast path when both are
    contiguous and shapes match.
    """
    if tuple(src.size()) != tuple(dst.size()):
        return True
    return not (src.is_contiguous() and dst.is_contiguous())


@neuron_op("aten::copy_", disable_dtype_autocast=True)
class CopyNeuronImpl(OperationImplementation):
    """Implements aten::copy_ for the Neuron backend using JAX when possible."""

    def can_handle(
        self,
        self_tensor: torch.Tensor,
        src: torch.Tensor,
        non_blocking: bool = False,
        *,
        out=None,
        **kwargs,
    ) -> bool:
        # Only handle cases where either src or dst (self) is a Neuron tensor.
        return (
            isinstance(self_tensor, torch.Tensor)
            and isinstance(src, torch.Tensor)
            and (self_tensor.device.type == "neuron" or src.device.type == "neuron")
        )

    def _execute_impl(
        self,
        self_tensor: torch.Tensor,
        src: torch.Tensor,
        non_blocking: bool = False,
        *,
        out=None,
        **kwargs,
    ) -> ExecutionResult:
        # 0) Trivial no-op: copying a tensor to itself
        if _is_same_storage(self_tensor, src):
            return ExecutionResult(success=True, output=self_tensor)

        # 1) Validate shapes
        if tuple(self_tensor.size()) != tuple(src.size()):
            return ExecutionResult(
                success=False,
                error_msg=(
                    "copy_(): Source and destination sizes must match. "
                    f"Got src: {tuple(src.size())} and dst: {tuple(self_tensor.size())}"
                ),
            )

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

                # Copy to Neuron using runtime API directly (no Tensor.to)
                import torch_neuronx._C as _C

                _C._nrt_copy_cpu_to_neuron_tensor(
                    cpu_prepared, self_tensor, non_blocking=non_blocking
                )
                return ExecutionResult(success=True, output=self_tensor)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=f"CPU->Neuron copy failed: {e}")

        # Case B: Neuron -> CPU
        if src_dev == "neuron" and dst_dev == "cpu":
            try:
                import torch_neuronx._C as _C

                # Fast path: matching dtype and contiguous layout
                if (
                    src.dtype == self_tensor.dtype
                    and src.is_contiguous()
                    and self_tensor.is_contiguous()
                ):
                    _C._nrt_copy_neuron_to_cpu_tensor(src, self_tensor, non_blocking=non_blocking)
                else:
                    # Copy into a temporary CPU buffer first (src dtype/shape), then convert
                    tmp_cpu = torch.empty_like(self_tensor, dtype=src.dtype, device="cpu")
                    _C._nrt_copy_neuron_to_cpu_tensor(src, tmp_cpu, non_blocking=non_blocking)
                    if tmp_cpu.dtype != self_tensor.dtype:
                        tmp_cpu = tmp_cpu.to(self_tensor.dtype)
                    # Use CPU copy for final write (CPU->CPU)
                    if self_tensor.shape != tmp_cpu.shape:
                        self_tensor.resize_(tmp_cpu.shape)
                    self_tensor.copy_(tmp_cpu, non_blocking=non_blocking)
                return ExecutionResult(success=True, output=self_tensor)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=f"Neuron->CPU copy failed: {e}")

        # Case C: Neuron -> Neuron
        if src_dev == "neuron" and dst_dev == "neuron":
            if _layout_change_needed(src, self_tensor):
                # If only the source is non-contiguous (common during backward
                # when grads are expanded), make it contiguous on-device first,
                # then perform the fast device-to-device copy.
                if self_tensor.is_contiguous() and not src.is_contiguous():
                    try:
                        # Avoid redispatch: make contiguous using internal helper
                        from .cast_policy import ensure_contiguous_on_device

                        src_prepared = ensure_contiguous_on_device(src)
                        # In case of D2D without CPU bounces, copies are not blocking.
                        return self._copy_neuron_to_neuron(
                            self_tensor, src_prepared, non_blocking=True
                        )
                    except Exception as e:
                        return ExecutionResult(
                            success=False,
                            error_msg=f"Neuron->Neuron src-contiguous path failed: {e}",
                        )
                if os.environ.get("TN_DEBUG_COPY") == "1":
                    print(
                        (
                            f"[copy_] layout_change src_contig={src.is_contiguous()} "
                            f"dst_contig={self_tensor.is_contiguous()} "
                        )
                        + (
                            f"src_stride={tuple(src.stride())} "
                            f"dst_stride={tuple(self_tensor.stride())}"
                        )
                    )
                return ExecutionResult(
                    success=False,
                    error_msg=(
                        "Neuron->Neuron copy with layout change is not supported yet. "
                        "Please make tensors contiguous first."
                    ),
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
        self, self_tensor: torch.Tensor, src: torch.Tensor, **kwargs
    ) -> ExecutionResult:
        """Handle empty tensors: copy_ on zero-sized tensors is a no-op returning dst.

        Matches PyTorch semantics where copy_ with numel()==0 leaves destination unchanged.
        """
        return ExecutionResult(success=True, output=self_tensor)

    # -------------------- Helpers --------------------
    def _copy_neuron_to_neuron(
        self, dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False
    ) -> ExecutionResult:
        """Copy on Neuron between Neuron tensors without unnecessary CPU hops.

        - Same dtype: perform a direct device-to-device copy using NRT helper.
        - Dtype change targeting 64-bit (int64/float64): perform CPU bounce for
          the cast (device lacks 64-bit compute), then copy back to Neuron.
        - Other dtype changes (<=32-bit targets): use a tiny JAX kernel to cast
          and write, letting the kernel handle the conversion on device.
        """
        try:
            from .cast_policy import write_neuron_to_neuron

            # In case of D2D without CPU bounces, copies are not blocking.
            write_neuron_to_neuron(dst, src, non_blocking=True)
            return ExecutionResult(success=True, output=dst)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
