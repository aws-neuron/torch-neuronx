"""Optimized aten::_foreach_copy_ for Neuron.

When all src and dst tensors are adjacent 1D views of contiguous flat buffers
with matching dtype, replaces N individual NRT copy calls with a single one.

This targets FSDPv2's all-gather copy-in path where _foreach_copy_ copies
splits of one flat buffer into splits of another.
"""

from __future__ import annotations

import logging

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


def _is_contiguous_range(tensors: list[torch.Tensor]) -> bool:
    """Check if tensors are adjacent, non-overlapping 1D views of one storage.

    Returns False if any tensor is zero-numel, non-contiguous, or not 1D.
    """
    if not tensors:
        return False

    first = tensors[0]
    if not first.is_contiguous() or first.dim() != 1 or first.numel() == 0:
        return False

    storage_ptr = first.untyped_storage().data_ptr()
    if storage_ptr == 0:
        return False

    elem_size = first.element_size()
    expected_offset = first.storage_offset()

    for t in tensors:
        if (
            not t.is_contiguous()
            or t.dim() != 1
            or t.untyped_storage().data_ptr() != storage_ptr
            or t.storage_offset() != expected_offset
            or t.element_size() != elem_size
        ):
            return False
        expected_offset += t.numel()

    return True


@neuron_op(
    "aten::_foreach_copy_",
    disable_dtype_autocast=True,
    disable_empty_shortcircuit=True,
    priority=100,
)
class ForeachCopyNeuronImpl(OperationImplementation):
    """Fused _foreach_copy_ that batches contiguous views into one NRT copy."""

    def can_handle(self, self_tensors, src_tensors, non_blocking=False, **kwargs) -> bool:
        if not self_tensors or not src_tensors:
            return False
        if len(self_tensors) != len(src_tensors):
            return False
        # Accept any case where at least one tensor is on Neuron
        for d, s in zip(self_tensors, src_tensors, strict=False):
            if d.device.type == "neuron" or s.device.type == "neuron":
                return True
        return False

    def _execute_impl(self, self_tensors, src_tensors, non_blocking=False, **kwargs):
        # Safety check: if any tensor is not on neuron, fall back to per-element copy_
        # This handles cases like host<->device that can_handle should reject
        for d, s in zip(self_tensors, src_tensors, strict=False):
            if d.device.type != "neuron" or s.device.type != "neuron":
                for dst, src in zip(self_tensors, src_tensors, strict=False):
                    dst.copy_(src)
                return ExecutionResult(success=True, output=None)
        try:
            same_dtype = all(
                d.dtype == s.dtype for d, s in zip(self_tensors, src_tensors, strict=False)
            )
            if same_dtype:
                if _is_contiguous_range(self_tensors) and _is_contiguous_range(src_tensors):
                    logger.debug(
                        "foreach_copy_: fused %d tensors into single NRT copy",
                        len(self_tensors),
                    )
                    return self._fused_copy(self_tensors, src_tensors)
                logger.debug("foreach_copy_: individual NRT copy for %d tensors", len(self_tensors))
                return self._individual_copy(self_tensors, src_tensors)

            # Cross-dtype path
            if _is_contiguous_range(self_tensors):
                logger.debug(
                    "foreach_copy_: cross-dtype fused cast %d tensors (%s -> %s)",
                    len(self_tensors),
                    src_tensors[0].dtype,
                    self_tensors[0].dtype,
                )
                return self._fused_cross_dtype_copy(self_tensors, src_tensors)

            logger.debug(
                "foreach_copy_: cross-dtype individual %d tensors (%s -> %s)",
                len(self_tensors),
                src_tensors[0].dtype,
                self_tensors[0].dtype,
            )
            return self._individual_cross_dtype_copy(self_tensors, src_tensors)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=f"foreach_copy_ failed: {e}")

    def _fused_copy(self, dst_list, src_list):
        """Single NRT copy covering the entire contiguous range."""
        import torch_neuronx._C as _C

        total_numel = sum(t.numel() for t in src_list)
        src_view = torch.as_strided(
            src_list[0], (total_numel,), (1,), storage_offset=src_list[0].storage_offset()
        )
        dst_view = torch.as_strided(
            dst_list[0], (total_numel,), (1,), storage_offset=dst_list[0].storage_offset()
        )
        _C._nrt_copy_neuron_to_neuron_tensor(src_view, dst_view)
        return ExecutionResult(success=True, output=None)

    def _individual_copy(self, dst_list, src_list):
        """Fallback: copy each tensor individually."""
        import torch_neuronx._C as _C

        for dst, src in zip(dst_list, src_list, strict=False):
            if dst.numel() > 0:
                _C._nrt_copy_neuron_to_neuron_tensor(src, dst)
        return ExecutionResult(success=True, output=None)

    def _fused_cross_dtype_copy(self, dst_list, src_list):
        """Cross-dtype with contiguous dst: cat scattered srcs, single device cast."""
        from .device_cast import device_cast_neuron_to_neuron

        # dst is contiguous range — build a single flat view
        total_numel = sum(t.numel() for t in dst_list)
        dst_flat = torch.as_strided(
            dst_list[0], (total_numel,), (1,), storage_offset=dst_list[0].storage_offset()
        )
        # cat scattered srcs into one contiguous tensor
        src_flat = torch.cat([s for s in src_list if s.numel() > 0])
        device_cast_neuron_to_neuron(src_flat, dst_flat)
        return ExecutionResult(success=True, output=None)

    def _individual_cross_dtype_copy(self, dst_list, src_list):
        """Cross-dtype with scattered dst: per-element device cast."""
        from .device_cast import device_cast_neuron_to_neuron

        for dst, src in zip(dst_list, src_list, strict=False):
            if dst.numel() > 0:
                device_cast_neuron_to_neuron(src, dst)
        return ExecutionResult(success=True, output=None)
