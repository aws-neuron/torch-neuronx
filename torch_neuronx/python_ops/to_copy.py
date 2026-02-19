"""Python implementation of aten::_to_copy for Neuron.

This mirrors the policy of copy_ but avoids recursion by never calling copy_.
It uses direct NRT helpers for CPU<->Neuron transfers for <=32-bit Neuron-to-Neuron
dtype casts, bouncing to CPU only for 64-bit.
"""

from __future__ import annotations

import logging

import torch

from torch_neuronx.python_ops import io_tensor

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


@neuron_op("aten::_to_copy", disable_dtype_autocast=True, priority=50)
class ToCopyNeuronImpl(OperationImplementation):
    def can_handle(
        self,
        self_tensor: torch.Tensor,
        dtype: torch.dtype | None = None,
        layout: torch.layout | None = None,
        device: torch.device | None = None,
        pin_memory: bool | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format | None = None,
        *,
        out=None,
        **kwargs,
    ) -> bool:
        # Resolve target device and dtype
        dst_device = (
            device
            if isinstance(device, torch.device)
            else (torch.device(device) if device is not None else self_tensor.device)
        )

        # Handle all neuron-to-neuron and CPU<->Neuron transfers
        return self_tensor.device.type == "neuron" or dst_device.type == "neuron"

    def _execute_impl(
        self,
        self_tensor: torch.Tensor,
        dtype: torch.dtype | None = None,
        layout: torch.layout | None = None,
        device: torch.device | None = None,
        pin_memory: bool | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format | None = None,
        *,
        out=None,
        **kwargs,
    ) -> ExecutionResult:
        try:
            pass
        except Exception as e:
            return ExecutionResult(success=False, error_msg=f"torch_neuronx C-ext missing: {e}")

        # Resolve target options
        target_device = (
            device
            if isinstance(device, torch.device)
            else (torch.device(device) if device is not None else self_tensor.device)
        )
        target_dtype = dtype if dtype is not None else self_tensor.dtype

        is_to_neuron = target_device.type == "neuron"
        is_from_neuron = self_tensor.device.type == "neuron"

        # Memory format rule for Neuron
        if is_to_neuron or is_from_neuron:
            try:
                from .cast_policy import validate_neuron_memory_format

                validate_neuron_memory_format(memory_format, op_name="aten::_to_copy")
            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))

        # No-op fast path (preserve/None memory format)
        no_layout_change = layout is None or layout == self_tensor.layout
        no_dtype_change = target_dtype == self_tensor.dtype
        no_device_change = target_device == self_tensor.device
        if (
            no_layout_change
            and no_dtype_change
            and no_device_change
            and (memory_format is None or memory_format == torch.preserve_format)
        ):
            return ExecutionResult(success=True, output=self_tensor)

        # CPU -> Neuron
        if self_tensor.device.type == "cpu" and is_to_neuron:
            from .cast_policy import copy_cpu_to_neuron

            result = copy_cpu_to_neuron(
                self_tensor, target_device, target_dtype, non_blocking=non_blocking
            )
            return ExecutionResult(success=True, output=result)

        # Neuron -> CPU
        if is_from_neuron and target_device.type == "cpu":
            from .cast_policy import copy_neuron_to_cpu

            cpu_dst = copy_neuron_to_cpu(
                self_tensor,
                target_dtype=(target_dtype if not no_dtype_change else None),
                non_blocking=non_blocking,
            )
            return ExecutionResult(success=True, output=cpu_dst)

        # Neuron -> Neuron (device may or may not change index)
        if is_from_neuron and is_to_neuron:
            from .cast_policy import write_neuron_to_neuron

            result = io_tensor.empty(self_tensor.shape, dtype=target_dtype, device=target_device)
            logger.debug(
                f"Using NRT API based copy for neuron-to-neuron _to_copy: "
                f"src_shape={tuple(self_tensor.shape)}, "
                f"src_dtype={self_tensor.dtype}, "
                f"src_device={self_tensor.device}, "
                f"dst_shape={tuple(result.shape)}, "
                f"dst_dtype={result.dtype}, "
                f"dst_device={result.device}, "
                f"non_blocking={non_blocking}"
            )
            try:
                # In case of D2D without CPU bounces, copies are not blocking.
                write_neuron_to_neuron(result, self_tensor, non_blocking=True)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))
            return ExecutionResult(success=True, output=result)

        # Fallback: other combinations (e.g., CPU->CPU)
        # Use standard PyTorch semantics safely (CPU path won't hit neuron copy_ implementation)
        result = io_tensor.empty(
            self_tensor.shape, dtype=target_dtype, device=target_device, pin_memory=pin_memory
        )
        # Avoid recursion by not calling copy_ when either side is Neuron (already handled)
        if self_tensor.device.type == "cpu" and target_device.type == "cpu":
            result.copy_(self_tensor, non_blocking=non_blocking)
            return ExecutionResult(success=True, output=result)

        return ExecutionResult(
            success=False,
            error_msg=(
                f"Unsupported _to_copy combination: src={self_tensor.device}, dst={target_device}, "
                f"dtype_src={self_tensor.dtype} dtype_dst={target_dtype}"
            ),
        )

    def _handle_empty_tensor(
        self,
        self_tensor: torch.Tensor,
        dtype: torch.dtype | None = None,
        layout: torch.layout | None = None,
        device: torch.device | None = None,
        pin_memory: bool | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format | None = None,
        *,
        out=None,
        **kwargs,
    ) -> ExecutionResult:
        # For empty tensors, just create an empty tensor with target options
        target_device = (
            device
            if isinstance(device, torch.device)
            else (torch.device(device) if device is not None else self_tensor.device)
        )
        target_dtype = dtype if dtype is not None else self_tensor.dtype
        empty = io_tensor.empty(self_tensor.shape, dtype=target_dtype, device=target_device)
        return ExecutionResult(success=True, output=empty)
