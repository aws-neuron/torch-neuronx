"""Handler for output tensor management."""

import logging

import torch

import torch_neuronx._C as _C
from torch_neuronx.python_ops.cast_policy import (
    copy_neuron_to_cpu,
    write_neuron_to_neuron,
)

logger = logging.getLogger(__name__)


class BaseOutputHandler:
    """Manage output tensor writes for operation results."""

    def handle_output_parameter(
        self, result: torch.Tensor | tuple, out_param: torch.Tensor | None
    ) -> torch.Tensor | tuple:
        """Write operation results into an optional output parameter.

        Args:
            result (torch.Tensor or tuple): Computed result tensor or tensors.
            out_param (torch.Tensor or tuple or None): Destination tensor or tensors.

        Returns:
            torch.Tensor or tuple: Result structure with destination tensors substituted,
            or the original result when ``out_param`` is ``None``.

        Raises:
            RuntimeError: If a native copy violates device, contiguity, byte-count, or
                storage-bound requirements.
            TypeError: If an up-to-32-bit cast targets a noncontiguous Neuron
                destination.

        Note:
            Supplied destination tensors are mutated in place and resized to the
            corresponding result shapes when needed. Neuron copies use the current
            stream and may complete asynchronously under the native lifetime policy.
        """
        if out_param is None:
            return result

        if isinstance(result, tuple):
            # Multi-output case
            if isinstance(out_param, tuple):
                # Multiple out tensors
                for out, res in zip(out_param, result, strict=False):
                    self._copy_to_output(out, res)
                return out_param
            else:
                # Single out for multi-result
                self._copy_to_output(out_param, result[0])
                return (out_param, *result[1:])
        else:
            # Single output case
            self._copy_to_output(out_param, result)
            return out_param

    def extract_output_params(
        self, kwargs: dict, output_param_names: tuple
    ) -> torch.Tensor | tuple | None:
        """Extract output tensors from keyword arguments.

        Args:
            kwargs (dict): Operation keyword arguments.
            output_param_names (tuple): Names of output parameters to collect.

        Returns:
            torch.Tensor or tuple or None: Present named output tensors, the universal
            ``out`` value as a fallback, or ``None`` if no output is present.
        """
        if not output_param_names:
            # Check for universal 'out' parameter
            return kwargs.get("out")

        # Collect named output tensors
        output_tensors = []
        for param in output_param_names:
            if param in kwargs:
                output_tensors.append(kwargs[param])

        if not output_tensors:
            return kwargs.get("out")  # Fall back to 'out'
        elif len(output_tensors) == 1:
            return output_tensors[0]
        else:
            return tuple(output_tensors)

    def _copy_to_output(
        self, out: torch.Tensor, result: torch.Tensor, non_blocking: bool = False
    ) -> None:
        """Copy result to output tensor with resizing if needed.

        Args:
            out: Output tensor
            result: Result tensor to copy
        """
        if out.shape != result.shape:
            self._resize_output(out, result.shape)

        if isinstance(result, torch.Tensor) and isinstance(out, torch.Tensor):
            src_dev = result.device.type
            dst_dev = out.device.type

            # Neuron -> Neuron: use device-side policy (handles ≤32-bit cast and 64-bit bounce)
            if src_dev == "neuron" and dst_dev == "neuron":
                write_neuron_to_neuron(out, result)
                return

            # CPU -> Neuron: perform CPU-side cast to match out.dtype, then raw copy
            if src_dev == "cpu" and dst_dev == "neuron":
                cpu_src = result
                if cpu_src.dtype != out.dtype:
                    cpu_src = cpu_src.to(out.dtype)
                if not cpu_src.is_contiguous():
                    cpu_src = cpu_src.contiguous()
                _C._nrt_copy_cpu_to_neuron_tensor(cpu_src, out, non_blocking)
                return

            # Neuron -> CPU: raw transfer; if dtype differs, cast on CPU after
            if src_dev == "neuron" and dst_dev == "cpu":
                if result.dtype == out.dtype:
                    _C._nrt_copy_neuron_to_cpu_tensor(result, out, non_blocking=non_blocking)
                else:
                    # Use temporary CPU buffer then cast into out
                    tmp_cpu = copy_neuron_to_cpu(
                        result, target_dtype=out.dtype, non_blocking=non_blocking
                    )
                    out.copy_(tmp_cpu)
                return

        # Fallback to standard copy if inputs non-tensors
        out.copy_(result)

    def _resize_output(self, out: torch.Tensor, expected_shape: tuple) -> None:
        """Resize output tensor if needed, with PyTorch-compatible warning.

        Args:
            out: Output tensor to resize
            expected_shape: Expected shape
        """
        out.resize_(expected_shape)
