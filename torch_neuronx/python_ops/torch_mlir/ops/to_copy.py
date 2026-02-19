"""MLIR implementation of to_copy operations."""

import torch

from torch_neuronx.python_ops import io_tensor

from ...auto_registration import neuron_op
from ...base import ExecutionResult
from ..op_impl import TorchMlirOpImpl


def _to_copy_fn(src, out_dtype=None, out=None):
    """To_copy function that handles dtype conversion.

    out_dtype is passed as a static kwarg so the compiled graph never
    references out.shape (which would produce an unsupported
    tensor<Nxindex> constant in StableHLO).
    """
    if out_dtype is not None:
        return src.to(out_dtype)
    return src.clone()


@neuron_op("aten::_to_copy", disable_dtype_autocast=True, priority=90)
class ToCopyMLIRImpl(TorchMlirOpImpl):
    """MLIR-based implementation for neuron-to-neuron to_copy operations."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::_to_copy",
            torch_fn=_to_copy_fn,
            output_params=None,
            static_argnames=("out_dtype",),
        )

    def can_handle(self, *args, **kwargs) -> bool:
        """Only handle neuron-to-neuron tensor copies.

        Args:
            args[0]: source tensor
            kwargs: dtype, layout, device, pin_memory, non_blocking, memory_format
        """
        src = args[0]

        # MLIR only handles tensor operations
        if not isinstance(src, torch.Tensor):
            return False

        dtype = kwargs.get("dtype")
        device = kwargs.get("device")

        target_dtype = dtype if dtype is not None else src.dtype
        target_device = device if device is not None else src.device

        if isinstance(target_device, str):
            target_device = torch.device(target_device)

        # Only handle neuron-to-neuron copies, excluding dtypes that need special handling:
        # - float64: Not natively supported on device, handled by NRT-based implementation
        # - bool: Requires special handling, handled by NRT-based implementation
        # - int32: Requires special handling, handled by NRT-based implementation
        unsupported_dtypes = (torch.float64, torch.bool, torch.int32)
        return (
            src.device.type == "neuron"
            and target_device.type == "neuron"
            and src.dtype not in unsupported_dtypes
            and target_dtype not in unsupported_dtypes
        )

    def execute(self, *args, **kwargs):
        """Execute to_copy operation using MLIR backend.

        Args:
            args[0]: source tensor
            kwargs: dtype, layout, device, pin_memory, non_blocking, memory_format
        """
        src = args[0]
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        non_blocking = kwargs.get("non_blocking", False)

        target_dtype = dtype if dtype is not None else src.dtype
        target_device = device if device is not None else src.device

        if isinstance(target_device, str):
            target_device = torch.device(target_device)

        if src.numel() == 0:
            empty = io_tensor.empty(src.shape, dtype=target_dtype, device=target_device)
            return ExecutionResult(success=True, output=empty)

        # Handle non-contiguous source tensor
        src_prepared = src
        if not src.is_contiguous():
            from ...cast_policy import ensure_contiguous_on_device

            src_prepared = ensure_contiguous_on_device(src)

        out = io_tensor.empty(src.shape, dtype=target_dtype, device=target_device)

        mlir_kwargs = {}
        if src_prepared.dtype != target_dtype:
            mlir_kwargs["out_dtype"] = target_dtype
        mlir_kwargs["out"] = out

        result = super().execute(src_prepared, **mlir_kwargs)

        if not non_blocking:
            import torch_neuronx

            stream = torch_neuronx.current_stream(target_device)
            stream.synchronize()

        return result
