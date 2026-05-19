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
        # Float-to-int requires explicit truncation to match PyTorch semantics
        if out_dtype in (torch.int32, torch.int16, torch.int8) and src.is_floating_point():
            src = src.trunc()
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
        # Override the kernel's infer_output_dtypes with _to_copy-specific logic:
        # output dtype is the target dtype kwarg, or src dtype if not specified.
        self.kernel.infer_output_dtypes = self._infer_to_copy_output_dtypes

    def _infer_to_copy_output_dtypes(self, *args, **kwargs):
        """For _to_copy, output dtype is kwargs['dtype'] or src dtype."""
        dtype = kwargs.get("dtype")
        if dtype is None and args:
            src = args[0]
            if isinstance(src, torch.Tensor):
                dtype = src.dtype
        return [dtype] if dtype is not None else []

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
        # - int64/uint64: Not natively supported on device, handled by NRT-based implementation
        unsupported_dtypes = (torch.float64, torch.bool, torch.int64, torch.uint64)
        if not (
            src.device.type == "neuron"
            and target_device.type == "neuron"
            and src.dtype not in unsupported_dtypes
            and target_dtype not in unsupported_dtypes
        ):
            return False

        from torch_neuronx.neuron import current_stream, default_stream
        from torch_neuronx.utils import is_host_cc_enabled

        return not (is_host_cc_enabled() and current_stream() != default_stream())

    def execute(self, *args, **kwargs):
        """Execute to_copy operation using MLIR backend.

        Args:
            args[0]: source tensor
            kwargs: dtype, layout, device, pin_memory, non_blocking, memory_format
        """
        src = args[0]
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")

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

        # Preserve source strides in output (matches PyTorch .to(dtype) semantics)
        # PyTorch preserves strides only for non-overlapping-and-dense tensors (e.g. transpose);
        # for tensors with gaps (e.g. slices), output is contiguous.
        if not src.is_contiguous() and torch.ops.aten.is_non_overlapping_and_dense(src):
            out = torch.empty_strided(
                src.shape, src.stride(), dtype=target_dtype, device=target_device
            )
        else:
            out = io_tensor.empty(src.shape, dtype=target_dtype, device=target_device)

        mlir_kwargs = {}
        if src_prepared.dtype != target_dtype:
            mlir_kwargs["out_dtype"] = target_dtype
        mlir_kwargs["out"] = out

        return super().execute(src_prepared, **mlir_kwargs)

    def get_lower_func(self):
        """Return a lower function that translates aten::_to_copy args to kernel kwargs.

        C++ Dispatch calls lower_func(src, dtype=..., layout=..., device=..., ...) with
        raw aten kwargs. The kernel expects lower(src, out_dtype=..., out=...).
        """
        kernel = self.kernel

        def to_copy_lower(*args, **kwargs):
            src = args[0]
            target_dtype = kwargs.get("dtype", src.dtype)

            # Translate to kernel kwargs
            translated_kwargs = {}
            if target_dtype != src.dtype:
                translated_kwargs["out_dtype"] = target_dtype

            return kernel.lower(src, **translated_kwargs)

        return to_copy_lower
