"""MLIR implementation of copy operations."""

import torch

from torch_neuronx.python_ops.base import ExecutionResult

from ...auto_registration import neuron_op
from ..op_impl import TorchMlirOpImpl


def _copy_fn(src, out_dtype=None, out=None):
    """Copy function that handles dtype conversion.

    Broadcasting is handled by the caller (CopyMLIRImpl.execute) which expands
    src to dst shape before invoking the kernel.  out_dtype is passed as a
    static kwarg so the compiled graph never references out.shape (which would
    produce an unsupported tensor<Nxindex> constant in StableHLO).
    """
    if out_dtype is not None:
        return src.to(out_dtype) + 0
    return src + 0


@neuron_op("aten::copy_", disable_dtype_autocast=True, priority=90)
class CopyMLIRImpl(TorchMlirOpImpl):
    """MLIR-based implementation for neuron-to-neuron copy operations."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::copy_",
            torch_fn=_copy_fn,
            output_params=None,
            static_argnames=("out_dtype",),
        )
        # Override the kernel's infer_output_dtypes with copy_-specific logic:
        # output dtype is always the dst (args[0]) dtype.
        self.kernel.infer_output_dtypes = self._infer_copy_output_dtypes

    def _infer_copy_output_dtypes(self, *args, **kwargs):
        """For copy_, output dtype is always the destination tensor's dtype."""
        dst = args[0] if args else None
        if isinstance(dst, torch.Tensor):
            return [dst.dtype]
        return []

    def can_handle(self, *args, **kwargs) -> bool:
        """Only handle neuron-to-neuron tensor copies.

        Args:
            args[0]: destination tensor
            args[1]: source tensor
        """
        dest = args[0]
        src = args[1]

        # MLIR only handles tensor-to-tensor copies, not scalars
        if not isinstance(src, torch.Tensor):
            return False

        # Only handle neuron-to-neuron copies, excluding dtypes that need special handling:
        # - float64: Not natively supported on device, handled by NRT-based implementation
        # - int64/uint64: Not natively supported on device, handled by NRT-based implementation
        # - bool: Requires special handling, handled by NRT-based implementation
        unsupported_dtypes = (torch.float64, torch.int64, torch.uint64, torch.bool)
        if not (
            dest.device.type == "neuron"
            and src.device.type == "neuron"
            and dest.dtype not in unsupported_dtypes
            and src.dtype not in unsupported_dtypes
        ):
            return False

        from torch_neuronx.neuron import current_stream, default_stream
        from torch_neuronx.utils import is_host_cc_enabled

        return not (is_host_cc_enabled() and current_stream() != default_stream())

    def execute(self, *args, **kwargs):
        """Execute copy operation using MLIR backend.

        Args:
            args[0]: destination tensor
            args[1]: source tensor
        """
        dest = args[0]
        src = args[1]

        if dest.numel() == 0:
            return ExecutionResult(success=True, output=dest)

        # Handle broadcasting: expand src to dest shape before compilation
        src_prepared = src
        if tuple(src.size()) != tuple(dest.size()):
            src_prepared = src.expand_as(dest).contiguous()

        if not src_prepared.is_contiguous():
            from ...cast_policy import ensure_contiguous_on_device

            src_prepared = ensure_contiguous_on_device(src_prepared)

        if not dest.is_contiguous():
            from ...cast_policy import _write_to_noncontiguous_neuron

            _write_to_noncontiguous_neuron(dest, src_prepared)
            return ExecutionResult(success=True, output=dest)

        # Pass out_dtype as static kwarg (only when dtypes differ)
        mlir_kwargs = dict(kwargs)
        if src_prepared.dtype != dest.dtype:
            mlir_kwargs["out_dtype"] = dest.dtype
        # kwargs may already contain 'out' from dispatch, but we rewrite for safety
        mlir_kwargs["out"] = dest

        return super().execute(src_prepared, **mlir_kwargs)

    def get_lower_func(self):
        """Return a lower function for aten::copy_(dest, src, non_blocking).

        C++ aten args: [dest(0), src(1), non_blocking(2)].
        Graph input: src (aten index 1). Output written to: dest (aten index 0).
        """
        kernel = self.kernel

        def copy_lower(*args, **kwargs):
            dest = args[0]
            src = args[1]

            lower_kwargs = {}
            if src.dtype != dest.dtype:
                lower_kwargs["out_dtype"] = dest.dtype

            try:
                result = kernel.lower(src, **lower_kwargs)
            except Exception:
                # Lowering can fail (e.g. dynamic shapes during torch.export).
                # Return None to signal C++ to fall back to CPU.
                return None
            if result is None:
                return None

            # Hardcode: src is aten flattened index 1, dest is output via schema alias
            from torch_neuronx.python_ops.lowering_result import InputSpec

            result.kept_input_indices = [1]
            result.input_specs = [InputSpec(source_arg_index=1)]
            result.is_inplace = False
            result.input_output_aliases = []

            return result

        return copy_lower
