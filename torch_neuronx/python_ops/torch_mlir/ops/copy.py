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
        # Float-to-int requires explicit truncation to match PyTorch semantics
        if out_dtype in (torch.int32, torch.int16, torch.int8) and src.is_floating_point():
            src = src.trunc()
        result = src.to(out_dtype)
    else:
        result = src
    # Force a copy (prevent no-op graph). Use bitwise OR for bool to avoid
    # int64 promotion from `bool + 0`.
    if (out_dtype or src.dtype) == torch.bool:
        return result | False
    return result + 0


@neuron_op("aten::copy_", disable_dtype_autocast=True, priority=90)
class CopyMLIRImpl(TorchMlirOpImpl):
    """Implement Neuron-to-Neuron copies with Torch-MLIR."""

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
        """Check whether MLIR supports the requested tensor copy.

        Args:
            *args (Any): Destination and source copy arguments.
            **kwargs (Any): Additional dispatch arguments.

        Returns:
            bool: ``True`` for supported Neuron-to-Neuron copies.

        Note:
            Rejects non-tensor sources, non-Neuron tensors, float64 or uint64 tensors, copies
                between int64 and floating dtypes, and nondefault streams when host collective
                communication is enabled.
        """
        dest = args[0]
        src = args[1]

        # MLIR only handles tensor-to-tensor copies, not scalars
        if not isinstance(src, torch.Tensor):
            return False

        # Only handle neuron-to-neuron copies, excluding dtypes that need special handling:
        # - float64: Not natively supported on device, handled by NRT-based implementation
        # - uint64: Not natively supported on device, handled by NRT-based implementation
        unsupported_dtypes = (torch.float64, torch.uint64)
        # An int64 to float pair needs all 64 bits. The device convert keeps the
        # low word only, so that pair goes to the NRT path. The C++ mirror of this
        # rule is IsCopyMlirPairSupported() in NeuronDispatcher.cpp.
        int64_float_pair = (src.dtype == torch.int64 and dest.dtype.is_floating_point) or (
            dest.dtype == torch.int64 and src.dtype.is_floating_point
        )
        if int64_float_pair:
            return False
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
        """Copy source data into a destination tensor.

        Args:
            *args (Any): Destination tensor followed by the source tensor.
            **kwargs (Any): Additional dispatch arguments.

        Returns:
            ExecutionResult: Mutated destination tensor or failure information.

        Note:
            Expands broadcast sources, makes sources contiguous, converts sources to the
                destination dtype, and handles noncontiguous destinations separately.
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

            if src_prepared.dtype != dest.dtype:
                src_prepared = src_prepared.to(dest.dtype)
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
        """Return the C++ dispatch lowering callable for ``aten::copy_``.

        Returns:
            Callable: Function that lowers the source and writes through the destination schema
                alias.

        Note:
            The lowering keeps flattened source index 1, treats destination index 0 as output, and
                signals CPU fallback by returning ``None`` on failure.
        """
        kernel = self.kernel

        def copy_lower(*args, **kwargs):
            """Lower a source tensor for the in-place copy schema.

            Args:
                *args (Any): Destination and source copy arguments.
                **kwargs (Any): Additional lowering arguments.

            Returns:
                LoweringResult or None: Lowered source copy, or ``None`` to request CPU fallback.
            """
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
