"""Empty-HLO contiguous handler for the C++ dispatch path.

Registered at the highest priority so the C++ dispatcher picks it for ALL
non-contiguous patterns.  The kernel produces a trivial clone HLO (empty
from a layout-transformation perspective) — the C++ prologue system
(ProcessInputTransformations) handles all non-contiguous layouts
(broadcast, transpose, slice, compound) by prepending transformations
to the kernel's NEFF before execution.

The specialized handlers (broadcast, transpose, slice, channels_last) remain
available for the Python fallback path (TORCH_NEURONX_CPP_DISPATCH=0) via
_contiguous_internal_impl in contiguous.py.
"""

import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


@neuron_op("aten::contiguous", disable_dtype_autocast=True, priority=200)
class ContiguousEmptyHloImpl(OperationImplementation):
    """Empty-HLO kernel for contiguous — prologue handles all layout work.

    The kernel is a trivial clone (no layout transformation in the HLO).
    The C++ prologue prepends broadcast/transpose/slice/compound
    transformations so the kernel receives already-contiguous data.

    Priority 200 ensures this is picked over specialized handlers
    (100, 98, 95, 90) when the C++ dispatcher resolves aten::contiguous.
    Only registered when TORCH_NEURONX_CPP_DISPATCH=1.
    """

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False
        if self_tensor.is_complex():
            return False
        # Channels-last handled by NRT strided copy in C++ DispatchContiguousImpl.
        return memory_format is None or memory_format in (
            torch.contiguous_format,
            torch.preserve_format,
        )

    def __init__(self):
        super().__init__()
        if ContiguousEmptyHloImpl._kernel is None:

            def empty_hlo_clone(x):
                # After prologue makes x contiguous, clone produces new storage.
                return x.clone()

            ContiguousEmptyHloImpl._kernel = TorchMlirKernel(
                empty_hlo_clone, op_name="aten::contiguous"
            )

    def _execute_impl(self, self_tensor, memory_format=None) -> ExecutionResult:
        try:
            output = ContiguousEmptyHloImpl._kernel(self_tensor)
            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    def get_lower_func(self):
        kernel = ContiguousEmptyHloImpl._kernel

        def contiguous_empty_hlo_lower(*args, **kwargs):
            original_args = kwargs.pop("original_args", None)
            original_kwargs = kwargs.pop("original_kwargs", None)
            return kernel.lower(
                args[0], original_args=original_args, original_kwargs=original_kwargs
            )

        return contiguous_empty_hlo_lower

    def get_infer_func(self):
        kernel = ContiguousEmptyHloImpl._kernel

        def contiguous_empty_hlo_infer(*args, **kwargs):
            return kernel.infer_output_dtypes(args[0])

        return contiguous_empty_hlo_infer
