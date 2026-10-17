"""Empty-HLO contiguous handler for the C++ dispatch path.

Registered at the highest priority so the C++ dispatcher picks it for ALL
non-contiguous patterns.  The kernel produces a trivial clone HLO (empty
from a layout-transformation perspective) — the C++ prologue system
(ProcessInputTransformations) handles all non-contiguous layouts
(broadcast, transpose, slice, compound) by prepending transformations
to the kernel's NEFF before execution.

The broadcast, transpose and slice handlers remain reachable through
_contiguous_internal_impl in contiguous.py, which cast_policy.py calls with
contiguous_format. C++ MakeChannelsLast owns the channels_last layouts.
"""

import torch

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

from .auto_registration import neuron_op
from .base import ExecutionResult, OperationImplementation


@neuron_op("aten::contiguous", disable_dtype_autocast=True, priority=200)
class ContiguousEmptyHloImpl(OperationImplementation):
    """Provide an empty-HLO contiguous kernel for the C++ dispatch path.

    Note:
        The kernel clones data without layout transformation in HLO because the
        C++ prologue prepends broadcast, transpose, slice, or compound transforms.
        Priority 200 selects this handler over the specialized handlers (priorities 100,
        98, 95, and 90) when the C++ dispatcher resolves ``aten::contiguous``.
    """

    _kernel: TorchMlirKernel | None = None

    def can_handle(self, self_tensor, memory_format=None) -> bool:
        """Check whether the C++ prologue can make a tensor contiguous.

        Args:
            self_tensor (torch.Tensor): Tensor to inspect.
            memory_format (torch.memory_format, optional): Requested memory format. Defaults to
                ``None``.

        Returns:
            bool: ``True`` for a non-contiguous, noncomplex Neuron device tensor
            with a supported memory format.

        Raises:
            RuntimeError: If a nonscalar tensor is not on the Neuron device.
        """
        if not super().can_handle(self_tensor, memory_format=memory_format):
            return False
        if self_tensor.is_complex():
            return False
        # Channels-last handled by NRT strided copy in C++ DispatchContiguousImpl.
        if memory_format is not None and memory_format not in (
            torch.contiguous_format,
            torch.preserve_format,
        ):
            return False
        return not self_tensor.is_contiguous()

    def __init__(self):
        super().__init__()
        if ContiguousEmptyHloImpl._kernel is None:

            def empty_hlo_clone(x):
                # After prologue makes x contiguous, clone produces new storage.
                """Clone a tensor after the C++ prologue makes it contiguous.

                Args:
                    x (torch.Tensor): Contiguous tensor to clone.

                Returns:
                    torch.Tensor: Cloned tensor with new storage.
                """
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
        """Return the lowering callback for the empty-HLO kernel.

        Returns:
            Callable: Callback that lowers the first input tensor.
        """
        kernel = ContiguousEmptyHloImpl._kernel

        def contiguous_empty_hlo_lower(*args, **kwargs):
            """Lower the first positional argument with the cached kernel.

            Args:
                *args (Any): Positional arguments whose first item is the input tensor.
                **kwargs (Any): Unused keyword arguments supplied by the dispatcher.

            Returns:
                LoweringResult: Lowered kernel representation.
            """
            return kernel.lower(args[0])

        return contiguous_empty_hlo_lower

    def get_infer_func(self):
        """Return the output-dtype inference callback for the empty-HLO kernel.

        Returns:
            Callable: Callback that infers dtypes from the first input tensor.
        """
        kernel = ContiguousEmptyHloImpl._kernel

        def contiguous_empty_hlo_infer(*args, **kwargs):
            """Infer output dtypes from the first positional argument.

            Args:
                *args (Any): Positional arguments whose first item is the input tensor.
                **kwargs (Any): Unused keyword arguments supplied by the dispatcher.

            Returns:
                list[torch.dtype | None]: Inferred output dtype structure.
            """
            return kernel.infer_output_dtypes(args[0])

        return contiguous_empty_hlo_infer
