"""MLIR implementation of split_with_sizes_copy for FSDP optimization."""

import torch

from torch_neuronx.python_ops.base import ExecutionResult

from ...auto_registration import neuron_op
from ..op_impl import TorchMlirOpImpl


def _split_copy_fn(input_tensor, split_sizes=None, dim=0, out=None):
    """Split function for MLIR tracing.

    split_sizes and dim are passed as static kwargs - they are concrete
    Python values during tracing, not symbolic tensors.

    Uses narrow+clone pattern which decomposes to stablehlo.slice ops.
    This is more efficient than torch.split which produces gather ops,
    since slice uses contiguous DMA copy rather than indexed random access.
    """
    results = []
    offset = 0
    for size in split_sizes:
        results.append(
            torch.narrow(input_tensor, dim, offset, size).clone(
                memory_format=torch.contiguous_format
            )
        )
        offset += size
    return tuple(results)


@neuron_op("aten::split_with_sizes_copy.out", disable_dtype_autocast=True, priority=90)
class SplitWithSizesCopyMLIR(TorchMlirOpImpl):
    """MLIR-based split_with_sizes_copy that compiles to a single kernel."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::split_with_sizes_copy.out",
            torch_fn=_split_copy_fn,
            output_params=("out",),
            # split_sizes and dim are static kwargs - concrete values during tracing
            static_argnames=("split_sizes", "dim"),
        )

    def can_handle(self, *args, **kwargs) -> bool:
        """Handle when input is on Neuron device."""
        input_tensor = args[0]
        if not isinstance(input_tensor, torch.Tensor):
            return False
        if input_tensor.device.type != "neuron":
            return False
        # Check output tensors
        out = kwargs.get("out")
        if out is None:
            return False
        return all(t.device.type == "neuron" for t in out)

    def execute(self, *args, **kwargs):
        """Execute split_with_sizes_copy using MLIR backend.

        Args:
            args[0]: input tensor
            args[1]: split_sizes (list/tuple of ints)
            args[2]: dim (default 0)
            kwargs['out']: output tensors
        """
        input_tensor = args[0]
        split_sizes = tuple(args[1]) if len(args) > 1 else ()
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        out = kwargs.get("out")

        n_outputs = len(split_sizes)

        # Handle edge cases
        if n_outputs == 0 or input_tensor.numel() == 0:
            return ExecutionResult(success=True, output=None)

        if out is None or len(out) != n_outputs:
            return ExecutionResult(
                success=False,
                error_msg=f"Expected {n_outputs} out tensors, got {len(out) if out else 0}",
            )

        # Ensure input is contiguous for efficient splitting
        src_prepared = input_tensor
        if not input_tensor.is_contiguous():
            from ...cast_policy import ensure_contiguous_on_device

            src_prepared = ensure_contiguous_on_device(input_tensor)

        # Pass split_sizes and dim as static kwargs (not traced)
        mlir_kwargs = {
            "split_sizes": split_sizes,
            "dim": dim,
            "out": tuple(out),
        }

        # Execute via parent - only src_prepared is a traced tensor
        result = super().execute(src_prepared, **mlir_kwargs)

        # .out variant must return None (data written to out tensors)
        if result.success:
            return ExecutionResult(success=True, output=None)
        return result
