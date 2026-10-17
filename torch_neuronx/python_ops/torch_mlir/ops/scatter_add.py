"""MLIR implementation of scatter_add operations."""

import torch

from ...auto_registration import neuron_op
from ..op_impl import TorchMlirOpImpl


def _scatter_add_fn(input, dim, index, src, *, out=None):
    # out is excluded from tracing - handled by output_params in base class
    # ATen accepts a 0-dim tensor against a 1-dim counterpart, but the StableHLO
    # conversion requires input, index and src to share a rank ("`index`, `input`
    # and `src` param should have the same rank"). Lift every 0-dim operand to
    # rank 1, then restore the ATen result rank, which always follows input.
    # A mismatch only ever spans rank 0 and 1, so dim is 0 in that case.
    ranks = (input.dim(), index.dim(), src.dim())
    if min(ranks) != max(ranks):
        aligned_input = input.reshape(1) if input.dim() == 0 else input
        aligned_index = index.reshape(1) if index.dim() == 0 else index
        aligned_src = src.reshape(1) if src.dim() == 0 else src
        if aligned_src.numel() < aligned_index.numel():
            aligned_src = aligned_src.expand(aligned_index.shape)
        result = torch.scatter_add(aligned_input, 0, aligned_index, aligned_src)
        return result.reshape(()) if input.dim() == 0 else result
    return torch.scatter_add(input, dim, index, src)


@neuron_op("aten::scatter_add", priority=40)
class ScatterAddMLIRImpl(TorchMlirOpImpl):
    """Implement MLIR scatter-add operation variants.

    Args:
        op_variant (str, optional): ATen scatter-add variant. Defaults to ``"scatter_add"``.
    """

    def __init__(self, op_variant="scatter_add"):
        if op_variant == "scatter_add_":
            aten_op_name = "aten::scatter_add_"
            output_params = None
        elif op_variant == "scatter_add.out":
            aten_op_name = "aten::scatter_add.out"
            output_params = ("out",)
        else:
            aten_op_name = "aten::scatter_add"
            output_params = None

        super().__init__(
            aten_op_name=aten_op_name,
            torch_fn=_scatter_add_fn,
            output_params=output_params,
            static_argnums=(1,),
        )
        self.op_variant = op_variant

    def can_handle(self, input, dim, index, src, out=None) -> bool:
        """Check whether MLIR can handle the scatter-add arguments.

        Args:
            input (torch.Tensor): Destination tensor to scatter into.
            dim (int): Dimension along which to scatter.
            index (torch.Tensor): Indices for the scatter operation.
            src (torch.Tensor): Values to add.
            out (torch.Tensor, optional): Unused output tensor. Defaults to ``None``.

        Returns:
            bool: ``True`` when MLIR ATen ops are enabled and ``input``, ``index``,
                and ``src`` use a Neuron device.
        """
        from torch_neuronx.utils import use_mlir_aten_ops

        return (
            use_mlir_aten_ops()
            and input.device.type == "neuron"
            and index.device.type == "neuron"
            and src.device.type == "neuron"
        )

    def execute(self, *args, **kwargs):
        """Execute the configured scatter-add variant.

        Args:
            *args (Any): Positional scatter-add arguments.
            **kwargs (Any): Keyword scatter-add arguments.

        Returns:
            ExecutionResult: The execution result, with in-place variants copied into the input.

        Note:
            The in-place variant mutates the first positional tensor on successful execution.
        """
        if self.op_variant == "scatter_add_":
            result = super().execute(*args, **kwargs)
            if result.success:
                args[0].copy_(result.output)
                from ...base import ExecutionResult

                return ExecutionResult(success=True, output=args[0])
            return result
        return super().execute(*args, **kwargs)


@neuron_op("aten::scatter_add_", priority=40)
class ScatterAddInplaceMLIRImpl(ScatterAddMLIRImpl):
    """Implement the in-place MLIR scatter-add operation."""

    def __init__(self):
        super().__init__("scatter_add_")


class ScatterAddOutMLIRImpl(ScatterAddMLIRImpl):
    """Implement the output-tensor MLIR scatter-add operation."""

    def __init__(self):
        super().__init__("scatter_add.out")
