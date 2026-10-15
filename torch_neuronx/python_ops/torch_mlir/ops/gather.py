"""MLIR implementation of gather operations."""

import torch

from ...auto_registration import neuron_op
from ..op_impl import TorchMlirOpImpl


def _gather_mlir(input, dim, index, sparse_grad=False, out=None):
    if index.numel() == 0:
        return torch.empty(index.shape, dtype=input.dtype, device=input.device)
    return torch.ops.aten.gather(input, dim, index, sparse_grad=sparse_grad)


@neuron_op("aten::gather", priority=40)
class GatherMLIRImpl(TorchMlirOpImpl):
    """Implement MLIR gather operation variants.

    Args:
        op_variant (str, optional): Use ``"gather.out"`` for the output variant;
            any other value selects ``"gather"``. Defaults to ``"gather"``.
    """

    def __init__(self, op_variant="gather"):
        if op_variant == "gather.out":
            aten_op_name = "aten::gather.out"
            output_params = ("out",)
        else:
            aten_op_name = "aten::gather"
            output_params = None

        super().__init__(
            aten_op_name=aten_op_name,
            torch_fn=_gather_mlir,
            output_params=output_params,
            static_argnums=(1,),
            static_argnames=("sparse_grad",),
        )
        self.op_variant = op_variant

    def can_handle(self, input, dim, index, sparse_grad=False, out=None) -> bool:
        """Check whether MLIR can handle the gather arguments.

        Args:
            input (torch.Tensor): Tensor checked for Neuron device placement.
            dim (int): Gather dimension, accepted but not inspected.
            index (torch.Tensor): Index tensor checked for Neuron device placement.
            sparse_grad (bool, optional): Sparse-gradient request, accepted but not inspected.
                Defaults to ``False``.
            out (torch.Tensor, optional): Output tensor, accepted but not inspected.
                Defaults to ``None``.

        Returns:
            bool: ``True`` when MLIR ATen ops are enabled and inputs use a Neuron device.
        """
        from torch_neuronx.utils import use_mlir_aten_ops

        return (
            use_mlir_aten_ops() and input.device.type == "neuron" and index.device.type == "neuron"
        )


class GatherOutMLIRImpl(GatherMLIRImpl):
    """Implement the output-tensor MLIR gather operation."""

    def __init__(self):
        super().__init__("gather.out")
