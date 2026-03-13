"""MLIR implementation of gather operations."""

import torch

from ..op_impl import TorchMlirOpImpl


def _gather_mlir(input, dim, index, sparse_grad=False, out=None):
    if index.numel() == 0:
        return torch.empty(index.shape, dtype=input.dtype, device=input.device)
    return torch.ops.aten.gather(input, dim, index, sparse_grad=sparse_grad)


class GatherMLIRImpl(TorchMlirOpImpl):
    """Register gather and gather.out"""

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
        from torch_neuronx.utils import use_mlir_aten_ops

        return (
            use_mlir_aten_ops() and input.device.type == "neuron" and index.device.type == "neuron"
        )


class GatherOutMLIRImpl(GatherMLIRImpl):
    def __init__(self):
        super().__init__("gather.out")
