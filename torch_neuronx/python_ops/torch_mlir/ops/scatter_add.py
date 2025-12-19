"""MLIR implementation of scatter_add operations."""

import torch

from ..op_impl import TorchMlirOpImpl


class ScatterAddMLIRImpl(TorchMlirOpImpl):
    """Register scatter_add, scatter_add_, and scatter_add.out"""

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
            torch_fn=torch.scatter_add,
            output_params=output_params,
            static_argnums=(1,),
        )
        self.op_variant = op_variant

    def _handle_empty_tensors(self, args, kwargs, context):
        """Handle empty tensor cases for scatter_add"""
        input = args[0]
        if self.op_variant == "scatter_add_":
            return input
        elif self.op_variant == "scatter_add.out":
            out = kwargs.get("out")
            if out is not None:
                out.copy_(input)
                return out
            return input.clone()
        else:
            return input.clone()

    def execute(self, *args, **kwargs):
        if self.op_variant == "scatter_add_":
            result = super().execute(*args, **kwargs)
            if result.success:
                args[0].copy_(result.output)
                from ...base import ExecutionResult

                return ExecutionResult(success=True, output=args[0])
            return result
        return super().execute(*args, **kwargs)


class ScatterAddInplaceMLIRImpl(ScatterAddMLIRImpl):
    def __init__(self):
        super().__init__("scatter_add_")


class ScatterAddOutMLIRImpl(ScatterAddMLIRImpl):
    def __init__(self):
        super().__init__("scatter_add.out")
