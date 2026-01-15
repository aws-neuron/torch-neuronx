"""Torch-MLIR compilation utilities."""

import logging

import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass

logger = logging.getLogger(__name__)


class TorchMlirCompiler:
    """Compiles PyTorch functions to StableHLO via torch-mlir."""

    def compile(self, torch_fn, op_name, args, kwargs):
        """Compile PyTorch function to StableHLO MLIR module.

        Args:
            torch_fn: PyTorch function to compile
            op_name: Operation name for function naming
            args: Example input arguments
            kwargs: Example keyword arguments

        Returns:
            MLIR module in StableHLO dialect
        """

        # Wrap function with kwargs
        class OpWrapper(torch.nn.Module):
            def __init__(self, op_fn, op_kwargs):
                super().__init__()
                self.op_fn = op_fn
                self.op_kwargs = op_kwargs

            def forward(self, *inputs):
                return self.op_fn(*inputs, **self.op_kwargs)

        wrapper = OpWrapper(torch_fn, kwargs)

        # Generate StableHLO using torch-mlir fx
        stablehlo_module = fx.export_and_import(
            wrapper,
            *args,
            output_type=OutputType.STABLEHLO,
        )

        # Apply custom passes
        stablehlo_module = dtype_conversion_pass(stablehlo_module)

        return stablehlo_module

    def export_to_hlo(self, mlir_module) -> str:
        """Export StableHLO MLIR module to text format.

        Args:
            mlir_module: MLIR module in StableHLO dialect

        Returns:
            StableHLO MLIR text (compiler supports StableHLO directly)
        """
        return str(mlir_module)
