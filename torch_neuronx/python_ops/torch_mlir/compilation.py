"""Torch-MLIR compilation utilities."""

import logging

import torch

from torch_neuronx.neuron_dynamo_backend.fx.passes.dtype_conversion import dtype_conversion_pass

logger = logging.getLogger(__name__)


class TorchMlirCompiler:
    """Compile PyTorch functions to StableHLO through torch-mlir."""

    def compile(self, torch_fn, op_name, args, kwargs):
        """Compile a PyTorch function to a StableHLO MLIR module.

        Args:
            torch_fn (Callable): PyTorch function to compile.
            op_name (str): Operation name accepted but not used.
            args (tuple): Example positional inputs for tracing.
            kwargs (dict): Example keyword inputs for tracing.

        Returns:
            Any: StableHLO MLIR module after dtype conversion.
        """
        from torch_mlir import fx
        from torch_mlir.compiler_utils import OutputType

        # Wrap function with kwargs
        class OpWrapper(torch.nn.Module):
            """Bind keyword arguments to a PyTorch operation for export.

            Args:
                op_fn (Callable): PyTorch operation to invoke.
                op_kwargs (dict): Keyword arguments bound to the operation.
            """

            def __init__(self, op_fn, op_kwargs):
                super().__init__()
                self.op_fn = op_fn
                self.op_kwargs = op_kwargs

            def forward(self, *inputs):
                """Invoke the wrapped operation with bound keyword arguments.

                Args:
                    *inputs: Positional operation inputs.

                Returns:
                    Any: Result returned by the wrapped operation.
                """
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
        """Serialize a StableHLO MLIR module as text.

        Args:
            mlir_module (Any): StableHLO MLIR module to serialize.

        Returns:
            str: String representation of the StableHLO MLIR module.
        """
        return str(mlir_module)
