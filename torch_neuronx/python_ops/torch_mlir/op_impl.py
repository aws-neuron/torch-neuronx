"""Torch-MLIR operation implementation."""

import logging

import torch

from torch_neuronx.python_ops.base import CompilableOpImpl, ExecutionResult
from torch_neuronx.python_ops.dtype_autocast import autocast_neuron

from ..jax.handlers import OutputHandler
from .kernel import TorchMlirKernel

logger = logging.getLogger(__name__)


class TorchMlirOpImpl(CompilableOpImpl):
    """Torch-MLIR based operation implementation.

    Executes PyTorch operations by compiling to StableHLO via torch-mlir,
    then to NEFF for Neuron execution.
    """

    def __init__(
        self,
        aten_op_name: str,
        torch_fn: callable,
        output_params: tuple[str, ...] | None = None,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
    ):
        super().__init__()

        self.aten_op_name = aten_op_name
        self.torch_fn = torch_fn
        self.output_params = output_params
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()

        # Initialize handlers
        self.output_handler = OutputHandler()

        # Create kernel
        self.kernel = TorchMlirKernel(
            torch_fn=torch_fn,
            op_name=aten_op_name,
            output_params=output_params,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ExecutionResult with output tensor(s)
        """
        # Use dtype autocast context
        with autocast_neuron() as context:
            # Process arguments
            context.original_inputs = args
            args, kwargs, _ = context.process_args(args, kwargs, output_params=self.output_params)

            # Check for empty tensors
            if self._should_handle_empty(args):
                result = self._handle_empty_tensors(args, kwargs, context)
                if result is not None:
                    return ExecutionResult(success=True, output=result)

            # Execute kernel
            result = self.kernel(*args, context=context, **kwargs)

            return ExecutionResult(success=True, output=result)

    def _execute_impl(self):
        pass

    def _should_handle_empty(self, args) -> bool:
        """Check if any input tensors are empty."""
        return any(isinstance(arg, torch.Tensor) and arg.numel() == 0 for arg in args)

    def _get_device(self, args: tuple) -> torch.device:
        """Get device from arguments.

        Args:
            args: Input arguments

        Returns:
            Device from first tensor, or CPU
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.device
        return torch.device("cpu")

    def _handle_empty_tensors(self, args, kwargs, context):
        """Handle empty tensor cases."""
        output_specs, _, _ = self.kernel._get_cached_output_specs(args, kwargs, context=context)
        device = self._get_device(args)

        outputs = []
        for output_spec in output_specs:
            outputs.append(torch.zeros(output_spec[0], dtype=output_spec[1], device=device))

        return tuple(outputs)
