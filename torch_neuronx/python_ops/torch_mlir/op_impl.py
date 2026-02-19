"""Torch-MLIR operation implementation."""

import logging
import traceback

import torch

from torch_neuronx.python_ops.base import CompilableOpImpl, ExecutionResult
from torch_neuronx.python_ops.dtype_autocast import autocast_neuron
from torch_neuronx.python_ops.shared import ExecutionContext

from ..handlers.output import BaseOutputHandler
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
        self.output_handler = BaseOutputHandler()

        # Check for preprocessing
        self._uses_preprocessing = getattr(self.torch_fn, "uses_preprocessing", False)
        self._postprocess_fn = None
        if not self._uses_preprocessing:
            self.kernel = TorchMlirKernel(
                torch_fn=torch_fn,
                op_name=aten_op_name,
                output_params=output_params,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
            )
        else:
            self.kernel = None

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ExecutionResult with output tensor(s)
        """
        if self._uses_preprocessing:
            preprocess_result = self.torch_fn(*args, **kwargs)

            # check if postprocess function is provided
            if len(preprocess_result) == 4:
                actual_torch_fn, args, kwargs, self._postprocess_fn = preprocess_result
            else:
                actual_torch_fn, args, kwargs = preprocess_result
                self._postprocess_fn = None

            if self.kernel is None:
                self.kernel = TorchMlirKernel(
                    torch_fn=actual_torch_fn,
                    op_name=self.aten_op_name,
                    output_params=self.output_params,
                    static_argnums=self.static_argnums,
                    static_argnames=self.static_argnames,
                )

        exec_context = ExecutionContext(original_inputs=args, original_kwargs=kwargs.copy())

        # Apply dtype autocast only if not disabled
        if not self.DISABLE_DTYPE_AUTOCAST:
            # Use dtype autocast context
            with autocast_neuron(cast_dtype_params=True) as context:
                # Process arguments, excluding output parameters from casting
                original_kwargs = kwargs
                args, kwargs, _ = context.process_args(
                    args, kwargs, output_params=self.output_params
                )

                # Validate output dtypes
                context.validate_output_dtypes(original_kwargs)

        # Execute kernel
        try:
            result = self.kernel(*args, context=exec_context, **kwargs)

            # Apply postprocessing if available
            if self._postprocess_fn is not None:
                result = self._postprocess_fn(result)

            return ExecutionResult(success=True, output=result)
        except Exception as e:
            # Do not fall back for user/input contract errors; match PyTorch behavior
            error_traceback = traceback.format_exc()
            if isinstance(e, IndexError | ValueError | TypeError):
                raise
            error_message = f"{e!s}\n{error_traceback}"
            logger.debug(f"Execution failed: {error_message}")
            return ExecutionResult(success=False, error_msg=error_message)

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
