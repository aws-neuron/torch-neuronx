"""Based the Op implementation on JaxOpImpl class."""

import logging
import traceback
from collections.abc import Callable

import torch

from torch_neuronx.python_ops.base import CompilableOpImpl, ExecutionResult
from torch_neuronx.python_ops.dtype_autocast import autocast_neuron
from torch_neuronx.python_ops.handlers import BaseEmptyTensorHandler, BaseOutputHandler

from .kernel import XLABuilderKernel

logger = logging.getLogger(__name__)


class XLABuilderOpImpl(CompilableOpImpl):
    """Implementation of XLABuilder based operations based on the
    CompilableOpImpl interface.
    This class provides functionality to execute XLA operations with
    automatic dtype casting, empty tensor handling, and output parameter management.
    Note: empty tensor handling is not implemented yet, and would be added
    in subsequent revisions

    Args:
        op_name (str): Name of the XLA operation
        static_argnums (tuple[int, ...] | None): Indices of static arguments
        static_argnames (tuple[str, ...] | None): Names of static keyword arguments
        output_params (tuple[str, ...] | None): Names of output parameters
        has_collectives (bool): Whether operation involves collective communications
    """

    def __init__(
        self,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        output_params: tuple[str, ...] | None = None,
        has_collectives: bool = False,
    ):
        super().__init__()

        # Store configuration
        self.op_name = op_name
        self.static_argnums = static_argnums
        self.static_argnames = static_argnames
        self.output_params = output_params
        self.has_collectives = has_collectives
        self.kernel = None

        # Initialize handlers
        self.output_handler = BaseOutputHandler()
        self.empty_handler = BaseEmptyTensorHandler()

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the XLA operation with automatic dtype casting.

        Handles dtype casting, empty tensor checking, and output parameter management.
        Executes the operation within an autocast context.

        Args:
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            ExecutionResult: Contains success status and output tensor(s) or error message
        """
        # Use dtype autocast context
        with autocast_neuron() as context:
            # Process arguments, excluding output parameters from casting
            original_kwargs = kwargs
            args, kwargs, _ = context.process_args(args, kwargs, output_params=self.output_params)

            # Validate output dtypes
            context.validate_output_dtypes(original_kwargs)

            # Check for empty tensors
            if self._should_handle_empty(args):
                # TODO: Need to handle empty tensors here
                raise NotImplementedError("Empty tensor handling is not implemented.")

            # Execute implementation
            try:
                if self.kernel is None:
                    self.kernel = self._create_kernel(self.hlo_fn)
                result = self._execute_impl(*args, **kwargs)

                # Handle output parameter
                out = kwargs.get("out", None)
                if out is not None:
                    result = self._handle_out_parameter(result, out)

                # Restore original dtypes
                result = context.restore_dtypes(result)

                return ExecutionResult(success=True, output=result)

            except Exception as e:
                error_traceback = traceback.format_exc()
                error_message = f"{e!s}\n{error_traceback}"
                logger.debug(f"Execution failed: {error_message}")
                return ExecutionResult(success=False, error_msg=error_message)

    def _handle_out_parameter(
        self, result: torch.Tensor | tuple, out: torch.Tensor
    ) -> torch.Tensor | tuple:
        """Handle 'out' parameter for in-place operations.

        Args:
            result: Computed result
            out: Output tensor

        Returns:
            Result with 'out' handled
        """
        return self.output_handler.handle_output_parameter(result, out)

    def _should_handle_empty(self, args: tuple) -> bool:
        """Check if empty tensor handling is needed.

        Args:
            args: Input arguments

        Returns:
            True if empty handling needed
        """
        return self.empty_handler.check_for_empty(*args)

    def _create_kernel(self, hlo_fn: Callable) -> XLABuilderKernel:
        """Create an XLABuilderKernel instance for the operation.

        Args:
            hlo_fn (Callable): HLO function implementing the operation

        Returns:
            XLABuilderKernel: Kernel instance configured with operation parameters
        """
        return XLABuilderKernel(
            hlo_fn,
            op_name=self.op_name,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
            output_params=self.output_params,
            has_collectives=self.has_collectives,
        )
