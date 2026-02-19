"""Refactored JAX operation implementation with clean separation."""

import logging
import traceback
from collections.abc import Callable

import torch

from torch_neuronx.python_ops.base import CompilableOpImpl, ExecutionResult
from torch_neuronx.python_ops.dtype_autocast import autocast_neuron
from torch_neuronx.python_ops.shared import ExecutionContext, ReductionOps

from .handlers import EmptyTensorHandler, OutputHandler
from .kernel import JaxKernel
from .registry import get_jax_function

logger = logging.getLogger(__name__)


class JaxOpImpl(CompilableOpImpl):
    """JAX-based operation implementation with clean architecture.

    This class provides a clean interface for executing JAX operations
    with proper separation of concerns:
    - EmptyTensorHandler: Handles empty tensor cases
    - OutputHandler: Manages output tensors and 'out' parameter
    - JaxKernel: Handles compilation and execution
    - ReductionOps: Configuration for reduction operations
    """

    def __init__(
        self,
        aten_op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        output_params: tuple[str, ...] | None = None,
        identity_value: int | float | bool | None | str = "auto",
    ):
        """Initialize the JAX operation implementation.

        Args:
            aten_op_name: ATen operation name (e.g., "aten::sqrt")
            static_argnums: Indices of static positional arguments
            static_argnames: Names of static keyword arguments
            output_params: Names of output tensor parameters
            identity_value: Identity value for reductions ("auto" to infer)
        """
        super().__init__()

        # Store configuration
        self.aten_op_name = aten_op_name
        self.static_argnums = static_argnums
        self.static_argnames = static_argnames
        self.output_params = output_params

        # Initialize handlers
        self.empty_handler = EmptyTensorHandler()
        self.output_handler = OutputHandler()

        # Determine if this is a reduction operation
        self._is_reduction = ReductionOps.is_reduction(aten_op_name)
        self._identity = self._determine_identity_value(identity_value)

        # Get JAX function
        self.jax_fn = self._get_jax_function()

        # Check for preprocessing
        self._uses_preprocessing = getattr(self.jax_fn, "uses_preprocessing", False)

        # Create kernel if not using preprocessing
        if not self._uses_preprocessing:
            self.kernel = self._create_kernel(self.jax_fn)
        else:
            self.kernel = None  # Will be created on first call

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation with automatic dtype casting.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            ExecutionResult with output tensor(s)
        """
        exec_context = ExecutionContext(original_inputs=args)
        # Use dtype autocast context
        with autocast_neuron() as context:
            # Process arguments, excluding output parameters from casting
            original_kwargs = kwargs
            args, kwargs, _ = context.process_args(args, kwargs, output_params=self.output_params)

            # Validate output dtypes
            context.validate_output_dtypes(original_kwargs)

            # Check for empty tensors
            if self._should_handle_empty(args):
                result = self._handle_empty_tensors(args, kwargs)
                if result is not None:
                    result.output = context.restore_dtypes(result.output)
                    return result

            # Check for 'out' parameter (but don't remove it from kwargs)
            out = kwargs.get("out", None)

            # Check for other output parameters (but don't remove them)
            output_tensors = []
            if self.output_params:
                for param_name in self.output_params:
                    output_tensor = kwargs.get(param_name, None)
                    if output_tensor is not None:
                        output_tensors.append(output_tensor)

            # If we have output tensors from output_params, use them as 'out'
            if output_tensors and out is None:
                out = tuple(output_tensors) if len(output_tensors) > 1 else output_tensors[0]

            # Build execution context with original inputs and kwargs
            if out is not None:
                if isinstance(out, tuple):
                    exec_context.expected_dtypes = [
                        t.dtype if isinstance(t, torch.Tensor) else None for t in out
                    ]
                elif isinstance(out, torch.Tensor):
                    exec_context.expected_dtypes = [out.dtype]

            # Execute implementation with ALL original kwargs
            try:
                result = self._execute_impl(*args, context=exec_context, **kwargs)

                # Handle output parameter for in-place operations
                if out is not None:
                    result = self._handle_out_parameter(result, out)

                return ExecutionResult(success=True, output=result)

            except Exception as e:
                # Do not fall back for user/input contract errors; match PyTorch behavior
                error_traceback = traceback.format_exc()
                if isinstance(e, IndexError | ValueError | TypeError):
                    raise
                error_message = f"{e!s}\n{error_traceback}"
                logger.debug(f"Execution failed: {error_message}")
                return ExecutionResult(success=False, error_msg=error_message)

    def _execute_impl(
        self, *args, context: ExecutionContext | None = None, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute using the kernel.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Output tensor(s)
        """
        if self._uses_preprocessing:
            # Ensure we have an execution context
            context = context or ExecutionContext(original_inputs=args)

            # Call preprocessor
            actual_jax_fn, processed_args, processed_kwargs = self.jax_fn(*args, **kwargs)

            # Create kernel on first call
            if self.kernel is None:
                self.kernel = self._create_kernel(actual_jax_fn)

            # Execute with processed args and explicit context
            return self.kernel(*processed_args, context=context, **processed_kwargs)
        else:
            # Direct execution (no preprocessing). Pass context uniformly.
            context = context or ExecutionContext(original_inputs=args)
            return self.kernel(*args, context=context, **kwargs)

    def _handle_empty_tensors(self, args: tuple, kwargs: dict) -> ExecutionResult | None:
        """Handle empty tensor case.

        Args:
            args: Input arguments
            kwargs: Keyword arguments

        Returns:
            ExecutionResult if handled, None otherwise
        """
        if not self.empty_handler.check_for_empty(*args):
            return None

        try:
            if self._uses_preprocessing:
                actual_jax_fn, args, kwargs = self.jax_fn(*args, **kwargs)
            else:
                actual_jax_fn = self.jax_fn

            result = self.empty_handler.handle_empty_operation(
                actual_jax_fn,
                args,
                kwargs,
                self.aten_op_name,
                self.static_argnums,
                self.static_argnames,
            )

            # Handle 'out' parameter
            out = kwargs.get("out")
            if out is not None:
                result = self._handle_out_parameter(result, out)

            return ExecutionResult(success=True, output=result)

        except Exception as e:
            logger.error(f"Empty tensor handling failed: {e}")
            return ExecutionResult(success=False, error_msg=str(e))

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

    def _determine_identity_value(
        self, identity_value: int | float | bool | None | str
    ) -> int | float | bool | None:
        """Determine identity value for reductions.

        Args:
            identity_value: Configured identity value

        Returns:
            Resolved identity value
        """
        if not self._is_reduction:
            return None

        if identity_value == "auto":
            return ReductionOps.get_identity_value(self.aten_op_name)

        return identity_value

    def _get_jax_function(self) -> Callable:
        """Get JAX function for the operation.

        Returns:
            JAX function

        Raises:
            ValueError: If no JAX implementation found
        """
        jax_fn = get_jax_function(self.aten_op_name)
        if jax_fn is None:
            raise ValueError(f"No JAX implementation found for {self.aten_op_name}")
        return jax_fn

    def _create_kernel(self, jax_fn: Callable) -> JaxKernel:
        """Create JAX kernel for the operation.

        Args:
            jax_fn: JAX function

        Returns:
            JaxKernel instance
        """
        # Always use the full aten operation name for consistency
        return JaxKernel(
            jax_fn,
            op_name=self.aten_op_name,
            static_argnums=self.static_argnums,
            static_argnames=self.static_argnames,
            output_params=self.output_params,
        )

    def _get_jax_equivalent(self) -> Callable:
        """Get the JAX equivalent function (for compatibility).

        Returns:
            JAX function
        """
        return self.jax_fn

    def _get_identity_value(self) -> int | float | bool | None:
        """Get identity value for reductions (for compatibility).

        Returns:
            Identity value if applicable
        """
        return self._identity
