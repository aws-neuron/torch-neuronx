"""Handler for empty tensor operations."""

import logging
from collections.abc import Callable

import jax
import torch

from torch_neuronx.python_ops.handlers import BaseEmptyTensorHandler
from torch_neuronx.python_ops.shared import IndexOps, ReductionOps

from ..type_converter import JaxTypeConverter

logger = logging.getLogger(__name__)


class EmptyTensorHandler(BaseEmptyTensorHandler):
    """Handles operations on empty tensors."""

    def __init__(self):
        """Initialize the handler."""
        self.type_converter = JaxTypeConverter()

    def handle_empty_operation(
        self,
        jax_fn: Callable,
        args: tuple,
        kwargs: dict,
        op_name: str | None = None,
        static_argnums: tuple | None = None,
        static_argnames: tuple | None = None,
    ) -> torch.Tensor:
        """Handle operation on empty tensors.

        Args:
            jax_fn: JAX function for the operation
            args: Input arguments
            kwargs: Keyword arguments
            op_name: Optional operation name for reduction handling
            static_argnums: static positional arguments
            static_argnames: static keyword arguments

        Returns:
            Result tensor(s) for empty input

        Raises:
            RuntimeError: If operation cannot be performed on empty tensors
        """
        # Convert arguments for shape inference
        jax_args = self._convert_args_to_abstract(args)
        jax_kwargs = self._convert_kwargs_to_abstract(kwargs)

        # Handle reductions specially
        if op_name and ReductionOps.is_reduction(op_name):
            return self._handle_empty_reduction(
                jax_fn, jax_args, jax_kwargs, args, kwargs, op_name, static_argnums, static_argnames
            )

        if op_name and IndexOps.is_index(op_name):
            return self._handle_empty_index(
                jax_fn, jax_args, jax_kwargs, args, kwargs, op_name, static_argnums, static_argnames
            )

        # For non-reductions, use JAX shape inference
        return self._handle_empty_general(
            jax_fn, jax_args, jax_kwargs, args, kwargs, static_argnums, static_argnames
        )

    def _handle_empty_index(
        self,
        jax_fn: Callable,
        jax_args: tuple,
        jax_kwargs: dict,
        original_args: tuple,
        original_kwargs: dict,
        op_name: str,
        static_argnums: tuple | None = None,
        static_argnames: tuple | None = None,
    ) -> torch.Tensor:
        """

        Handle empty index tensor

        Args:
            jax_fn: JAX function
            jax_args: Abstract JAX arguments
            jax_kwargs: Abstract JAX kwargs
            original_args: Original torch arguments
            original_kwargs: Original torch kwargs
            op_name: Operation name
            static_argnums: static positional arguments
            static_argnames: static keyword arguments

        Returns:
            Result with appropriate identity value

        Returns:
            Input tensor if empty index tensor, else default behavior
        """
        index_argnum = IndexOps.get_index_argnum(op_name)
        input_argnum = IndexOps.get_input_argnum(op_name)
        if index_argnum is not None and input_argnum is not None:
            input_tensor, index_tensor = original_args[input_argnum], original_args[index_argnum]
            if (
                isinstance(input_tensor, torch.Tensor)
                and isinstance(index_tensor, torch.Tensor)
                and input_tensor.numel() > 0
                and index_tensor.numel() == 0
            ):
                return input_tensor
        return self._handle_empty_general(
            jax_fn, jax_args, jax_kwargs, jax_args, jax_kwargs, static_argnums, static_argnames
        )

    def _handle_empty_reduction(
        self,
        jax_fn: Callable,
        jax_args: tuple,
        jax_kwargs: dict,
        original_args: tuple,
        original_kwargs: dict,
        op_name: str,
        static_argnums: tuple | None = None,
        static_argnames: tuple | None = None,
    ) -> torch.Tensor:
        """Handle empty tensor reduction.

        Args:
            jax_fn: JAX function
            jax_args: Abstract JAX arguments
            jax_kwargs: Abstract JAX kwargs
            original_args: Original torch arguments
            original_kwargs: Original torch kwargs
            op_name: Operation name
            static_argnums: static positional arguments
            static_argnames: static keyword arguments

        Returns:
            Result with appropriate identity value
        """
        # Get identity value
        identity = ReductionOps.get_identity_value(op_name)
        if identity is None:
            raise RuntimeError(f"Cannot perform {op_name} on empty tensor (no identity value)")

        # Create reduction wrapper if needed
        if "dim" in original_kwargs:
            dim_val = original_kwargs["dim"]
            keepdim_val = original_kwargs.get("keepdim", False)

            def reduction_fn(x):
                return jax_fn(x, dim=dim_val, keepdim=keepdim_val)

            # Remove from kwargs for eval_shape
            jax_kwargs = {k: v for k, v in jax_kwargs.items() if k not in ["dim", "keepdim"]}
        else:
            reduction_fn = jax_fn

        if static_argnums or static_argnames:
            jit_kwargs = {}
            if static_argnums:
                jit_kwargs["static_argnums"] = static_argnums
            if static_argnames:
                jit_kwargs["static_argnames"] = static_argnames

            jitted_fn = jax.jit(reduction_fn, **jit_kwargs)
            output_info = jitted_fn.eval_shape(*jax_args, **jax_kwargs)
        else:
            output_info = jax.eval_shape(reduction_fn, *jax_args, **jax_kwargs)

        # Get device and dtype
        device = self._get_device(original_args)
        dtype = self._get_dtype(original_args)

        # Create output with identity value
        if isinstance(output_info, tuple):
            # Handle multi-output reductions
            value_info = output_info[0]
            output_value = torch.full(value_info.shape, identity, dtype=dtype, device=device)

            if len(output_info) > 1:
                # Handle index outputs (e.g., argmax)
                indices_info = output_info[1]
                indices = torch.zeros(indices_info.shape, dtype=torch.long, device=device)
                return (output_value, indices)

            return output_value
        else:
            return torch.full(output_info.shape, identity, dtype=dtype, device=device)

    def _handle_empty_general(
        self,
        jax_fn: Callable,
        jax_args: tuple,
        jax_kwargs: dict,
        original_args: tuple,
        original_kwargs: dict,
        static_argnums: tuple | None = None,
        static_argnames: tuple | None = None,
    ) -> torch.Tensor:
        """Handle empty tensor for general operations.

        Args:
            jax_fn: JAX function
            jax_args: Abstract JAX arguments
            jax_kwargs: Abstract JAX kwargs
            original_args: Original torch arguments
            original_kwargs: Original torch kwargs
            static_argnums: static positional arguments
            static_argnames: static keyword arguments

        Returns:
            Empty result tensor with correct shape/dtype
        """
        if static_argnums or static_argnames:
            # Use jit with static arguments
            jit_kwargs = {}
            if static_argnums:
                jit_kwargs["static_argnums"] = static_argnums
            if static_argnames:
                jit_kwargs["static_argnames"] = static_argnames

            jitted_fn = jax.jit(jax_fn, **jit_kwargs)
            output_info = jitted_fn.eval_shape(*jax_args, **jax_kwargs)
        else:
            # No static arguments, use regular eval_shape
            output_info = jax.eval_shape(jax_fn, *jax_args, **jax_kwargs)

        # Get device
        device = self._get_device(original_args)

        # Handle tuple outputs
        if isinstance(output_info, tuple):
            outputs = []
            for info in output_info:
                torch_dtype = self.type_converter.jax_to_torch_dtype(info.dtype)
                output = torch.empty(info.shape, dtype=torch_dtype, device=device)
                outputs.append(output)
            return tuple(outputs)

        # Single output
        torch_dtype = self.type_converter.jax_to_torch_dtype(output_info.dtype)
        return torch.zeros(output_info.shape, dtype=torch_dtype, device=device)

    def _convert_args_to_abstract(self, args: tuple) -> tuple:
        """Convert arguments to JAX abstract values.

        Args:
            args: PyTorch arguments

        Returns:
            JAX abstract values
        """
        result = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                result.append(self.type_converter.tensor_to_jax_abstract(arg))
            elif isinstance(arg, list | tuple):
                result.append(self._convert_args_to_abstract(arg))
            else:
                result.append(arg)
        return tuple(result)

    def _convert_kwargs_to_abstract(self, kwargs: dict) -> dict:
        """Convert kwargs to JAX abstract values.

        Args:
            kwargs: PyTorch kwargs

        Returns:
            JAX abstract kwargs
        """
        result = {}
        for key, value in kwargs.items():
            if key == "out":
                continue  # Skip 'out' parameter
            elif isinstance(value, torch.Tensor):
                result[key] = self.type_converter.tensor_to_jax_abstract(value)
            else:
                result[key] = value
        return result

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

    def _get_dtype(self, args: tuple) -> torch.dtype:
        """Get dtype from arguments.

        Args:
            args: Input arguments

        Returns:
            Dtype from first tensor, or float32
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg.dtype
        return torch.float32
