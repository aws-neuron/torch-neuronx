"""
Context manager for handling unsupported dtypes (float64, int64) on Neuron hardware.
"""

from contextlib import contextmanager
from typing import Any

import torch


class DtypeAutocastContext:
    """Context for handling unsupported dtypes (float64, int64) on Neuron hardware"""

    def __init__(
        self, cast_float64: bool = True, cast_int64: bool = True, cast_dtype_params: bool = False
    ):
        """
        Args:
            cast_float64: If True, cast float64 to float32
            cast_int64: If True, cast int64 to int32
            cast_dtype_params: If True, cast dtype parameters
        """
        self.cast_float64 = cast_float64
        self.cast_int64 = cast_int64
        self.cast_dtype_params = cast_dtype_params

        # Track which dtypes we had in inputs (for restoration)
        self.had_float64 = False
        self.had_int64 = False

    def _should_cast(self, dtype: torch.dtype) -> bool:
        """Check if a dtype should be cast"""
        if dtype == torch.float64 and self.cast_float64:
            return True
        return bool(dtype == torch.int64 and self.cast_int64)

    def _get_target_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Get the target dtype for casting"""
        if dtype == torch.float64:
            return torch.float32
        if dtype == torch.int64:
            return torch.int32
        return dtype

    def _process_value(self, value: Any) -> tuple[Any, bool, bool]:
        """
        Process a single value, returning (processed_value, had_float64, had_int64)
        """
        had_float64 = False
        had_int64 = False

        if isinstance(value, torch.Tensor) and value.ndim != 0 and self._should_cast(value.dtype):
            if value.dtype == torch.float64:
                had_float64 = True
            elif value.dtype == torch.int64:
                had_int64 = True
            return value.to(self._get_target_dtype(value.dtype)), had_float64, had_int64
        elif isinstance(value, tuple | list) and any(isinstance(v, torch.Tensor) for v in value):
            # Handle tuple/list of tensors
            new_value = []
            for v in value:
                if isinstance(v, torch.Tensor) and self._should_cast(v.dtype):
                    if v.dtype == torch.float64:
                        had_float64 = True
                    elif v.dtype == torch.int64:
                        had_int64 = True
                    new_value.append(v.to(self._get_target_dtype(v.dtype)))
                else:
                    new_value.append(v)
            return type(value)(new_value), had_float64, had_int64
        elif isinstance(value, torch.dtype) and self.cast_dtype_params:
            # Handle dtype kwarg
            if value == torch.float64 and self.cast_float64:
                return torch.float32, True, False
            elif value == torch.int64 and self.cast_int64:
                return torch.int32, False, True

        return value, had_float64, had_int64

    def process_args(
        self, args: tuple, kwargs: dict, output_params: tuple[str, ...] | None = None
    ) -> tuple[tuple, dict, dict[str, bool]]:
        """
        Process arguments: cast unsupported dtypes and track what we had.
        Returns: (new_args, new_kwargs, dtype_info)

        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            output_params: Names of output parameters to exclude from casting

        Note: Output parameters are never cast - they keep their original dtypes.
        """
        dtype_info = {"had_float64": False, "had_int64": False}

        # Build set of output parameter names
        output_param_names = {"out"}  # Always include 'out'
        if output_params:
            output_param_names.update(output_params)

        # Cast positional arguments
        new_args = []
        for arg in args:
            processed_arg, had_f64, had_i64 = self._process_value(arg)
            new_args.append(processed_arg)
            dtype_info["had_float64"] |= had_f64
            dtype_info["had_int64"] |= had_i64

        # Cast keyword arguments (excluding output parameters)
        new_kwargs = {}
        for key, value in kwargs.items():
            if key in output_param_names:
                # Keep output parameters unchanged
                new_kwargs[key] = value
            else:
                # Process input parameters
                processed_value, had_f64, had_i64 = self._process_value(value)
                new_kwargs[key] = processed_value
                dtype_info["had_float64"] |= had_f64
                dtype_info["had_int64"] |= had_i64

        # Store for later use (based on inputs only, not outputs)
        self.had_float64 = dtype_info["had_float64"]
        self.had_int64 = dtype_info["had_int64"]

        return tuple(new_args), new_kwargs, dtype_info

    def restore_dtypes(self, output: Any) -> Any:
        """Restore output dtypes based on PyTorch promotion rules"""
        if output is None:
            return output

        if isinstance(output, torch.Tensor):
            # Restore float64 if we had float64 inputs and output is float32
            if self.had_float64 and output.dtype == torch.float32:
                return output.to(torch.float64)
            # Restore int64 if we had int64 inputs and output is int32
            elif self.had_int64 and output.dtype == torch.int32:
                return output.to(torch.int64)
            return output
        elif isinstance(output, tuple | list):
            # Recursively handle collections
            result = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    if self.had_float64 and item.dtype == torch.float32:
                        result.append(item.to(torch.float64))
                    elif self.had_int64 and item.dtype == torch.int32:
                        result.append(item.to(torch.int64))
                    else:
                        result.append(item)
                else:
                    result.append(item)
            return type(output)(result)
        return output

    def validate_output_dtypes(self, original_kwargs: dict) -> None:
        """Check that unsupported output dtypes are only requested with matching inputs.

        This should be called with the ORIGINAL kwargs before processing, not after.
        """
        out = original_kwargs.get("out")
        if out is not None:
            tensors = (
                [out]
                if isinstance(out, torch.Tensor)
                else list(out)
                if isinstance(out, tuple | list)
                else []
            )

            for tensor in tensors:
                if isinstance(tensor, torch.Tensor):
                    if tensor.dtype == torch.float64 and not self.had_float64:
                        if self.cast_float64:
                            # Allow float64 out tensors by marking that we should
                            # restore dtype after execution even if no float64 inputs
                            self.had_float64 = True
                        else:
                            raise TypeError(
                                "Output tensor has dtype float64 which is not supported on Neuron. "
                                "Please use float32 for the output tensor."
                            )
                    elif tensor.dtype == torch.int64 and not self.had_int64:
                        if self.cast_int64:
                            # Similarly allow int64 outs by recording the requirement
                            self.had_int64 = True
                        else:
                            raise TypeError(
                                "Output tensor has dtype int64 which is not supported on Neuron. "
                                "Please use int32 for the output tensor."
                            )


@contextmanager
def autocast_neuron(
    cast_float64: bool = True, cast_int64: bool = True, cast_dtype_params: bool = False
):
    """
    Context manager for handling unsupported dtypes on Neuron hardware.

    Args:
        cast_float64: If True, cast float64 to float32 (default: True)
        cast_int64: If True, cast int64 to int32 (default: True)
        cast_dtype_params: If True, cast dtype parameters (default: False)

    Example:
        with autocast_neuron():
            # Both float64 and int64 are handled
            result = torch.matmul(tensor_f64, tensor_i64)

        with autocast_neuron(cast_int64=False):
            # Only float64 is handled, int64 will error if unsupported
            result = some_op(tensor_f64)

        with autocast_neuron(cast_dtype_params=True):
            # Also cast dtype parameters (needed for MLIR path)
            result = some_op(tensor_f64, dtype=torch.int64)
    """
    context = DtypeAutocastContext(
        cast_float64=cast_float64, cast_int64=cast_int64, cast_dtype_params=cast_dtype_params
    )
    yield context
