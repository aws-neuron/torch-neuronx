"""
Context manager for handling unsupported dtypes (float64, int64) on Neuron hardware.
"""

import logging
from contextlib import contextmanager
from typing import Any

import torch
from torch._logging import warning_once

logger = logging.getLogger(__name__)


def _warn_dtype_cast_once(dtype_name: str) -> None:
    """Show one-time warning about dtype casting."""
    target_dtype = "float32" if dtype_name == "float64" else "int32"
    warning_once(
        logger,
        f"Neuron backend does not support {dtype_name}. Automatically casting to {target_dtype}. "
        f"Consider using {target_dtype} directly for better performance until native "
        f"{dtype_name} support is added.",
    )


class DtypeAutocastContext:
    """Configure dtype autocasting for Neuron operations.

    Args:
        cast_float64 (bool, optional): Cast ``float64`` tensors to ``float32``.
            Defaults to ``True``.
        cast_int64 (bool, optional): Cast ``int64`` tensors to ``int32``. Defaults to
            ``True``.
        cast_dtype_params (bool, optional): Cast standalone ``torch.dtype`` arguments.
            Defaults to ``False``.
    """

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
                _warn_dtype_cast_once("float64")
            elif value.dtype == torch.int64:
                had_int64 = True
                _warn_dtype_cast_once("int64")
            return value.to(self._get_target_dtype(value.dtype)), had_float64, had_int64
        elif isinstance(value, tuple | list) and any(isinstance(v, torch.Tensor) for v in value):
            # Handle tuple/list of tensors
            new_value = []
            for v in value:
                if isinstance(v, torch.Tensor) and self._should_cast(v.dtype):
                    if v.dtype == torch.float64:
                        had_float64 = True
                        _warn_dtype_cast_once("float64")
                    elif v.dtype == torch.int64:
                        had_int64 = True
                        _warn_dtype_cast_once("int64")
                    new_value.append(v.to(self._get_target_dtype(v.dtype)))
                else:
                    new_value.append(v)
            return type(value)(new_value), had_float64, had_int64
        elif isinstance(value, torch.dtype) and self.cast_dtype_params:
            # Handle dtype kwarg
            if value == torch.float64 and self.cast_float64:
                _warn_dtype_cast_once("float64")
                return torch.float32, True, False
            elif value == torch.int64 and self.cast_int64:
                _warn_dtype_cast_once("int64")
                return torch.int32, False, True

        return value, had_float64, had_int64

    def process_args(
        self, args: tuple, kwargs: dict, output_params: tuple[str, ...] | None = None
    ) -> tuple[tuple, dict, dict[str, bool]]:
        """Cast unsupported input dtypes and record the original dtype families.

        Args:
            args (tuple): Positional arguments to process.
            kwargs (dict): Keyword arguments to process.
            output_params (tuple[str, ...] | None, optional): Keyword output names to
                exclude from casting. Defaults to ``None``.

        Returns:
            tuple[tuple, dict, dict[str, bool]]: Processed positional arguments,
            processed keyword arguments, and ``had_float64``/``had_int64`` flags.

        Note:
            Named output parameters retain their original dtypes only when passed by keyword;
            positional arguments are processed without signature binding. Top-level scalar
            tensors remain unchanged, while tensor elements of an immediate tuple or list are
            cast regardless of rank.
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
        """Restore output dtypes using the recorded dtype requirements.

        Args:
            output (Any): Tensor or tensor collection produced after dtype conversion.

        Returns:
            Any: Eligible tensors, including immediate tensor members of a tuple or list,
                restored to ``float64`` or ``int64``; other values are returned unchanged.
        """
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
        """Validate the keyword ``out`` dtype against the autocast configuration.

        Args:
            original_kwargs (dict): Original keyword arguments before input processing.

        Raises:
            TypeError: If an unsupported ``float64`` output cannot be autocast.

        Note:
            Call this with the original keyword arguments. Eligible ``out`` tensors set the
            recorded dtype flags so later output restoration copies back at that dtype.
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
                    # Similarly allow int64 outs by recording the requirement
                    # cast_int64=False means int64 is natively supported (S64 MLIR HLO)
                    elif tensor.dtype == torch.int64 and not self.had_int64 and self.cast_int64:
                        self.had_int64 = True


@contextmanager
def autocast_neuron(
    cast_float64: bool = True, cast_int64: bool = True, cast_dtype_params: bool = False
):
    """Yield a context for explicit Neuron device dtype conversion.

    Args:
        cast_float64 (bool, optional): Cast ``float64`` tensors to ``float32``.
            Defaults to ``True``.
        cast_int64 (bool, optional): Cast ``int64`` tensors to ``int32``. Defaults to
            ``True``.
        cast_dtype_params (bool, optional): Cast standalone ``torch.dtype`` arguments.
            Defaults to ``False``.

    Yields:
        DtypeAutocastContext: Context used to process inputs and restore outputs.
    """
    context = DtypeAutocastContext(
        cast_float64=cast_float64, cast_int64=cast_int64, cast_dtype_params=cast_dtype_params
    )
    yield context
