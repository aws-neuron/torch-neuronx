import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.schema_utils import is_inplace_op
from torch_neuronx.utils import move_pytree_to_cpu

from .handlers import BaseEmptyTensorHandler

# Set to track debug messages that have been printed once
_debug_messages_printed = set()


def _needs_python_side_logging(impl, *args):
    """
    Helper function to determine if op should be logged from Python.

    Ops that go through the C++ async pipeline have their execution logged
    from C++. Ops without a kernel or with empty tensors are handled in
    Python and need Python-side logging.
    """
    return (
        not hasattr(impl, "kernel")
        or impl.kernel is None
        or BaseEmptyTensorHandler().check_for_empty(*args)
    )


@dataclass
class ExecutionResult:
    """Represent the outcome of an operation implementation.

    Args:
        success (bool): Whether execution completed successfully.
        output (Any | None, optional): Produced value. Defaults to ``None``.
        error_msg (str | None, optional): Failure detail. Defaults to ``None``.
    """

    success: bool
    output: Any | None = None
    error_msg: str | None = None


class OperationImplementation(ABC):
    """Define the base interface for operation implementations."""

    # Execution policy toggles (default: current behavior)
    # - When DISABLE_DTYPE_AUTOCAST is True, skip float64->float32 input pre-cast and
    #   related post-processing of results.
    # - When DISABLE_EMPTY_SHORTCIRCUIT is True, skip empty-tensor short-circuiting.
    DISABLE_DTYPE_AUTOCAST: bool = False
    DISABLE_EMPTY_SHORTCIRCUIT: bool = False

    def translate_args(self, *args, **kwargs):
        """Translate raw ATen dispatcher arguments for this implementation.

        Args:
            *args (Any): Positional ATen dispatcher arguments.
            **kwargs (Any): Keyword ATen dispatcher arguments.

        Returns:
            tuple: A pair containing the positional-argument tuple and
            keyword-argument dictionary.

        Note:
            Override this when an ATen schema supplies arguments that an
            implementation does not consume.
        """
        return args, kwargs

    def can_handle(self, *args, **kwargs) -> bool:
        """Check whether this implementation accepts the supplied arguments.

        Args:
            *args (Any): Positional operation arguments.
            **kwargs (Any): Keyword operation arguments.

        Returns:
            bool: ``True`` when the implementation can handle the arguments.

        Raises:
            RuntimeError: If a non-scalar tensor passed directly, inside a positional
                list or tuple, or as a direct keyword is not on a Neuron device.

        Note:
            This base check does not reject complex dtypes.
        """
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if arg.ndim != 0 and arg.device.type != "neuron":
                    raise RuntimeError(
                        f"Non-scalar tensor arg{i} is on {arg.device.type} device, expected neuron"
                    )
            elif isinstance(arg, list | tuple):
                for j, value in enumerate(arg):
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim != 0
                        and value.device.type != "neuron"
                    ):
                        raise RuntimeError(
                            f"Non-scalar tensor arg{i}[{j}] is on {value.device.type} "
                            "device, expected neuron"
                        )

        for key, kwarg in kwargs.items():
            if (
                isinstance(kwarg, torch.Tensor)
                and kwarg.ndim != 0
                and kwarg.device.type != "neuron"
            ):
                raise RuntimeError(
                    f"Non-scalar tensor kwarg '{key}' is on {kwarg.device.type}"
                    "device, expected neuron"
                )

        return True

    @torch._dynamo.disable
    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation according to its policy flags.

        Args:
            *args (Any): Positional operation arguments.
            **kwargs (Any): Keyword operation arguments.

        Returns:
            ExecutionResult: The implementation result.

        Raises:
            TypeError: If a float64 ``out`` tensor is supplied without a float64
                positional input.
            NotImplementedError: If an empty input reaches the default empty-tensor
                handler.

        Note:
            By default, this casts non-scalar float64 tensor inputs to float32,
            short-circuits empty tensors, and casts float64 tensors in tuple
            outputs to float32. Class flags can disable autocasting or the
            empty-tensor short circuit.
        """
        # pre-cast (skip when disabled)
        args2, kwargs2 = args, kwargs
        if not self.DISABLE_DTYPE_AUTOCAST:
            args2, kwargs2 = self._cast_float64_to_float32(args2, kwargs2)

        # empty-tensor short-circuit (skip when disabled)
        if not self.DISABLE_EMPTY_SHORTCIRCUIT:
            empty_result = self._check_and_handle_empty(*args2, **kwargs2)
            if empty_result is not None:
                # If dtype autocast was enabled and used an internal out buffer,
                # keep existing copy-back behavior.
                if (
                    not self.DISABLE_DTYPE_AUTOCAST
                    and hasattr(self, "_float64_out")
                    and self._float64_out is not None
                ):
                    self._float64_out.copy_(empty_result.output.to(torch.float64))
                    empty_result.output = self._float64_out
                    self._float64_out = None  # Clean up
                return empty_result

        # Actual implementation
        result = self._execute_impl(*args2, **kwargs2)

        # post-processing (skip when autocast disabled)
        # ToDo(thangakr): Handle gradient flowing backward
        if (
            not self.DISABLE_DTYPE_AUTOCAST
            and result.success
            and result.output is not None
            and isinstance(result.output, tuple)
        ):
            # Cast any float64 tensors in the tuple to float32
            new_output = []
            for item in result.output:
                if isinstance(item, torch.Tensor) and item.dtype == torch.float64:
                    new_output.append(item.to(torch.float32))
                else:
                    new_output.append(item)
            result.output = tuple(new_output)

        return result

    def _has_float64_tensors(self, args):
        """Check if args contain any float64 tensors."""
        for arg in args:
            if (isinstance(arg, torch.Tensor) and arg.dtype == torch.float64) or (
                isinstance(arg, (list | tuple))
                and any(
                    isinstance(elem, torch.Tensor) and elem.dtype == torch.float64 for elem in arg
                )
            ):
                return True
        return False

    def _cast_tensor_if_float64(self, tensor):
        """Cast tensor to float32 if it's float64 and not a scalar."""
        return (
            tensor.to(torch.float32)
            if isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.float64
            and getattr(tensor, "ndim", None) != 0
            else tensor
        )

    def _cast_float64_to_float32(self, args, kwargs):
        """Cast float64 tensor inputs to float32

        Neuron hardware doesn't support float64, but we can handle float64 inputs
        by converting them to float32. This avoids crashes when ops fallback to CPU
        and produce float64 results that get passed to subsequent Neuron ops.
        """
        has_float64_input = self._has_float64_tensors(args)

        # Cast args
        new_args = []
        for arg in args:
            if isinstance(arg, (list | tuple)):
                # Handle list of tensors (e.g., for torch.cat)
                promoted_tensors = [self._cast_tensor_if_float64(item) for item in arg]
                new_args.append(type(arg)(promoted_tensors))  # Preserve list vs tuple
            else:
                new_args.append(self._cast_tensor_if_float64(arg))

        # Cast kwargs
        new_kwargs = {}
        float64_out = None

        for key, value in kwargs.items():
            if key == "out" and value is not None and isinstance(value, torch.Tensor):
                if value.dtype == torch.float64:
                    if has_float64_input:
                        # If we have float64 inputs and float64 output, we'll handle it
                        # by using a float32 output internally and copying back
                        float64_out = value
                        # Create a float32 output tensor for internal use
                        new_kwargs[key] = None  # Let the op create its own float32 output
                    else:
                        # Float64 output without float64 inputs is not allowed
                        raise TypeError(
                            "Output tensor has dtype float64 which is not supported on Neuron. "
                            "Please use float32 for the output tensor."
                        )
                else:
                    new_kwargs[key] = value
            else:
                new_kwargs[key] = self._cast_tensor_if_float64(value)

        # Store the original float64 output tensor if we need to copy back to it
        self._float64_out = float64_out

        return tuple(new_args), new_kwargs

    def _check_and_handle_empty(self, *args, **kwargs) -> ExecutionResult | None:
        """Returns ExecutionResult if empty tensor detected, None otherwise"""
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if any(t.numel() == 0 for t in tensors):
            return self._handle_empty_tensor(*args, **kwargs)
        return None

    def _handle_empty_tensor(self, *args, **kwargs) -> ExecutionResult:
        """Handle empty tensor case - override in subclasses for specific behavior"""
        raise NotImplementedError(
            f"Empty tensor handling not implemented for {self.__class__.__name__}. "
            f"Override _handle_empty_tensor() method."
        )

    @abstractmethod
    def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
        """Actual implementation - called only for non-empty tensors"""
        pass

    @property
    def priority(self) -> int:
        """Return the implementation selection priority.

        Returns:
            int: The priority, where higher values are tried first. Defaults to ``50``.
        """
        return 50


class UnaryOpImplementation(OperationImplementation):
    """Provide base behavior for element-wise unary operation implementations."""

    def _handle_empty_tensor(self, *args, **kwargs) -> ExecutionResult:
        """Unary ops preserve input shape for empty tensors"""
        # Extract input tensor (first argument) and optional out parameter
        input = args[0] if args else None
        if input is None or not isinstance(input, torch.Tensor):
            return ExecutionResult(success=False, error_msg="No input tensor provided")

        out = kwargs.get("out")
        output = io_tensor.empty_like(input) if out is None else out
        return ExecutionResult(success=True, output=output)

    def _validate_shapes(self, input: torch.Tensor, out=None) -> bool:
        """Validate that output shape matches input shape"""
        if out is not None and out.shape != input.shape:
            raise RuntimeError(f"Output shape {out.shape} doesn't match input {input.shape}")
        return True


class BinaryOpImplementation(OperationImplementation):
    """Provide base behavior for element-wise binary operation implementations."""

    def _handle_empty_tensor(self, input1, input2, *, out=None, **kwargs) -> ExecutionResult:
        """Binary ops follow broadcasting rules for empty tensors"""
        # Determine output shape using broadcasting rules
        shape1 = input1.shape if isinstance(input1, torch.Tensor) else torch.Size([])
        shape2 = input2.shape if isinstance(input2, torch.Tensor) else torch.Size([])

        try:
            output_shape = torch.broadcast_shapes(shape1, shape2)
        except RuntimeError as e:
            return ExecutionResult(success=False, error_msg=str(e))

        # Get output dtype and device
        dtype = self._get_output_dtype(input1, input2)
        device = input1.device if isinstance(input1, torch.Tensor) else input2.device

        # Create output tensor
        output = io_tensor.empty(output_shape, dtype=dtype, device=device) if out is None else out
        return ExecutionResult(success=True, output=output)

    def _get_output_dtype(self, input1, input2):
        """Determine output dtype from inputs"""
        if isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
            # Use PyTorch's type promotion rules
            return torch.result_type(input1, input2)
        elif isinstance(input1, torch.Tensor):
            return input1.dtype
        else:
            return input2.dtype

    def _validate_broadcasting(self, input1, input2) -> bool:
        """Validate that tensors can be broadcast together"""
        shape1 = input1.shape if isinstance(input1, torch.Tensor) else torch.Size([])
        shape2 = input2.shape if isinstance(input2, torch.Tensor) else torch.Size([])
        try:
            torch.broadcast_shapes(shape1, shape2)
            return True
        except RuntimeError:
            return False


class ReductionOpImplementation(OperationImplementation):
    """Provide base behavior for reduction operation implementations."""

    @abstractmethod
    def _get_identity_value(self) -> Any:
        """Return the identity value for this reduction operation

        Returns None if no identity exists (e.g., max, min).
        """
        pass

    def _handle_empty_tensor(
        self, input: torch.Tensor, dim=None, keepdim=False, *, out=None, **kwargs
    ) -> ExecutionResult:
        """Reduction ops return identity values or error for empty tensors"""
        # Calculate output shape after reduction
        output_shape = self._get_reduced_shape(input.shape, dim, keepdim)

        # Get identity value
        identity = self._get_identity_value()
        if identity is None:
            # Operations like max/min have no identity
            return ExecutionResult(
                success=False,
                error_msg=(
                    f"Cannot perform {self.__class__.__name__} on empty tensor (no identity value)"
                ),
            )

        # Create output with identity value
        output = torch.full(output_shape, identity, dtype=input.dtype, device=input.device)
        if out is not None:
            out.copy_(output)
            output = out

        return ExecutionResult(success=True, output=output)

    def _get_reduced_shape(self, shape, dim, keepdim):
        """Calculate shape after reduction"""
        if dim is None:
            # Reduce all dimensions
            return torch.Size([]) if not keepdim else torch.Size([1] * len(shape))

        # Handle single dimension or tuple of dimensions
        if isinstance(dim, int):
            dim = [dim]

        # Normalize negative dimensions
        dim = [d if d >= 0 else len(shape) + d for d in dim]

        # Build output shape
        output_shape = []
        for i, size in enumerate(shape):
            if i in dim:
                if keepdim:
                    output_shape.append(1)
            else:
                output_shape.append(size)

        return torch.Size(output_shape)


class MatrixOpImplementation(OperationImplementation):
    """Provide base behavior for matrix operation implementations."""

    def _handle_empty_tensor(self, *args, **kwargs) -> ExecutionResult:
        """Matrix ops need careful dimension checking for empty tensors"""
        try:
            output_shape = self._compute_output_shape(*args)
            dtype = self._get_output_dtype(*args)
            device = args[0].device if isinstance(args[0], torch.Tensor) else args[1].device

            out = kwargs.get("out")
            output = (
                io_tensor.empty(output_shape, dtype=dtype, device=device) if out is None else out
            )

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))

    @abstractmethod
    def _compute_output_shape(self, *args) -> torch.Size:
        """Compute output shape for matrix operation"""
        pass

    def _get_output_dtype(self, *args):
        """Get output dtype for matrix operation"""
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if tensors:
            return tensors[0].dtype
        return torch.float32


class ComparisonOpImplementation(OperationImplementation):
    """Provide base behavior for comparison operation implementations."""

    def _handle_empty_tensor(self, input1, input2, *, out=None) -> ExecutionResult:
        """Comparison ops always return bool dtype for empty tensors"""
        # Handle scalar comparisons
        shape1 = input1.shape if isinstance(input1, torch.Tensor) else torch.Size([])
        shape2 = input2.shape if isinstance(input2, torch.Tensor) else torch.Size([])

        try:
            output_shape = torch.broadcast_shapes(shape1, shape2)
        except RuntimeError as e:
            return ExecutionResult(success=False, error_msg=str(e))

        # Get device from tensor input
        device = input1.device if isinstance(input1, torch.Tensor) else input2.device

        # Create bool output tensor
        output = (
            io_tensor.empty(output_shape, dtype=torch.bool, device=device) if out is None else out
        )
        return ExecutionResult(success=True, output=output)


class CompilableOpImpl(OperationImplementation):
    """Mark an operation implementation as compilable."""

    pass


# Helper function to safely get dispatch keys
def _get_dispatch_keys(tensor):
    try:
        if hasattr(torch._C, "_dispatch_keys"):
            return str(torch._C._dispatch_keys(tensor))
    except Exception:
        pass
    return "unavailable"


def _format_tensor_info(tensor: torch.Tensor) -> str:
    """Format tensor information string."""
    try:
        dispatch_keys = _get_dispatch_keys(tensor)
    except Exception:
        dispatch_keys = "unavailable"

    return (
        f"Tensor(shape={tensor.shape}, dtype={tensor.dtype}, "
        f"device={tensor.device}, dispatch keys: {dispatch_keys})"
    )


def _format_value_info(value, prefix: str = "") -> str:
    """Format value information string with optional prefix."""
    if isinstance(value, torch.Tensor):
        return f"{prefix}{_format_tensor_info(value)}"
    elif isinstance(value, (list | tuple)):
        items = []
        for i, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                items.append(f"[{i}]: {_format_tensor_info(item)}")
            else:
                items.append(f"[{i}]: {type(item).__name__}={item}")
        return f"{prefix}{type(value).__name__}({', '.join(items)})"
    else:
        return f"{prefix}{type(value).__name__}={value}"


def _build_args_kwargs_info(args: tuple, kwargs: dict) -> tuple[list[str], list[str]]:
    """Build string representations of args and kwargs."""
    args_info = [_format_value_info(arg, f"arg{i}: ") for i, arg in enumerate(args)]
    kwargs_info = [_format_value_info(v, f"{k}=") for k, v in kwargs.items()]
    return args_info, kwargs_info


def _build_detailed_message(
    base_message: str,
    args: tuple,
    kwargs: dict,
    failure_messages: list[str] | None = None,
) -> str:
    """Build a detailed message with args/kwargs information.

    Args:
        base_message: Base error message
        args: Operation arguments
        kwargs: Operation keyword arguments
        failure_messages: Optional list of failure messages from implementations

    Returns:
        Formatted detailed message string
    """
    args_info, kwargs_info = _build_args_kwargs_info(args, kwargs)
    failure_detail = "; ".join(failure_messages) if failure_messages else ""

    message = f"{base_message}. Args: [{', '.join(args_info)}]. Kwargs: [{', '.join(kwargs_info)}]"

    if failure_detail:
        message += f". Details: {failure_detail}"

    return message


def _create_and_raise_detailed_error(
    error_type: type[Exception],
    op_name: str,
    base_message: str,
    args: tuple,
    kwargs: dict,
    failure_messages: list[str] | None = None,
) -> None:
    """Create and raise a detailed error with args/kwargs information.

    Args:
        error_type: Exception type to raise (RuntimeError, NotImplementedError, etc.)
        op_name: Operation name
        base_message: Base error message
        args: Operation arguments
        kwargs: Operation keyword arguments
        failure_messages: Optional list of failure messages from implementations
    """
    message = _build_detailed_message(base_message, args, kwargs, failure_messages)
    raise error_type(message)


class Operation(ABC):
    """Coordinate implementations and CPU fallback for one ATen operation.

    Note:
        Implementations are tried in descending priority. Successful Neuron execution
        and CPU fallback are logged once through native logging bindings.
    """

    def __init__(self):
        self._implementations: list[OperationImplementation] = []
        self._setup_implementations()
        # Check for decorator but don't fail if missing (temporary backward compatibility)

        for impl in self._implementations:
            if not hasattr(impl.__class__, "_has_neuron_op"):
                # ToDo(thangakr): Remove the legacy code after all ops are migrated
                # Set defaults for legacy implementations without decorator
                impl.__class__._aten_op_name = impl.__class__.__name__
                impl.__class__._has_neuron_op = False  # Mark as legacy
                # warnings.warn(
                #     f"{impl.__class__.__name__} should be decorated with @neuron_op. "
                #     f"Add @neuron_op('aten::op_name') to the class definition. "
                #     f"This will be required in a future version.",
                #     DeprecationWarning,
                #     stacklevel=2,
                # )
        # Sort by priority (highest first)
        self._implementations.sort(key=lambda x: x.priority, reverse=True)

    @abstractmethod
    def _setup_implementations(self):
        """Register all implementations for this operation"""
        pass

    @property
    def is_inplace(self) -> bool:
        """Check whether the operation is in-place from its schema or name.

        Returns:
            bool: ``True`` when the schema marks the first argument writable or the
                fallback name heuristic identifies an in-place operation.
        """
        return is_inplace_op(getattr(self, "_actual_aten_name", self.op_name))

    @property
    @abstractmethod
    def op_name(self) -> str:
        """Return the ATen operation name.

        Returns:
            str: The operation name, such as ``contiguous`` or ``add.Tensor``.
        """
        pass

    def _get_expected_output_shape(self, *args, **kwargs) -> torch.Size | None:
        """
        Get the expected output shape for this operation.

        Override this method in subclasses to enable automatic output tensor resizing.
        This is called before execution to handle PyTorch's resize_output behavior.

        Args:
            *args: Positional arguments to the operation
            **kwargs: Keyword arguments to the operation (excluding 'out')

        Returns:
            Expected output shape or None if not applicable
        """
        return None

    @staticmethod
    def _resize_output_if_needed(
        out: torch.Tensor | None, expected_shape: tuple | torch.Size
    ) -> None:
        """
        Resize output tensor if it doesn't match expected shape, matching PyTorch behavior.

        This implements PyTorch's resize_output behavior which:
        1. Issues a deprecation warning if resizing a non-empty tensor
        2. Resizes the tensor to the expected shape

        Args:
            out: The output tensor to check and possibly resize
            expected_shape: The expected shape for the output tensor
        """
        if out is None:
            return

        if out.shape != expected_shape:
            out.resize_(expected_shape)

    def _get_aten(self):
        try:
            aten_op = getattr(torch.ops.aten, self.op_name)
        except Exception:
            aten_op = None
        return aten_op

    def _unhandled_cpu_fallback(self, *args, **kwargs):
        """
        Fallback to CPU for all auto registered aten ops which could not be handled.

        This method transfers tensors to CPU, executes the operation there, and then
        transfers the results back to the Neuron device.

        Args:
            *args: Variable positional arguments to be passed to the ATen operation.
                  Tensor arguments will be transferred to CPU before execution.
            **kwargs: Variable keyword arguments to be passed to the ATen operation.
                     Tensor values will be transferred to CPU before execution.

        Returns:
            ExecutionResult: An object containing:
                - success (bool): Whether the operation succeeded
                - output (Any): The operation result transferred back to Neuron device
                - error_msg (str, optional): Error message if the operation failed

        Raises:
            RuntimeError: If CPU fallback execution fails
        """

        aten_op = self._get_aten()
        if aten_op:
            # Transfer input tensors to CPU
            cpu_args = [move_pytree_to_cpu(arg) for arg in args]
            cpu_kwargs = {k: move_pytree_to_cpu(v) for k, v in kwargs.items()}

            # Ensure factory ops and others that accept a 'device' kwarg fall back to CPU
            if "device" in cpu_kwargs:
                dev = cpu_kwargs["device"]
                if isinstance(dev, torch.device):
                    if dev.type != "cpu":
                        cpu_kwargs["device"] = torch.device("cpu")
                else:
                    # Accept strings like 'neuron:0'
                    if str(dev).startswith("neuron"):
                        cpu_kwargs["device"] = "cpu"

            if self.is_inplace and "out" in cpu_kwargs:
                cpu_kwargs.pop("out")

            # Run operation on CPU
            result = aten_op(*cpu_args, **cpu_kwargs)

            # Transfer result back to Neuron without triggering .to() on Neuron.
            def _to_neuron_without_to(t):
                from torch_neuronx.python_ops.cast_policy import copy_cpu_to_neuron

                if isinstance(t, torch.Tensor):
                    if t.device.type == "neuron":
                        return t
                    # Copy CPU tensor to Neuron, preserving dtype
                    return copy_cpu_to_neuron(t, torch.device("neuron"), t.dtype)
                elif isinstance(t, tuple):
                    return tuple(_to_neuron_without_to(x) for x in t)
                elif isinstance(t, list):
                    return type(t)(_to_neuron_without_to(x) for x in t)
                elif isinstance(t, dict):
                    return {k: _to_neuron_without_to(v) for k, v in t.items()}
                else:
                    return t

            result = _to_neuron_without_to(result)

            # If an output tensor was provided, resize if needed and copy the result to it
            if "out" in kwargs and kwargs["out"] is not None and isinstance(result, torch.Tensor):
                out_tensor = kwargs["out"]
                # Ensure output tensor has correct shape before copying
                self._resize_output_if_needed(out_tensor, result.shape)
                if not out_tensor.is_contiguous():
                    raise RuntimeError(
                        "CPU fallback failed. Cannot copy result to output tensor "
                        "which is not contiguous"
                    )

                out_tensor.copy_(result)
                return ExecutionResult(success=True, output=out_tensor)

            return ExecutionResult(success=True, output=result)

        return ExecutionResult(
            success=False, error_msg=f"No ATen operation found for {self.op_name}"
        )

    @torch._dynamo.disable
    def __call__(self, *args, **kwargs):
        """Execute the operation by trying implementations in order"""
        # Handle output tensor resizing if needed (PyTorch compatibility)
        out = kwargs.get("out")
        if out is not None:
            # Get expected shape without the 'out' parameter
            kwargs_without_out = {k: v for k, v in kwargs.items() if k != "out"}
            expected_shape = self._get_expected_output_shape(*args, **kwargs_without_out)
            if expected_shape is not None:
                self._resize_output_if_needed(out, expected_shape)

        # Create CPU fallback context with original inputs (args and kwargs)
        # This is created once here before any implementation can modify inputs
        cpu_fallback_context = {"original_inputs": list(args), "original_kwargs": dict(kwargs)}
        result = None
        first_failure_message = None
        fallback_only_for_unimplemented = (
            os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1"
        )
        neuron_impl_failed = False
        last_failed_impl = "unknown"
        # IMPORTANT: When C++ dispatch is enabled, the dispatch cache key depends
        # only on op name, tensor shapes, dtypes, strides, storage offsets, and
        # static scalar args.  The key does NOT encode which implementation was
        # selected by can_handle().  This means the FIRST impl whose can_handle()
        # returns True will be lowered and cached — all subsequent calls with the
        # same cache key will reuse that lowering, even if a different impl's
        # can_handle() would also return True.
        #
        # Correctness requirement: for a given cache key (same shapes/dtypes/strides),
        # all implementations that return can_handle()=True MUST produce equivalent
        # HLO.  If two impls have different can_handle() criteria that go beyond
        # shapes/dtypes/strides (e.g., checking tensor values or runtime state),
        # they risk cache collisions and incorrect lowering reuse.
        for impl in self._implementations:
            impl_args, impl_kwargs = impl.translate_args(*args, **kwargs)
            can_handle_result = impl.can_handle(*impl_args, **impl_kwargs)
            if isinstance(can_handle_result, tuple):
                can_handle_impl, can_handle_error_msg = can_handle_result
            else:
                can_handle_impl, can_handle_error_msg = can_handle_result, None

            if can_handle_impl:
                # Store context directly on kernel for access
                if hasattr(impl, "kernel") and impl.kernel is not None:
                    impl.kernel._cpu_fallback_context = cpu_fallback_context
                try:
                    result = impl.execute(*impl_args, **impl_kwargs)
                finally:
                    # Clear fallback context to avoid holding tensor references
                    # in long-lived cached kernel objects
                    if hasattr(impl, "kernel") and impl.kernel is not None:
                        impl.kernel._cpu_fallback_context = None
                if result.success:
                    # Log that this operation was executed on Neuron
                    import torch_neuronx

                    # Log the op for all impl other than XLA/c10d kernels
                    # or NEURON_LAUNCH_BLOCKING=1
                    # TODO - remove this condition once all ops are onboarded to async pipeline
                    if _needs_python_side_logging(impl, *args):
                        torch_neuronx._C._log_executed_op(self.op_name)
                    return result.output
                # Log failure and continue to next implementation - debug print once only
                debug_msg = (
                    f"Neuron implementation {impl.__class__.__name__} failed: {result.error_msg}"
                )
                if debug_msg not in _debug_messages_printed:
                    logging.debug(debug_msg)
                    _debug_messages_printed.add(debug_msg)
                neuron_impl_failed = True
                last_failed_impl = impl.__class__.__name__
                if first_failure_message is None:
                    if result.error_msg:
                        first_failure_message = f"{impl.__class__.__name__}: {result.error_msg}"
                    else:
                        first_failure_message = (
                            f"{impl.__class__.__name__}: Unknown error during Neuron execution"
                        )
            elif can_handle_error_msg:
                # Implementation can't handle - add the specific error reason
                if first_failure_message is None:
                    first_failure_message = f"{impl.__class__.__name__}: {can_handle_error_msg}"

        # Synchronize since we had and error and might have to wrap up things.
        stream = torch.neuron.current_stream()
        stream.synchronize()

        # Build debug message before CPU fallback
        base_debug_msg = (
            f"No implementation could handle operation {self.op_name}. Falling back on CPU"
        )
        debug_msg = _build_detailed_message(
            base_debug_msg, args, kwargs, [first_failure_message] if first_failure_message else None
        )

        if debug_msg not in _debug_messages_printed:
            if not fallback_only_for_unimplemented:
                logging.debug(debug_msg)
            else:
                logging.error(debug_msg)
            _debug_messages_printed.add(debug_msg)

        # No implementation could handle the operation
        if fallback_only_for_unimplemented and neuron_impl_failed:
            base_msg = (
                "CPU fallback disabled because "
                "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. "
                f"Operation {self.op_name} failed to execute on Neuron "
                f"(last implementation: {last_failed_impl}). "
                f"Details: {first_failure_message if first_failure_message else 'Unknown error'}"
            )
            _create_and_raise_detailed_error(
                RuntimeError, self.op_name, base_msg, args, kwargs, None
            )
        elif fallback_only_for_unimplemented:
            base_msg = (
                "CPU fallback disabled because "
                "TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. "
                f"Operation {self.op_name} failed to be handled by any Neuron implementation "
                f"(last implementation: {last_failed_impl}). "
                f"Details: {can_handle_error_msg if can_handle_error_msg else 'Unknown error'}"
            )
            _create_and_raise_detailed_error(
                RuntimeError, self.op_name, base_msg, args, kwargs, None
            )

        if self._get_aten() is not None:
            result = self._unhandled_cpu_fallback(*args, **kwargs)

        if result is None:
            base_msg = f"Operation {self.op_name} is not implemented for Neuron"
            _create_and_raise_detailed_error(
                NotImplementedError,
                self.op_name,
                base_msg,
                args,
                kwargs,
                [first_failure_message] if first_failure_message else None,
            )

        if result.success:
            # Log that this operation was offloaded to CPU
            import torch_neuronx

            # Log the operation name
            try:
                torch_neuronx._C._log_offloaded_op(self._actual_aten_name)
            except AttributeError:
                # No aten name found fallback to op_name
                torch_neuronx._C._log_offloaded_op(self.op_name)

            return result.output

        # Raise error with detailed message about the fallback failure
        base_msg = (
            f"Operation {self.op_name} is not implemented for Neuron and CPU fallback "
            f"failed with error: {result.error_msg}"
        )
        _create_and_raise_detailed_error(
            NotImplementedError,
            self.op_name,
            base_msg,
            args,
            kwargs,
            [first_failure_message] if first_failure_message else None,
        )


# Category-specific Operation base classes that provide default _get_expected_output_shape


class UnaryOperation(Operation):
    """Provide unary operations with default output-shape inference.

    Note:
        The inferred output shape equals the input shape.
    """

    def _get_expected_output_shape(self, input: torch.Tensor, **kwargs) -> torch.Size | None:
        """For unary operations, output shape matches input shape"""
        return input.shape if isinstance(input, torch.Tensor) else None


class BinaryOperation(Operation):
    """Provide binary operations with default output-shape inference.

    Note:
        Output shapes follow PyTorch broadcasting rules.
    """

    def _get_expected_output_shape(self, input, other=None, **kwargs) -> torch.Size | None:
        """For binary operations, output shape follows broadcasting rules"""
        # Shouldn't really happen
        if other is None and len(kwargs) == 0:
            return None

        # Handle the standard (input, other) pattern used by PyTorch ops
        if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
            shape1 = input.shape if isinstance(input, torch.Tensor) else torch.Size([])
            shape2 = other.shape if isinstance(other, torch.Tensor) else torch.Size([])
            try:
                return torch.broadcast_shapes(shape1, shape2)
            except RuntimeError:
                # If shapes can't be broadcast, let validation fail later
                return None
        return None


class ReductionOperation(Operation):
    """Provide reduction operations with default output-shape inference.

    Note:
        Output shapes are derived from the reduction dimensions and ``keepdim``.
    """

    def _get_expected_output_shape(
        self, input: torch.Tensor, dim=None, keepdim=False, **kwargs
    ) -> torch.Size | None:
        """Compute output shape after reduction"""
        if not isinstance(input, torch.Tensor):
            return None

        if dim is None:
            # Reduce all dimensions
            return torch.Size([]) if not keepdim else torch.Size([1] * input.dim())

        # Handle single dimension or tuple of dimensions
        if isinstance(dim, int):
            dims = [dim]
        elif isinstance(dim, list | tuple):
            dims = list(dim)
        else:
            return None

        # Normalize negative dimensions
        dims = [d if d >= 0 else input.dim() + d for d in dims]

        # Build output shape
        output_shape = []
        for i, size in enumerate(input.shape):
            if i in dims:
                if keepdim:
                    output_shape.append(1)
            else:
                output_shape.append(size)

        return torch.Size(output_shape)


class ComparisonOperation(Operation):
    """Provide comparison operations with default output-shape inference.

    Note:
        Output shapes follow the same broadcasting rules as binary operations.
    """

    def _get_expected_output_shape(self, input, other=None, **kwargs) -> torch.Size | None:
        """For comparison operations, output shape follows broadcasting rules"""
        # Handle both (input1, input2) and (input, other) parameter patterns
        if other is None and len(kwargs) == 0:
            return None

        # Handle the standard (input, other) pattern used by PyTorch ops
        if isinstance(input, torch.Tensor) or isinstance(other, torch.Tensor):
            shape1 = input.shape if isinstance(input, torch.Tensor) else torch.Size([])
            shape2 = other.shape if isinstance(other, torch.Tensor) else torch.Size([])
            try:
                return torch.broadcast_shapes(shape1, shape2)
            except RuntimeError:
                # If shapes can't be broadcast, let validation fail later
                return None
        return None


class AttentionOpImpl(OperationImplementation):
    """Provide common validation for attention operation implementations."""

    def can_handle(
        self,
        query: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        attn_bias: "torch.Tensor | None" = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: "float | None" = None,
    ) -> bool:
        """Check whether common attention inputs satisfy Neuron constraints.

        Args:
            query (torch.Tensor): Query tensor in batch-head-sequence-feature layout.
            key (torch.Tensor): Key tensor in batch-head-sequence-feature layout.
            value (torch.Tensor): Value tensor in batch-head-sequence-feature layout.
            attn_bias (torch.Tensor | None, optional): Attention bias. Defaults to ``None``.
            dropout_p (float, optional): Dropout probability. Defaults to ``0.0``.
            is_causal (bool, optional): Whether attention is causal. Defaults to ``False``.
            return_debug_mask (bool, optional): Debug-mask request. Defaults to ``False``.
            scale (float | None, optional): Attention scale. Defaults to ``None``.

        Returns:
            bool: ``True`` when the tensor inputs and debug-mask request are supported.

        Raises:
            RuntimeError: If query heads are not divisible by key/value heads.

        Note:
            Validation ignores ``attn_bias``, ``dropout_p``, ``is_causal``, and ``scale``.
        """
        # Must be on Neuron device
        if (
            query.device.type != "neuron"
            or key.device.type != "neuron"
            or value.device.type != "neuron"
        ):
            return False

        # Check that all tensors have the same dtype
        if not (query.dtype == key.dtype == value.dtype):
            return False

        # Only support float32, float16, bfloat16
        if query.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            return False

        # Check shapes - must be 4D tensors
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return False

        # Extract dimensions
        batch_size, q_heads, seq_len_q, embed_dim = query.shape
        batch_k, kv_heads, seq_len_k, embed_k = key.shape
        batch_v, v_heads, seq_len_v, embed_v = value.shape

        # Batch size must match
        if batch_size != batch_k or batch_size != batch_v:
            return False

        if seq_len_k != seq_len_v or kv_heads != v_heads:
            return False

        # Embedding dimension must match between Q and K
        if embed_dim != embed_k:
            return False

        # query heads must be divisible by KV heads for GQA and MHA
        if q_heads % kv_heads != 0:
            raise RuntimeError(
                "Number of heads in key and value must divide the number of heads in query"
            )

        # Debug mask not supported
        return not return_debug_mask

    def _check_and_handle_empty(self, *args, **kwargs) -> "ExecutionResult | None":
        """Check for empty tensors and reject them"""

        if len(args) >= 3:
            tensors = [args[0], args[1], args[2]]
            if not all(isinstance(t, torch.Tensor) and t.numel() > 0 for t in tensors):
                return ExecutionResult(
                    success=False,
                    error_msg="Expected non-empty torch.Tensor arguments for "
                    "query, key, and value.",
                )
        return None
