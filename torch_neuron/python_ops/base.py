import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from torch_neuron.utils import move_pytree_to_cpu

from .handlers import BaseEmptyTensorHandler

# Set to track debug messages that have been printed once
_debug_messages_printed = set()


def _should_use_sync_execution(impl, *args):
    """
    Helper function to determine if op should be logged
    or if current CPU fallback should be used for sync
    execution (not async pipeline)
    """
    return (
        not hasattr(impl, "kernel")
        or impl.kernel is None
        or impl.kernel.__class__.__name__ in ["TorchNeuronNKIKernel"]
        or BaseEmptyTensorHandler().check_for_empty(*args)
        or os.environ.get("TORCH_NEURON_SYNC_MODE", "1") == "1"
        or os.environ.get("NEURON_LAUNCH_BLOCKING", "1") == "1"
    )


@dataclass
class ExecutionResult:
    success: bool
    output: Any | None = None
    error_msg: str | None = None


class OperationImplementation(ABC):
    """Base class for all operation implementations"""

    # Execution policy toggles (default: current behavior)
    # - When DISABLE_DTYPE_AUTOCAST is True, skip float64->float32 input pre-cast and
    #   related post-processing of results.
    # - When DISABLE_EMPTY_SHORTCIRCUIT is True, skip empty-tensor short-circuiting.
    DISABLE_DTYPE_AUTOCAST: bool = False
    DISABLE_EMPTY_SHORTCIRCUIT: bool = False

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given arguments

        Default implementation returns True, assuming single implementation
        for the operation. Override this method when multiple implementations
        exist and need to filter based on input arguments.
        """
        return True

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation honoring policy flags.

        By default, this:
          - casts float64 inputs to float32,
          - short-circuits empty tensors,
          - calls implementation, then post-processes outputs for dtype fixes.

        Ops can set class flags to disable autocast and/or empty short-circuiting
        to avoid recursion or to own these behaviors explicitly.
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

    def _cast_float64_to_float32(self, args, kwargs):
        """Cast float64 tensor inputs to float32

        Neuron hardware doesn't support float64, but we can handle float64 inputs
        by converting them to float32. This avoids crashes when ops fallback to CPU
        and produce float64 results that get passed to subsequent Neuron ops.
        """
        # Check if we have float64 inputs
        has_float64_input = any(
            isinstance(arg, torch.Tensor) and arg.dtype == torch.float64 for arg in args
        )

        # Cast args
        new_args = []
        for arg in args:
            if (
                isinstance(arg, torch.Tensor)
                and arg.dtype == torch.float64
                and getattr(arg, "ndim", None) != 0
            ):
                new_args.append(arg.to(torch.float32))
            else:
                new_args.append(arg)

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
            elif (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float64
                and getattr(value, "ndim", None) != 0
            ):
                new_kwargs[key] = value.to(torch.float32)
            else:
                new_kwargs[key] = value

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
        """Priority for ordering implementations (higher = tried first)

        Default priority is 50. Override this method to change priority
        when multiple implementations exist for the same operation.
        """
        return 50


class UnaryOpImplementation(OperationImplementation):
    """Base class for element-wise unary operations (sqrt, neg, gelu, relu, etc.)"""

    def _handle_empty_tensor(self, *args, **kwargs) -> ExecutionResult:
        """Unary ops preserve input shape for empty tensors"""
        # Extract input tensor (first argument) and optional out parameter
        input = args[0] if args else None
        if input is None or not isinstance(input, torch.Tensor):
            return ExecutionResult(success=False, error_msg="No input tensor provided")

        out = kwargs.get("out")
        output = torch.empty_like(input) if out is None else out
        return ExecutionResult(success=True, output=output)

    def _validate_shapes(self, input: torch.Tensor, out=None) -> bool:
        """Validate that output shape matches input shape"""
        if out is not None and out.shape != input.shape:
            raise RuntimeError(f"Output shape {out.shape} doesn't match input {input.shape}")
        return True


class BinaryOpImplementation(OperationImplementation):
    """Base class for element-wise binary operations (add, mul, div, etc.)"""

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
        output = torch.empty(output_shape, dtype=dtype, device=device) if out is None else out
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
    """Base class for reduction operations (sum, mean, max, min, etc.)"""

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
    """Base class for matrix operations (mm, bmm, addmm, etc.)"""

    def _handle_empty_tensor(self, *args, **kwargs) -> ExecutionResult:
        """Matrix ops need careful dimension checking for empty tensors"""
        try:
            output_shape = self._compute_output_shape(*args)
            dtype = self._get_output_dtype(*args)
            device = args[0].device if isinstance(args[0], torch.Tensor) else args[1].device

            out = kwargs.get("out")
            output = torch.empty(output_shape, dtype=dtype, device=device) if out is None else out

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
    """Base class for comparison operations (eq, lt, gt, le, ge, ne, etc.)"""

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
        output = torch.empty(output_shape, dtype=torch.bool, device=device) if out is None else out
        return ExecutionResult(success=True, output=output)


class CompilableOpImpl(OperationImplementation):
    """Base class for compilable operations that extends OperationImplementation."""

    pass


class Operation(ABC):
    """Base class for operations that can have multiple implementations"""

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
        """Check if this is an in-place operation based on the operation name."""
        op_name = self.op_name
        return "_." in op_name or (op_name.endswith("_") and "." not in op_name)

    @property
    @abstractmethod
    def op_name(self) -> str:
        """The ATen operation name (e.g., 'contiguous', 'add.Tensor')"""
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
                from torch_neuron.python_ops.cast_policy import copy_cpu_to_neuron

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
                out_tensor.copy_(result)
                return ExecutionResult(success=True, output=out_tensor)

            return ExecutionResult(success=True, output=result)

        return ExecutionResult(
            success=False, error_msg=f"No ATen operation found for {self.op_name}"
        )

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

        result = None
        failure_messages: list[str] = []
        fallback_only_for_unimplemented = (
            os.environ.get("TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1"
        )
        neuron_impl_failed = False
        last_failed_impl = "unknown"
        for impl in self._implementations:
            if impl.can_handle(*args, **kwargs):
                result = impl.execute(*args, **kwargs)
                if result.success:
                    # Log that this operation was executed on Neuron
                    import torch_neuron

                    # Log the op for all impl other than JAX/XLA/c10d kernels
                    # or NEURON_LAUNCH_BLOCKING=1 or TORCH_NEURON_SYNC_MODE=1
                    # TODO - remove this condition once all ops are onboarded to async pipeline
                    if _should_use_sync_execution(impl, *args):
                        torch_neuron._C._log_executed_op(self.op_name)
                    return result.output
                # Log failure and continue to next implementation - debug print once only
                debug_msg = (
                    f"Neuron implementation {impl.__class__.__name__}failed: {result.error_msg}"
                )
                if debug_msg not in _debug_messages_printed:
                    logging.debug(debug_msg)
                    _debug_messages_printed.add(debug_msg)
                neuron_impl_failed = True
                last_failed_impl = impl.__class__.__name__
                if result.error_msg:
                    failure_messages.append(f"{impl.__class__.__name__}: {result.error_msg}")
                else:
                    failure_messages.append(
                        f"{impl.__class__.__name__}: Unknown error during Neuron execution"
                    )
        torch.neuron.synchronize()
        # All Neuron implementations failed - attempt CPU fallback
        # Build a safe string representation of args for error message
        args_info = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                shape_str = f"shape={arg.shape}"
                dtype_str = f"dtype={arg.dtype}"
                device_str = f"device={arg.device}"
                args_info.append(f"arg{i}: Tensor({shape_str}, {dtype_str}, {device_str})")
            else:
                args_info.append(f"arg{i}: {type(arg).__name__}={arg}")

        kwargs_info = []
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape_str = f"shape={v.shape}"
                dtype_str = f"dtype={v.dtype}"
                device_str = f"device={v.device}"
                kwargs_info.append(f"{k}=Tensor({shape_str}, {dtype_str}, {device_str})")
            else:
                kwargs_info.append(f"{k}={v}")

        # Try to fall back on CPU with a debug message - print once only
        debug_msg = (
            f"No implementation could handle operation {self.op_name}. Falling back on CPU "
            f"Args: [{', '.join(args_info)}]. "
            f"Kwargs: [{', '.join(kwargs_info)}]"
        )
        if debug_msg not in _debug_messages_printed:
            logging.debug(debug_msg)
            _debug_messages_printed.add(debug_msg)

        if fallback_only_for_unimplemented and neuron_impl_failed:
            failure_detail = "; ".join(failure_messages) if failure_messages else "Unknown error"
            raise RuntimeError(
                "CPU fallback disabled because \
                    TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1. "
                f"Operation {self.op_name} failed to execute on Neuron (last implementation: "
                f"{last_failed_impl}). "
                f"Details: {failure_detail}. Args: [{', '.join(args_info)}]. "
                f"Kwargs: [{', '.join(kwargs_info)}]"
            )
        if self._get_aten() is not None:
            result = self._unhandled_cpu_fallback(*args, **kwargs)

        if result is None:
            # Raise error with detailed message about the fallback failure
            raise NotImplementedError(
                f"Operation {self.op_name} is not implemented for Neuron for "
                f"Args: [{', '.join(args_info)}]. "
                f"Kwargs: [{', '.join(kwargs_info)}]"
            )

        if result.success:
            # Log that this operation was offloaded to CPU
            import torch_neuron

            # Log the operation name
            try:
                torch_neuron._C._log_offloaded_op(self._actual_aten_name)
            except AttributeError:
                # No aten name found fallback to op_name
                torch_neuron._C._log_offloaded_op(self.op_name)

            return result.output

        # Raise error with detailed message about the fallback failure
        raise NotImplementedError(
            f"Operation {self.op_name} is not implemented for Neuron and CPU fallback "
            f"failed with error: {result.error_msg}. "
            f"Args: [{', '.join(args_info)}]. "
            f"Kwargs: [{', '.join(kwargs_info)}]"
        )


# Category-specific Operation base classes that provide default _get_expected_output_shape


class UnaryOperation(Operation):
    """Base class for unary operations (sqrt, neg, relu, exp, etc.)

    Provides default output shape computation for unary ops (output shape = input shape).
    """

    def _get_expected_output_shape(self, input: torch.Tensor, **kwargs) -> torch.Size | None:
        """For unary operations, output shape matches input shape"""
        return input.shape if isinstance(input, torch.Tensor) else None


class BinaryOperation(Operation):
    """Base class for binary operations (add, mul, div, pow, etc.)

    Provides default output shape computation using PyTorch broadcasting rules.
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
    """Base class for reduction operations (sum, mean, max, min, etc.)

    Provides default output shape computation based on reduction dimensions.
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
    """Base class for comparison operations (eq, lt, gt, le, ge, ne, etc.)

    Provides default output shape computation using broadcasting rules (same as binary ops).
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
    """Base class for attention operation implementations with common validation"""

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
        """Common validation for attention operations"""

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

        if key.shape != value.shape:
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
