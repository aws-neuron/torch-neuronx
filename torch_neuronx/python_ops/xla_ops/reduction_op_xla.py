"""XLA implementation of reduction operations (mean, sum, etc.)."""

import traceback
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.base import ExecutionResult, ReductionOpImplementation

# Type definition for reduction functions that must include dtype parameter
T = TypeVar("T")
ReductionFnType = Callable[[Any, Any, Any, Any], T]


class ReductionXLAImpl(ReductionOpImplementation):
    """Base class for XLA implementation of reduction operations."""

    def __init__(self, op_name: str, reduction_fn: ReductionFnType, identity_value: Any = None):
        """Initialize the reduction operation.

        Args:
            op_name: Name of the reduction operation (e.g., "mean", "sum")
            reduction_fn: JAX function to perform the reduction (e.g., jnp.mean, jnp.sum)
                          Must accept dtype parameter in its signature
            identity_value: The identity value for this reduction (e.g., 0 for sum, 1 for prod).
                           None if no identity exists (e.g., max, min)
        """
        self.op_name = op_name
        self.identity_value = identity_value

        def reduction_computation(input, dim, dtype, keepdims):
            return reduction_fn(input, axis=dim, dtype=dtype, keepdims=keepdims)

        self.kernel = TorchNeuronXLAKernel(reduction_computation, op_name, static_argnums=(1, 2, 3))

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        if len(args) < 1 or len(args) > 3:
            return False

        input_tensor = args[0]

        # Check that dim is either None, int, or tuple/list of ints
        dim = kwargs.get("dim")
        if dim is not None and not (
            isinstance(dim, int)
            or (isinstance(dim, list | tuple) and all(isinstance(d, int) for d in dim))
        ):
            return False

        # Tensor must be on Neuron device
        return input_tensor.device.type == "neuron"

    def _get_identity_value(self) -> Any:
        """Return the identity value for this reduction operation"""
        return self.identity_value

    def _execute_impl(
        self,
        input: torch.Tensor,
        dim=None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
        out=None,
    ) -> ExecutionResult:
        """Execute reduction operation using XLA."""
        try:
            # Use provided output tensor or create a new one
            if out is None:
                # Calculate output shape based on input shape, dim, and keepdim
                output_dtype = dtype if dtype is not None else input.dtype
                if dim is None:
                    # Reduction over all dimensions -> scalar output
                    output = torch.empty((), dtype=output_dtype, device=input.device)
                else:
                    # Convert single dim to tuple for consistent handling
                    dims = (dim,) if isinstance(dim, int) else dim

                    # Calculate output shape
                    output_shape = list(input.shape)
                    if keepdim:
                        # Replace reduced dimensions with 1
                        for d in dims:
                            # Handle negative indices
                            actual_dim = d if d >= 0 else d + len(output_shape)
                            output_shape[actual_dim] = 1
                    else:
                        # Remove reduced dimensions (starting from highest dim
                        # to avoid index shifting).
                        for d in sorted(dims, reverse=True):
                            # Handle negative indices
                            actual_dim = d if d >= 0 else d + len(output_shape)
                            output_shape.pop(actual_dim)

                    # Create output tensor with calculated shape and dtype
                    output = torch.empty(output_shape, dtype=output_dtype, device=input.device)

            else:
                output = out

            # Handle dim parameter - convert to None or tuple for XLA
            if dim is not None:
                if isinstance(dim, int):
                    dim = (dim,)
                elif isinstance(dim, list | tuple):
                    # Make dim a tuple for hashing
                    dim = tuple(dim) if len(dim) > 0 else None
                # Note: if dim is already None, it stays None

            if dtype is None:
                dtype = input.dtype

            self.kernel(input, dim, dtype, keepdim, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            # Add traceback for debugging
            # TODO(apoorvgu): create a debug flag to place this.
            print(traceback.format_exc())
            return ExecutionResult(success=False, error_msg=str(e))
