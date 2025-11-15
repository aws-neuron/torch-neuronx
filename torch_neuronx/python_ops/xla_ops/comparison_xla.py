"""XLA implementation of comparison operations."""

import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.kernels.type_promotion import promote_binary_op
from torch_neuronx.python_ops.base import ComparisonOpImplementation, ExecutionResult


class ComparisonXLAImpl(ComparisonOpImplementation):
    """Base class for XLA comparison implementations."""

    def __init__(self, comparison_fn, op_name, is_tensor_comparison=False):
        """Initialize with a comparison function.

        Args:
            comparison_fn: JAX comparison function to use
            op_name: Operation name for cache key generation
            is_tensor_comparison: True for tensor-tensor comparisons, False for tensor-scalar
        """
        self.kernel = TorchNeuronXLAKernel(comparison_fn, op_name)
        self.is_tensor_comparison = is_tensor_comparison

    def _execute_impl(self, input: torch.Tensor, other, *, out=None) -> ExecutionResult:
        """Execute comparison operation."""
        try:
            if self.is_tensor_comparison:
                # Handle broadcasting for tensor comparisons
                output_shape = torch.broadcast_shapes(input.shape, other.shape)
                output = (
                    torch.empty(output_shape, dtype=torch.bool, device=input.device)
                    if out is None
                    else out
                )
            else:
                # Scalar comparison - output shape matches input
                output = (
                    torch.empty(input.shape, dtype=torch.bool, device=input.device)
                    if out is None
                    else out
                )

            # Execute kernel
            self.kernel(input, other, output=output)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


# Factory function to create comparison implementations
def create_comparison_impl(jax_fn, op_name, is_tensor_comparison=False):
    """Create a comparison implementation class.

    Args:
        jax_fn: JAX comparison function (e.g., jnp.equal, jnp.less)
        op_name: Operation name for cache key generation
        is_tensor_comparison: True for tensor-tensor comparisons, False for tensor-scalar
    """

    class ComparisonImpl(ComparisonXLAImpl):
        def __init__(self):
            if is_tensor_comparison:

                def comparison_fn(x, y):
                    # Handle type promotion for tensor-tensor comparisons
                    x, y = promote_binary_op(x, y)
                    return jax_fn(x, y)
            else:

                def comparison_fn(x, scalar):
                    # Handle type promotion for tensor-scalar comparisons
                    x, scalar = promote_binary_op(x, scalar)
                    return jax_fn(x, scalar)

            super().__init__(comparison_fn, op_name, is_tensor_comparison)

    return ComparisonImpl


# Scalar comparison implementations
EqScalarXLAImpl = create_comparison_impl(jnp.equal, "eq_scalar", is_tensor_comparison=False)
NeScalarXLAImpl = create_comparison_impl(jnp.not_equal, "ne_scalar", is_tensor_comparison=False)
LtScalarXLAImpl = create_comparison_impl(jnp.less, "lt_scalar", is_tensor_comparison=False)
LeScalarXLAImpl = create_comparison_impl(jnp.less_equal, "le_scalar", is_tensor_comparison=False)
GtScalarXLAImpl = create_comparison_impl(jnp.greater, "gt_scalar", is_tensor_comparison=False)
GeScalarXLAImpl = create_comparison_impl(jnp.greater_equal, "ge_scalar", is_tensor_comparison=False)

# Tensor comparison implementations
EqTensorXLAImpl = create_comparison_impl(jnp.equal, "eq_tensor", is_tensor_comparison=True)
NeTensorXLAImpl = create_comparison_impl(jnp.not_equal, "ne_tensor", is_tensor_comparison=True)
LtTensorXLAImpl = create_comparison_impl(jnp.less, "lt_tensor", is_tensor_comparison=True)
LeTensorXLAImpl = create_comparison_impl(jnp.less_equal, "le_tensor", is_tensor_comparison=True)
GtTensorXLAImpl = create_comparison_impl(jnp.greater, "gt_tensor", is_tensor_comparison=True)
GeTensorXLAImpl = create_comparison_impl(jnp.greater_equal, "ge_tensor", is_tensor_comparison=True)
