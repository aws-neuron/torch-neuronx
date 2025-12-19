import logging
from functools import reduce

import jax.numpy as jnp
import torch

from torch_neuronx.python_ops.auto_registration import neuron_op

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


@neuron_op("aten::cat.out", disable_dtype_autocast=True)
class ConcatXLAImpl(OperationImplementation):
    """Concat implementation using XLA"""

    def __init__(self):
        """Initialize the concat kernel with JAX computation"""
        super().__init__()

        def concat_computation(*inputs):
            """JAX computation for concat"""
            *tensors, dim = inputs
            return jnp.concat(tensors, axis=dim)

        self.kernel = TorchNeuronXLAKernel(concat_computation, "concat", static_argnums=(-1,))

    def can_handle(self, *args, **kwargs):
        if not super().can_handle(*args, **kwargs):
            return False

        tensors = args[0]
        return all(t.device.type == "neuron" for t in tensors)

    def _execute_impl(self, tensors: list[torch.Tensor], dim=0, *, out=None) -> ExecutionResult:
        """Execute the concatenation operation - only called for non-empty tensors"""
        try:
            # Determine target dtype
            if out is not None:
                # If out is provided, use its dtype and cast inputs if needed
                target_dtype = out.dtype
                if any(t.dtype != target_dtype for t in tensors):
                    tensors = [
                        t.to(target_dtype) if t.dtype != target_dtype else t for t in tensors
                    ]
            else:
                # Calculate output shape and dtype
                out_shape = list(tensors[0].shape)
                out_shape[dim] = sum(t.shape[dim] for t in tensors)

                target_dtype = reduce(
                    torch.promote_types, (t.dtype for t in tensors[1:]), tensors[0].dtype
                )

                out = torch.empty(
                    out_shape,
                    dtype=target_dtype,
                    requires_grad=any(t.requires_grad for t in tensors),
                    device="neuron",
                )

            # Filter empty tensors
            filtered_tensors = [t for t in tensors if t.numel() > 0]

            if len(filtered_tensors) == 1:
                out.copy_(filtered_tensors[0])
            elif len(filtered_tensors) > 1:
                self.kernel(*filtered_tensors, dim, output=out)

            return ExecutionResult(success=True, output=out)

        except Exception as e:
            logger.error(f"Failed to execute concatenation: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
