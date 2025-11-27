"""JAX implementation of torch.masked_select."""

import logging

import jax.numpy as jnp
import torch

from torch_neuronx.kernels.xla_kernel import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.xla_ops.nonzero_xla import safe_nonzero

logger = logging.getLogger(__name__)


@neuron_op("aten::masked_select")
@neuron_op("aten::masked_select.out")
@neuron_op("aten::masked_select.Tensor_out")
class MaskedSelectXLAImpl(OperationImplementation):
    """masked_select implementation using XLA"""

    def __init__(self):
        """Initialize the masked_select kernel with JAX computation"""
        super().__init__()

        def masked_select_fn(tensor, mask, size):
            """JAX computation for masked_select"""
            broadcast_shape = jnp.broadcast_shapes(tensor.shape, mask.shape)

            if tensor.shape != broadcast_shape:
                tensor = jnp.broadcast_to(tensor, broadcast_shape)
            if mask.shape != broadcast_shape:
                mask = jnp.broadcast_to(mask, broadcast_shape)

            tensor_flat = tensor.flatten()
            mask_flat = mask.flatten()

            true_indices = safe_nonzero(mask_flat, size=size)[0]
            return tensor_flat[true_indices].astype(tensor.dtype)

        self.kernel = TorchNeuronXLAKernel(masked_select_fn, "masked_select", static_argnums=(2,))

    def can_handle(self, tensor, mask, *args, **kwargs):
        return tensor.device.type == "neuron"

    def _execute_impl(self, tensor, mask, *args, out=None) -> ExecutionResult:
        """Execute masked_select kernel and return selected values."""
        try:
            size = mask.sum().item()
            # Multiply the size after broadcast
            broadcasted_shape = torch.broadcast_shapes(tensor.shape, mask.shape)
            size = int(size * broadcasted_shape.numel() / mask.shape.numel())

            if size == 0:
                result = torch.empty(0, device=tensor.device)
            else:
                result = self.kernel(tensor, mask, size)

            if out is not None:
                out.copy_(result)

            return ExecutionResult(success=True, output=result)

        except Exception as e:
            logger.error(f"Failed to execute masked_select: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
