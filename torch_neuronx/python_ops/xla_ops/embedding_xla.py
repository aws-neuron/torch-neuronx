"""XLA implementation of embedding op."""

import jax.nn as jnn
import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation


@neuron_op("aten::embedding_dense_backward")
class EmbeddingDenseBackwardXLAImpl(OperationImplementation):
    """XLA implementation of embedding dense backward."""

    def __init__(self):
        self.kernels = {}

    @classmethod
    def scatter_add_scaled(cls, masked_grad, masked_indices, num_weights, scale):
        """
        Perform scatter-add operation with scaling
        """
        # Reshape inputs
        indices_flat = masked_indices.reshape(-1)
        grad_flat = masked_grad.reshape(-1, masked_grad.shape[-1])

        # Create one-hot indices
        indices_oh = jnn.one_hot(indices_flat, num_weights)

        # Apply scaling
        grad_flat = grad_flat * scale[indices_flat][:, None]

        # Perform scatter-add using tensordot
        return jnp.tensordot(indices_oh.astype(grad_flat.dtype), grad_flat, axes=([0], [0]))

    @classmethod
    def embedding_dense_backward_computation(
        cls, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    ):
        if scale_grad_by_freq:
            indices_flat = indices.reshape(-1)
            one_hot = jnn.one_hot(indices_flat, num_weights)
            counts = jnp.sum(one_hot, axis=0)
            # Avoid division by zero
            counts = jnp.maximum(counts, 1)
            scale = 1.0 / counts
        else:
            # Set dtype to prevent type promotion for the output
            scale = jnp.ones(num_weights, dtype=grad_output.dtype)

        # Create mask for non-padding indices
        mask = indices != padding_idx

        # Apply mask and compute gradients
        masked_indices = jnp.where(mask, indices, 0)
        masked_grad = jnp.where(mask[..., None], grad_output, 0.0)

        # Compute gradient using scatter-add
        grad_weight = EmbeddingDenseBackwardXLAImpl.scatter_add_scaled(
            masked_grad, masked_indices, num_weights, scale
        )

        return grad_weight

    def _get_kernel(self, scale_grad_by_freq):
        if f"scale_grad_by_freq{scale_grad_by_freq}" not in self.kernels:
            self.kernels[f"scale_grad_by_freq{scale_grad_by_freq}"] = TorchNeuronXLAKernel(
                EmbeddingDenseBackwardXLAImpl.embedding_dense_backward_computation,
                "embedding_dense_backward",
                static_argnums=(2, 4),
            )
        return self.kernels[f"scale_grad_by_freq{scale_grad_by_freq}"]

    def can_handle(self, *args, **kwargs):
        if not super().can_handle(*args, **kwargs):
            return False

        grad_output = args[0]
        indices = args[1]
        if not (indices.dtype == torch.int32 or indices.dtype == torch.int64):
            return False

        return indices.device.type == "neuron" and grad_output.device.type == "neuron"

    def _execute_impl(
        self,
        grad_output: torch.Tensor,
        indices: torch.Tensor,
        num_weights: int,
        padding_idx: int,
        scale_grad_by_freq: bool,
    ) -> ExecutionResult:
        """Execute embedding dense backward using XLA."""
        try:
            grad_weight = torch.empty(
                (num_weights, grad_output.shape[-1]), dtype=grad_output.dtype, device="neuron"
            )

            # Execute kernel
            self._get_kernel(scale_grad_by_freq)(
                grad_output,
                indices,
                num_weights,
                padding_idx,
                scale_grad_by_freq,
                output=grad_weight,
            )

            return ExecutionResult(success=True, output=grad_weight)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
