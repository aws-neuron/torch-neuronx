import logging

import jax.numpy as jnp
import torch

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


# @neuron_op("aten::atleast_2d.Sequence")
class Atleasts2dXLAImpl(OperationImplementation):
    """atleast_2d_seq implementation using XLA"""

    def __init__(self):
        """Initialize the atleast_2d_seq kernel with JAX computation"""
        super().__init__()

        def atleast2d_sequence(*inputs):
            """JAX computation for atleast2d"""
            return jnp.atleast_2d(*inputs)

        self.kernel = TorchNeuronXLAKernel(atleast2d_sequence, "atleast_2d.Sequence")

    def _execute_impl(self, inputs: list[torch.Tensor]) -> ExecutionResult:
        """Execute the atleast_2d_seq operation - only called for non-empty tensors

        Args:
            inputs: List of tensors to perform atleast_2d on individually
        Returns:
            ExecutionResult with the atleast2d result
        """
        try:
            non_empty_indices = []
            non_empty_tensors = []

            for i, tensor in enumerate(inputs):
                if tensor.numel() != 0:
                    non_empty_indices.append(i)
                    non_empty_tensors.append(tensor)

            non_empty_results = self.kernel(*non_empty_tensors) if non_empty_tensors else ()

            result = [None] * len(inputs)

            for result_idx, original_idx in enumerate(non_empty_indices):
                result[original_idx] = non_empty_results[result_idx]

            for i, tensor in enumerate(inputs):
                if tensor.shape == (0,):
                    result[i] = torch.empty((1, 0), dtype=tensor.dtype, device=tensor.device)

            return ExecutionResult(success=True, output=tuple(result))

        except Exception as e:
            logger.error(f"Failed to execute atleast_2d: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
