import logging

import jax.numpy as jnp
import torch

from torch_neuron.python_ops.auto_registration import neuron_op

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation

logger = logging.getLogger(__name__)


@neuron_op("aten::cat.out")
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
        """Execute the concatenation operation - only called for non-empty tensors

        Args:
            tensors: List of tensors to be concatenated
            dim: axis along which the tensors need to be concatenated
        Returns:
            ExecutionResult with the concatenation result
        """
        try:
            if out is None:
                out_shape = tensors[0].shape
                out_shape[dim] *= len(tensors)
                out = torch.empty(
                    out_shape,
                    dtype=torch.result_type(*tensors),
                    requires_grad=tensors[0].requires_grad,
                    device="neuron",
                )

            # handle empty tensors as a special case.
            # torch.cat will ignore the empty tensor, while jnp.concatenate
            # will error if the dims > 0.
            filtered_tensors = [t for t in tensors if not (t.ndim == 1 and t.shape[0] == 0)]
            if filtered_tensors:
                # Execute the kernel
                self.kernel(*filtered_tensors, dim, output=out)

            return ExecutionResult(success=True, output=out)

        except Exception as e:
            logger.error(f"Failed to execute multiplication: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
