"""Jax implementation of torch.index."""

import logging

import torch

from torch_neuronx.kernels.xla_kernel import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import create_auto_operation, neuron_op
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.xla_ops.nonzero_xla import NonzeroXLAImpl

logger = logging.getLogger(__name__)

nonzero_op = create_auto_operation("nonzero", [NonzeroXLAImpl])


def index_checking(tensor, indices):
    for dim, idx in enumerate(indices):
        if idx is None or idx.dtype == torch.bool:
            continue

        # Index dtype check
        if idx.is_floating_point():
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")

        # Bounds checking on PyTorch tensors
        size = tensor.shape[dim]
        if torch.any(idx >= size) or torch.any(idx < -size):
            raise IndexError(
                f"index {torch.max(idx).item()} is out of bounds "
                f"for dimension {dim} with size {size}"
            )


def convert_indices(indices):
    """Convert boolean indices into integer indices."""
    integer_indices = []
    for index in indices:
        if index is None:
            integer_indices.append(index)
        elif index.dtype == torch.bool:
            # torch.nonzero produces torch.int64, calling kernel directly to get torch.int32 output
            integer_indices.extend(nonzero_op(index, as_tuple=True, out_dtype=torch.int32))
        else:
            integer_indices.append(index.to(torch.int32))
    return tuple(integer_indices)


@neuron_op("aten::index.Tensor")
@neuron_op("aten::index.Tensor_out")
class IndexXLAImpl(OperationImplementation):
    """index implementation using XLA"""

    def __init__(self):
        """Initialize the index kernel with JAX computation"""
        super().__init__()

        def index_fn(tensor, indices):
            """JAX computation for nonzero"""
            processed_indices = [slice(None, None, None) if i is None else i for i in indices]
            return tensor[tuple(processed_indices)]

        self.kernel = TorchNeuronXLAKernel(index_fn, "index")

    def can_handle(self, tensor, indices, out=None):
        return tensor.device.type == "neuron"

    def _execute_impl(
        self, tensor: torch.Tensor, indices: list[torch.Tensor], *, out=None
    ) -> ExecutionResult:
        """Execute index kernel and return selected values.

        This op cannot use meta tensor for output shape evaluation for boolean indices
        Because it uses torch.nonzero during eval https://github.com/pytorch/pytorch/blob/3f1824742cac2ffb9a3afd90953c492c6c7f2f50/torch/_meta_registrations.py#L3369
        """
        try:
            index_checking(tensor, indices)
            integer_indices = convert_indices(indices)
            result = self.kernel(tensor, integer_indices)
            return ExecutionResult(success=True, output=result)

        except Exception as e:
            logger.error(f"Failed to execute index: {e}")
            return ExecutionResult(success=False, error_msg=str(e))
