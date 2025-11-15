import logging

import torch

from ..base import BinaryOpImplementation

logger = logging.getLogger(__name__)


class ArithmeticXLABase(BinaryOpImplementation):
    """Base op for arithmetic operations (tensor-tensor and tensor-scalar) using JAX"""

    def can_handle(self, *args, **kwargs) -> bool:
        """Check if this implementation can handle the given inputs"""
        if not super().can_handle(*args, **kwargs):
            return False

        inputs = [args[0], args[1]]

        # At least one must be a tensor on Neuron device
        tensors = [inp for inp in inputs if torch.is_tensor(inp)]
        if not tensors:
            return False

        # Check if at least one tensor is on Neuron device
        if not any(t.device.type == "neuron" for t in tensors):
            return False

        # Both arguments must be either tensor or scalar
        for inp in inputs:
            if not (torch.is_tensor(inp) or isinstance(inp, int | float | bool)):
                return False

        # If both are tensors, check shape compatibility
        if len(tensors) == 2:
            try:
                torch.broadcast_shapes(tensors[0].shape, tensors[1].shape)
            except RuntimeError:
                return False

        return True

    def _get_out_tensor(self, tensor_args):
        # Calculate output shape
        tensors = [x for x in tensor_args if isinstance(x, torch.Tensor)]
        if len(tensors) > 1:
            output_shape = torch.broadcast_shapes(tensor_args[0].shape, tensor_args[1].shape)
        else:
            output_shape = tensors[0].shape
            if output_shape is None:
                raise RuntimeError("Failed to get the scalar's shape")

        return torch.empty(
            output_shape,
            dtype=torch.result_type(*tensor_args),
            device="neuron",
            requires_grad=tensors[0].requires_grad,
        )
