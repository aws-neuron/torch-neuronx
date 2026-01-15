"""XLA implementation of gather operation"""

import jax.numpy as jnp
import torch

from torch_neuronx.utils import use_mlir_aten_ops

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation
from .scatter_add_xla import ScatterGatherIndexHelper


class GatherXLAImpl(OperationImplementation):
    """XLA implementation of gather"""

    def __init__(self):
        def compute_gather(input, dim, index):
            """JAX gather implementation"""
            if input.ndim == 0:
                return jnp.broadcast_to(input, index.shape)
            if not all(index.shape):
                return jnp.zeros(index.shape, dtype=input.dtype)
            if dim < 0:
                dim += input.ndim
            input_indexes, _ = ScatterGatherIndexHelper.scatter_index(dim, index)
            return input[input_indexes]

        self.kernel = TorchNeuronXLAKernel(compute_gather, "gather", static_argnums=(1,))

    def can_handle(self, *args, **kwargs) -> bool:
        if use_mlir_aten_ops():
            return False
        return super().can_handle(*args, **kwargs)

    def _handle_64bit_dtype(self, result, original_dtype):
        """Handle 64-bit dtype conversion"""
        from ..cast_policy import copy_cpu_to_neuron, copy_neuron_to_cpu, is_64bit

        if is_64bit(original_dtype):
            cpu_cast = copy_neuron_to_cpu(result, target_dtype=original_dtype)
            result = copy_cpu_to_neuron(cpu_cast, result.device, original_dtype)
        return result

    def _handle_empty_tensor(self, input, dim, index, sparse_grad=False) -> ExecutionResult:
        """Handle empty tensor case for gather"""
        result = torch.zeros(index.shape, dtype=input.dtype, device=input.device)
        return ExecutionResult(success=True, output=result)

    def _execute_impl(self, input, dim, index, sparse_grad=False) -> ExecutionResult:
        """Execute XLA gather"""
        try:
            if sparse_grad:
                raise NotImplementedError("sparse_grad=True is not supported on Neuron")

            result = self.kernel(input, dim, index)
            result = self._handle_64bit_dtype(result, input.dtype)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class GatherOutXLAImpl(GatherXLAImpl):
    """XLA implementation of gather.out"""

    def __init__(self):
        super().__init__()

        # Override kernel with different name
        self.kernel.op_name = "gather.out"

    def _handle_empty_tensor(
        self, input, dim, index, sparse_grad=False, *, out=None
    ) -> ExecutionResult:
        """Handle empty tensor case for gather.out"""
        if out is not None:
            result = out
        else:
            result = torch.zeros(index.shape, dtype=input.dtype, device=input.device)
        return ExecutionResult(success=True, output=result)

    def can_handle(self, *args, **kwargs) -> bool:
        if use_mlir_aten_ops():
            return False
        return super().can_handle(*args, **kwargs)

    def _execute_impl(self, input, dim, index, sparse_grad=False, *, out=None) -> ExecutionResult:
        try:
            if sparse_grad:
                raise NotImplementedError("sparse_grad=True is not supported on Neuron")

            result = self.kernel(input, dim, index, output=out)
            result = self._handle_64bit_dtype(result, input.dtype)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
