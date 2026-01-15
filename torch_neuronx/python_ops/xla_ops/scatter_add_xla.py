import jax.numpy as jnp
import torch

from torch_neuronx.utils import use_mlir_aten_ops

from ...kernels.xla_kernel import TorchNeuronXLAKernel
from ..base import ExecutionResult, OperationImplementation


class ScatterGatherIndexHelper:
    """Helper class for scatter/gather index operations"""

    @staticmethod
    def scatter_index(dim, index):
        """Helper to generate scatter/gather indices - shared by scatter_add and gather"""

        index_shape = list(index.shape)
        input_indexes = []
        source_indexes = []
        if dim < 0:
            dim += len(index_shape)
        for i in range(len(index_shape)):
            source_indexes.append(slice(0, index_shape[i]))
            if i == dim:
                input_indexes.append(index)
            else:
                target_shape = [1] * len(index_shape)
                target_shape[i] = index_shape[i]
                input_indexes.append(
                    jnp.broadcast_to(jnp.arange(index_shape[i]).reshape(target_shape), index_shape)
                )
        return tuple(input_indexes), tuple(source_indexes)


class ScatterAddXLAImpl(OperationImplementation):
    """XLA implementation of scatter_add (out-of-place)"""

    def __init__(self):
        def compute_scatter_add(input, dim, index, src):
            input_indexes, source_indexes = ScatterGatherIndexHelper.scatter_index(dim, index)
            input = input.at[input_indexes].add(src[source_indexes])
            return input

        self.kernel = TorchNeuronXLAKernel(compute_scatter_add, "scatter_add", static_argnums=(1,))

    def can_handle(self, *args, **kwargs) -> bool:
        """Only handle if MLIR is not enabled."""
        if use_mlir_aten_ops():
            return False
        return super().can_handle(*args, **kwargs)

    def _handle_empty_tensor(self, input, dim, index, src) -> ExecutionResult:
        """Handle empty tensor case for scatter_add (out-of-place)"""
        result = torch.zeros_like(input)
        return ExecutionResult(success=True, output=result)

    def _execute_impl(self, input, dim, index, src) -> ExecutionResult:
        try:
            result = self.kernel(input, dim, index, src)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class ScatterAddInplaceXLAImpl(OperationImplementation):
    """XLA implementation of scatter_add_ (in-place)"""

    def __init__(self):
        def compute_scatter_add(input, dim, index, src):
            input_indexes, source_indexes = ScatterGatherIndexHelper.scatter_index(dim, index)
            input = input.at[input_indexes].add(src[source_indexes])
            return input

        self.kernel = TorchNeuronXLAKernel(compute_scatter_add, "scatter_add_", static_argnums=(1,))

    def can_handle(self, *args, **kwargs) -> bool:
        if use_mlir_aten_ops():
            return False
        return super().can_handle(*args, **kwargs)

    def _handle_empty_tensor(self, input, dim, index, src) -> ExecutionResult:
        """Handle empty tensor case for scatter_add_ (in-place)"""
        return ExecutionResult(success=True, output=input)

    def _execute_impl(self, input, dim, index, src) -> ExecutionResult:
        try:
            self.kernel(input, dim, index, src, output=input)
            return ExecutionResult(success=True, output=input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class ScatterAddOutXLAImpl(OperationImplementation):
    """XLA implementation of scatter_add.out"""

    def __init__(self):
        def compute_scatter_add(input, dim, index, src):
            input_indexes, source_indexes = ScatterGatherIndexHelper.scatter_index(dim, index)
            input = input.at[input_indexes].add(src[source_indexes])
            return input

        self.kernel = TorchNeuronXLAKernel(
            compute_scatter_add, "scatter_add.out", static_argnums=(1,)
        )

    def can_handle(self, *args, **kwargs) -> bool:
        if use_mlir_aten_ops():
            return False
        return super().can_handle(*args, **kwargs)

    def _handle_empty_tensor(self, input, dim, index, src, *, out=None) -> ExecutionResult:
        """Handle empty tensor case for scatter_add.out"""
        if out is not None:
            out.copy_(input)
            result = out
        else:
            result = input.clone()
        return ExecutionResult(success=True, output=result)

    def _execute_impl(self, input, dim, index, src, *, out=None) -> ExecutionResult:
        try:
            out = self.kernel(input, dim, index, src, output=out)
            return ExecutionResult(success=True, output=out)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
