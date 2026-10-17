"""Gather operation with NKI, MLIR and XLA implementations"""

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, Operation
from .torch_mlir.op_impl import TorchMlirOpImpl
from .torch_mlir.ops.gather import GatherMLIRImpl, GatherOutMLIRImpl


def _gather_nki_traceable(
    input: torch.Tensor, dim: int, index: torch.Tensor, sparse_grad: bool = False
) -> torch.Tensor:
    """Traceable version of NKI gather for make_fx tracing."""
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    index_1d = index[:, 0]
    from .nki_kernels.gather import gather_kernel

    return gather_kernel(input, dim, index_1d)


@neuron_op("aten::gather", priority=60)
class GatherNKIImpl(TorchMlirOpImpl):
    """Implement gather with an NKI kernel."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::gather",
            torch_fn=_gather_nki_traceable,
            static_argnums=(1,),
            static_argnames=("sparse_grad",),
        )

    def can_handle(self, input, dim, index, sparse_grad=False) -> bool:
        """Check whether the NKI gather kernel supports the inputs.

        Args:
            input (torch.Tensor): Tensor from which to gather values.
            dim (int): Dimension along which to gather.
            index (torch.Tensor): Indices that select source elements.
            sparse_grad (bool, optional): Whether to request a sparse gradient.
                Defaults to ``False``.

        Returns:
            bool: ``True`` when ``input`` is a two-dimensional Neuron device tensor of
            dtype ``float16``, ``bfloat16``, or ``float32``; ``dim`` is zero;
            ``index`` is two-dimensional with zero stride in dimension one; and
            ``sparse_grad`` is ``False``.
        """
        if input.ndim != 2:
            return False
        if dim != 0:
            return False
        if input.device.type != "neuron":
            return False
        if input.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            return False
        if sparse_grad:
            return False
        if index.ndim != 2:
            return False
        return index.stride(1) == 0

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute gather with the NKI kernel.

        Args:
            *args (Any): Positional ATen arguments in ``input``, ``dim``, ``index`` order.
            **kwargs (Any): Ignored keyword arguments accepted for interface compatibility.

        Returns:
            ExecutionResult: Successful kernel output or a captured error message.
        """
        try:
            from torch_neuronx.python_ops.cast_policy import (
                bounce_64bit,
                copy_cpu_to_neuron,
                copy_neuron_to_cpu,
            )
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.gather import gather_kernel

            input, dim, index = args[0], args[1], args[2]
            original_input_dtype = input.dtype

            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = gather_kernel(input, dim, index_1d)
            if bounce_64bit(original_input_dtype):
                cpu_cast = copy_neuron_to_cpu(result, target_dtype=original_input_dtype)
                result = copy_cpu_to_neuron(cpu_cast, result.device, original_input_dtype)

            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class GatherOp(Operation):
    """Dispatch gather through its registered implementation."""

    def _setup_implementations(self):
        self._implementations.append(GatherMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the gather operation name.

        Returns:
            str: The ``gather`` operation name.
        """
        return "gather"


class GatherOutOp(Operation):
    """Dispatch gather with an output tensor."""

    def _setup_implementations(self):
        self._implementations.append(GatherOutMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the output-variant gather operation name.

        Returns:
            str: The ``gather.out`` operation name.
        """
        return "gather.out"


_gather_op = GatherOp()
_gather_out_op = GatherOutOp()


def gather_neuron(input, dim, index, sparse_grad=False):
    """Dispatch gather for Neuron tensors.

    Args:
        input (torch.Tensor): Tensor from which to gather values.
        dim (int): Dimension along which to gather.
        index (torch.Tensor): Indices that select source elements.
        sparse_grad (bool, optional): Whether to request a sparse gradient.
            Defaults to ``False``.

    Returns:
        torch.Tensor: Tensor containing the gathered values.
    """
    return _gather_op(input, dim, index, sparse_grad=sparse_grad)


def gather_out_neuron(input, dim, index, sparse_grad=False, *, out=None):
    """Dispatch gather into a supplied output tensor.

    Args:
        input (torch.Tensor): Tensor from which to gather values.
        dim (int): Dimension along which to gather.
        index (torch.Tensor): Indices that select source elements.
        sparse_grad (bool, optional): Whether to request a sparse gradient.
            Defaults to ``False``.
        out (torch.Tensor, optional): Tensor that receives the result. Defaults to ``None``.

    Returns:
        torch.Tensor: The output tensor containing the gathered values.
    """
    return _gather_out_op(input, dim, index, sparse_grad=sparse_grad, out=out)
