"""Scatter add operation with NKI, MLIR and XLA implementations"""

import torch

from .auto_registration import neuron_op
from .base import ExecutionResult, Operation
from .torch_mlir.op_impl import TorchMlirOpImpl
from .torch_mlir.ops.scatter_add import (
    ScatterAddInplaceMLIRImpl,
    ScatterAddMLIRImpl,
    ScatterAddOutMLIRImpl,
)


def _scatter_add_nki_traceable(
    input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """Traceable version of NKI scatter_add for make_fx tracing."""
    if index.dtype == torch.int64:
        index = index.to(torch.int32)
    index_1d = index[:, 0]
    from .nki_kernels.scatter_add import scatter_add_kernel

    return scatter_add_kernel(input, dim, index_1d, src)


def _can_handle_scatter_add(input, dim, index, src) -> bool:
    """Shared can_handle logic for scatter_add and scatter_add_."""
    if input.ndim != 2 or src.ndim != 2:
        return False
    if dim != 0:
        return False
    if input.device.type != "neuron":
        return False
    if input.dtype not in [torch.float32, torch.bfloat16, torch.float16]:
        return False
    if index.ndim != 2:
        return False
    return index.stride(1) == 0


@neuron_op("aten::scatter_add", priority=60)
class ScatterAddNKIImpl(TorchMlirOpImpl):
    """Implement out-of-place scatter-add with an NKI kernel."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::scatter_add",
            torch_fn=_scatter_add_nki_traceable,
            static_argnums=(1,),
        )

    def can_handle(self, input, dim, index, src) -> bool:
        """Check whether the NKI scatter-add kernel supports the inputs.

        Args:
            input (torch.Tensor): Destination values to which updates are added.
            dim (int): Dimension along which to scatter updates.
            index (torch.Tensor): Indices that select destination elements.
            src (torch.Tensor): Values to add at the selected indices.

        Returns:
            bool: ``True`` when inputs are two-dimensional, ``dim`` is zero, ``input``
                is a supported Neuron floating dtype, and ``index.stride(1)`` is zero.
        """
        return _can_handle_scatter_add(input, dim, index, src)

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute out-of-place scatter-add with the NKI kernel.

        Args:
            *args (Any): Positional ATen arguments in ``input``, ``dim``, ``index``, ``src`` order.
            **kwargs (Any): Ignored; required arguments must be positional.

        Returns:
            ExecutionResult: Successful kernel output or a captured error message.
        """
        try:
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.scatter_add import scatter_add_kernel

            input, dim, index, src = args[0], args[1], args[2], args[3]
            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = scatter_add_kernel(input, dim, index_1d, src)
            return ExecutionResult(success=True, output=result)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


@neuron_op("aten::scatter_add_", priority=60)
class ScatterAddInplaceNKIImpl(TorchMlirOpImpl):
    """Implement in-place scatter-add with an NKI kernel."""

    def __init__(self):
        super().__init__(
            aten_op_name="aten::scatter_add_",
            torch_fn=_scatter_add_nki_traceable,
            static_argnums=(1,),
        )

    def can_handle(self, input, dim, index, src) -> bool:
        """Check whether the in-place NKI scatter-add kernel supports the inputs.

        Args:
            input (torch.Tensor): Tensor to update in place.
            dim (int): Dimension along which to scatter updates.
            index (torch.Tensor): Indices that select destination elements.
            src (torch.Tensor): Values to add at the selected indices.

        Returns:
            bool: ``True`` when inputs are two-dimensional, ``dim`` is zero, ``input``
                is a supported Neuron floating dtype, and ``index.stride(1)`` is zero.
        """
        return _can_handle_scatter_add(input, dim, index, src)

    def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute in-place scatter-add with the NKI kernel.

        Args:
            *args (Any): Positional ATen arguments in ``input``, ``dim``, ``index``, ``src`` order.
            **kwargs (Any): Ignored; required arguments must be positional.

        Returns:
            ExecutionResult: Success with the mutated first input, or a captured error message.
        """
        try:
            from torch_neuronx.utils import convert_for_neuron

            from .nki_kernels.scatter_add import scatter_add_kernel

            input, dim, index, src = args[0], args[1], args[2], args[3]
            index = convert_for_neuron(index)
            index_1d = index[:, 0]
            result = scatter_add_kernel(input, dim, index_1d, src)
            input.copy_(result)
            return ExecutionResult(success=True, output=input)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


class ScatterAddOp(Operation):
    """Dispatch out-of-place scatter-add through its registered implementation."""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the out-of-place scatter-add operation name.

        Returns:
            str: ``"scatter_add"``.
        """
        return "scatter_add"


class ScatterAddInplaceOp(Operation):
    """Dispatch in-place scatter-add through its registered implementation."""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddInplaceMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the in-place scatter-add operation name.

        Returns:
            str: ``"scatter_add_"``.
        """
        return "scatter_add_"


class ScatterAddOutOp(Operation):
    """Dispatch scatter-add with an output tensor."""

    def _setup_implementations(self):
        self._implementations.append(ScatterAddOutMLIRImpl())

    @property
    def op_name(self) -> str:
        """Return the output-variant scatter-add operation name.

        Returns:
            str: ``"scatter_add.out"``.
        """
        return "scatter_add.out"


_scatter_add_op = ScatterAddOp()
_scatter_add_inplace_op = ScatterAddInplaceOp()
_scatter_add_out_op = ScatterAddOutOp()


def scatter_add_neuron(input, dim, index, src):
    """Dispatch out-of-place scatter-add for Neuron tensors.

    Args:
        input (torch.Tensor): Destination values to which updates are added.
        dim (int): Dimension along which to scatter updates.
        index (torch.Tensor): Indices that select destination elements.
        src (torch.Tensor): Values to add at the selected indices.

    Returns:
        torch.Tensor: Tensor containing the accumulated updates.
    """
    return _scatter_add_op(input, dim, index, src)


def scatter_add_inplace_neuron(input, dim, index, src):
    """Dispatch in-place scatter-add for Neuron tensors.

    Args:
        input (torch.Tensor): Tensor to update in place.
        dim (int): Dimension along which to scatter updates.
        index (torch.Tensor): Indices that select destination elements.
        src (torch.Tensor): Values to add at the selected indices.

    Returns:
        torch.Tensor: The mutated ``input`` tensor.
    """
    return _scatter_add_inplace_op(input, dim, index, src)


def scatter_add_out_neuron(input, dim, index, src, *, out=None):
    """Dispatch scatter-add into a supplied output tensor.

    Args:
        input (torch.Tensor): Destination values to which updates are added.
        dim (int): Dimension along which to scatter updates.
        index (torch.Tensor): Indices that select destination elements.
        src (torch.Tensor): Values to add at the selected indices.
        out (torch.Tensor, optional): Tensor written with the result. Defaults to ``None``.

    Returns:
        torch.Tensor: ``out`` after accumulation when supplied; otherwise the produced tensor.
    """
    return _scatter_add_out_op(input, dim, index, src, out=out)
