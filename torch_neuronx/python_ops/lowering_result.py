"""Structured result from Python lowering, consumed by C++ dispatcher.

LoweringResult is defined in C++ (LoweringResult.h) and bound via pybind11.
This module re-exports it so existing imports continue to work:
    from torch_neuronx.python_ops.lowering_result import LoweringResult

InputSpec and KwInputSpec are Python-only dataclasses used within the
lowering pipeline but not consumed by C++ dispatch.
"""

from dataclasses import dataclass

from torch_neuronx._C import LoweringResult  # noqa: F401 — re-export


@dataclass
class InputSpec:
    """Describe how an IR input derives from a positional PyTorch argument.

    Args:
        source_arg_index (int): Position in the raw positional arguments.
        source_type (str): Either ``"tensor"`` or ``"tensor_list"``. Defaults to
            ``"tensor"``.
        elem_index (int): Tensor-list element to select. Defaults to ``0``.
        target_dtype (torch.dtype, optional): IR dtype to cast to, or ``None`` when no
            cast is needed. Defaults to ``None``.

    Note:
        This specification is used only by Python lowering and is not parsed by C++ dispatch.
    """

    source_arg_index: int
    source_type: str = "tensor"
    elem_index: int = 0
    target_dtype: object = None


@dataclass
class KwInputSpec:
    """Describe how an IR input derives from a PyTorch keyword argument.

    Args:
        kwarg_name (str): Name of the raw keyword argument, such as ``"out"``.
        source_type (str): Either ``"tensor"`` or ``"tensor_list"``. Defaults to
            ``"tensor"``.
        elem_index (int): Tensor-list element to select. Defaults to ``0``.
        target_dtype (torch.dtype, optional): IR dtype to cast to, or ``None`` when no
            cast is needed. Defaults to ``None``.

    Note:
        This specification is used only by Python lowering and is not parsed by C++ dispatch.
    """

    kwarg_name: str
    source_type: str = "tensor"
    elem_index: int = 0
    target_dtype: object = None
