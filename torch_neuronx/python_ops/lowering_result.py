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
    """Describes how one IR input is derived from a raw PyTorch positional arg.

    Used within the Python lowering pipeline only (not parsed by C++).

    Attributes:
        source_arg_index: Position in the raw PyTorch positional args.
        source_type: How to interpret the raw arg:
            "tensor"        — arg is already a Tensor
            "scalar_int"    — arg is a Python int, convert to tensor
            "scalar_float"  — arg is a Python float, convert to tensor
            "tensor_list"   — arg is a TensorList, elem_index selects which element
        elem_index: For tensor_list source_type, which element in the list.
        target_dtype: torch.dtype the IR expects for this input.
            If None, no dtype cast is needed (use as-is).
            If set and differs from the raw tensor's dtype, C++ must cast.
    """

    source_arg_index: int
    source_type: str = "tensor"
    elem_index: int = 0
    target_dtype: object = None


@dataclass
class KwInputSpec:
    """Describes how one IR input is derived from a keyword argument.

    Used within the Python lowering pipeline only (not parsed by C++).

    Attributes:
        kwarg_name: Name of the keyword argument (e.g., "out", "alpha").
        source_type: Same as InputSpec.source_type.
        elem_index: Same as InputSpec.elem_index.
        target_dtype: Same as InputSpec.target_dtype.
    """

    kwarg_name: str
    source_type: str = "tensor"
    elem_index: int = 0
    target_dtype: object = None
