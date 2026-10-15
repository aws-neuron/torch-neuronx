import torch

from torch_neuronx import nki_op, wrap_nki
from torch_neuronx.python_ops.nki_kernels._grouped_mm_2d_2d import grouped_mm_2d_2d
from torch_neuronx.utils import get_logical_neuron_cores


def grouped_mm_2d_2d_nki(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Run aligned 2D-by-2D grouped matrix multiplication through NKI.

    Args:
        a (torch.Tensor): Left operand with shape ``(D1, T)``.
        b (torch.Tensor): Right operand with shape ``(T, D2)``.
        offs (torch.Tensor): Aligned cumulative group end offsets.

    Returns:
        torch.Tensor: Per-group result with shape ``(G, D1, D2)``.

    Raises:
        AssertionError: If contraction sizes, data types, or launch sharding are invalid.
    """
    return wrap_nki(grouped_mm_2d_2d)[(int(get_logical_neuron_cores()),)](a, b, offs)


@nki_op("nki_kernels::grouped_mm_2d_2d")
def grouped_mm_2d_2d_op(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Expose aligned 2D-by-2D grouped multiplication as an NKI op.

    Args:
        a (torch.Tensor): Left operand with shape ``(D1, T)``.
        b (torch.Tensor): Right operand with shape ``(T, D2)``.
        offs (torch.Tensor): Aligned cumulative group end offsets.

    Returns:
        torch.Tensor: Per-group result with shape ``(G, D1, D2)``.

    Raises:
        AssertionError: If contraction sizes, data types, or launch sharding are invalid.
    """
    return grouped_mm_2d_2d_nki(a, b, offs)
