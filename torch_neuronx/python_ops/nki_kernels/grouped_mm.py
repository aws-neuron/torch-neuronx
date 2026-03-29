import torch
from neuronxcc.nki._pre_prod_kernels.experimental.gmm import grouped_mm_2d_2d

from torch_neuronx import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores


def grouped_mm_2d_2d_nki(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Grouped matmul via NKI: a (d1, t), b (t, d2), offs (g,) -> (g, d1, d2)"""
    return wrap_nki(grouped_mm_2d_2d)[(int(get_logical_neuron_cores()),)](a, b, offs)


@nki_op("nki_kernels::grouped_mm_2d_2d")
def grouped_mm_2d_2d_op(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Grouped matmul: a (d1, t), b (t, d2), offs (g,) -> (g, d1, d2)"""
    return grouped_mm_2d_2d_nki(a, b, offs)
