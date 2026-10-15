"""NKI kernel wrapper for scatter_add operation"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .klir_scatter_add import scatter_add


@nki_op("nki_kernels::scatter_add_kernel", mutates_args={})
def scatter_add_kernel(
    input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """Scatter source rows into an input tensor with the NKI kernel.

    Args:
        input (torch.Tensor): Two-dimensional destination tensor modified in place.
        dim (int): Scatter dimension, which must be ``0``.
        index (torch.Tensor): One-dimensional destination row indices.
        src (torch.Tensor): Two-dimensional source rows to add.

    Returns:
        torch.Tensor: The modified ``input`` tensor.

    Raises:
        AssertionError: If shapes, ``dim``, or logical-core sharding are unsupported.

    Note:
        Indices within each 128-row source tile must be unique for correct accumulation.
    """
    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,)
    result = wrap_nki(scatter_add)[grid](input, dim, index, src)
    return result
