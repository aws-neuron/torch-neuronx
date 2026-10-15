"""NKI kernel wrapper for gather operation"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .klir_gather import gather


@nki_op("nki_kernels::gather", mutates_args={})
def gather_kernel(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Gather rows through the wrapped NKI gather kernel.

    Args:
        input (torch.Tensor): Two-dimensional source tensor.
        dim (int): Gather dimension, which must be ``0``.
        index (torch.Tensor): One-dimensional row indices.

    Returns:
        torch.Tensor: Gathered rows with shape ``(index.numel(), input.shape[1])``.

    Raises:
        AssertionError: If the ranks or dimension are unsupported, or if the index
            length is not divisible by the logical Neuron core count.

    Note:
        Out-of-bounds index values fault in the wrapped ``@nki.jit`` kernel.
    """
    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,)

    result = wrap_nki(gather)[grid](input=input, dim=dim, index=index)
    return result
