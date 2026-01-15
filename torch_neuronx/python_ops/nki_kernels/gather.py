"""NKI kernel wrapper for gather operation"""

import neuronxcc.nki.language as nl
import torch
from neuronxcc.nki._pre_prod_kernels.experimental.misc.gather import gather

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

wrapped_gather_nki = wrap_nki(gather)


@nki_op("nki_kernels::gather", mutates_args={})
def gather_kernel(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for NKI gather kernel.

    Args:
        input: Input tensor (2D)
        dim: Dimension to gather from (must be 0)
        index: Index tensor (2D)

    Returns:
        Gathered tensor
    """

    logical_neuron_cores = int(get_logical_neuron_cores())

    result = wrapped_gather_nki[(nl.nc(logical_neuron_cores),)](input=input, dim=dim, index=index)

    return result
