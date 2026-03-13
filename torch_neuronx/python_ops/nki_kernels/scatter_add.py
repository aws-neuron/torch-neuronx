"""NKI kernel wrapper for scatter_add operation"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

try:
    from neuronxcc.nki._pre_prod_kernels.experimental.misc.klir_scatter_add import scatter_add
except ImportError:
    raise ImportError(
        "scatter_add requires neuronxcc >= 2.23. Please upgrade your compiler."
    ) from None

wrapped_scatter_add_nki = wrap_nki(scatter_add)


@nki_op("nki_kernels::scatter_add_kernel", mutates_args={})
def scatter_add_kernel(
    input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """
    Wrapper for NKI scatter_add kernel.

    Args:
        input: Input tensor (2D) - modified in-place
        dim: Dimension (must be 0)
        index: Index tensor (2D)
        src: Source tensor (2D)

    Returns:
        input: Modified input tensor
    """

    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,)
    result = wrapped_scatter_add_nki[grid](input, dim, index, src)
    return result
