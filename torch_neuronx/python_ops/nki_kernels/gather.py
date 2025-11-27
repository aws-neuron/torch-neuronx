"""NKI kernel wrapper for gather operation"""

import neuronxcc.nki.language as nl

from torch_neuronx.utils import get_logical_neuron_cores


def gather_kernel(input, dim, index):
    """
    Wrapper for NKI gather kernel.

    Args:
        input: Input tensor (2D)
        dim: Dimension to gather from (must be 0)
        index: Index tensor (2D)

    Returns:
        Gathered tensor
    """
    from neuronxcc.nki._pre_prod_kernels.experimental.misc.gather import gather

    from torch_neuronx import TorchNeuronNKIKernel

    logical_neuron_cores = int(get_logical_neuron_cores())
    traced_kernel = TorchNeuronNKIKernel(
        func=getattr(gather, "func", gather),
        grid=(nl.nc(logical_neuron_cores),),  # LNC2 - uses 2 physical cores
        kernel_return=True,
    )

    result = traced_kernel(input=input, dim=dim, index=index)

    return result
