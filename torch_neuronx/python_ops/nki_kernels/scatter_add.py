"""NKI kernel wrapper for scatter_add operation"""

import neuronxcc.nki.language as nl

from torch_neuronx.utils import get_logical_neuron_cores


def scatter_add_kernel(input, dim, index, src):
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
    from neuronxcc.nki._pre_prod_kernels.experimental.misc.scatter_add import scatter_add

    from torch_neuronx import TorchNeuronNKIKernel

    logical_neuron_cores = int(get_logical_neuron_cores())
    traced_kernel = TorchNeuronNKIKernel(
        func=getattr(scatter_add, "func", scatter_add),
        grid=(nl.nc(logical_neuron_cores),),
        kernel_return=True,
        return_tensor_overrides=input,
    )

    result = traced_kernel(input=input, dim=dim, index=index, src=src)

    return result
