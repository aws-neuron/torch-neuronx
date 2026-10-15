"""FFT operations.

The math lives in the shared decomposition table (see the ``_fft_r2c`` entry in
``neuron_dynamo_backend/decompositions.py``). The eager kernel traces this
passthrough with ``make_fx``, which applies that decomposition. So the eager and
compile paths use one implementation.
"""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::_fft_r2c", "aten::_fft_r2c.out"], static_argnums=(1, 2, 3))
def torch_fft_r2c(self, dim, normalization, onesided, out=None):
    """Execute a real-to-complex FFT through the registered ATen operation.

    Args:
        self (torch.Tensor): Real-valued input tensor.
        dim (Sequence[int]): Dimensions to transform.
        normalization (int): ATen FFT normalization mode.
        onesided (bool): Whether to return only nonredundant frequencies.
        out (torch.Tensor, optional): Output accepted for schema registration but unused.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Real-to-complex FFT result.

    Note:
        Tracing uses the shared direct-DFT decomposition for supported single-axis
        inputs and otherwise defers to fallback lowering.
    """
    return torch.ops.aten._fft_r2c(self, dim, normalization, onesided)
