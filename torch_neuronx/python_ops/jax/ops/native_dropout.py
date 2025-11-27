import logging

import torch

from torch_neuronx.python_ops.jax.operation_registry import register_aten

logger = logging.getLogger(__name__)


@register_aten(
    [
        "aten::native_dropout",
    ],
    static_argnums=(1, 2),
    uses_preprocessing=True,
)
def _aten_native_dropout(input, p: float, train: bool | None = True, **kwargs):
    def jax_native_dropout(x, p, train, mask, scale):
        y = x * mask.astype(x.dtype) * scale
        return y, mask

    if train is None:
        train = True

    if train and not (0.0 <= float(p) <= 1.0):
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not train or float(p) == 0.0:
        mask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        scale = 1.0
    elif float(p) == 1.0:
        mask = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        scale = 0.0
    else:
        # Generate this on CPU and bring to neuron
        mask = torch.empty(input.shape, dtype=torch.bool)
        keep_prob = 1.0 - float(p)
        mask = torch.bernoulli(mask, keep_prob).to(input.device)
        scale = 1.0 / keep_prob

    # Return (actual_jax_fn, processed_args). Keep (p, train) to honor static_argnums.
    return jax_native_dropout, (input, p, train, mask, scale), {}


@register_aten(
    [
        "aten::native_dropout_backward",
    ],
    static_argnums=(2,),
)
def _aten_native_dropout_backward(grad_output, mask, scale: float, **kwargs):
    """Backward for dropout: grad_input = grad_output * mask * scale."""
    return grad_output * mask.astype(grad_output.dtype) * scale
