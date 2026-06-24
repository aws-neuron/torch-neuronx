"""Pooling operations registered via register_aten for eager dispatch."""

import torch

from ..operation_registry import register_aten


def _validate_adaptive_avg_pool2d(x, output_size, out=None):
    """Reject non-floating-point input, matching CPU's adaptive_avg_pool2d."""
    if not x.is_floating_point():
        raise RuntimeError(f"\"adaptive_avg_pool2d\" not implemented for '{x.dtype}'")


@register_aten(
    ["aten::_adaptive_avg_pool2d", "aten::_adaptive_avg_pool2d.out"],
    output_params=("out",),
    static_argnums=(1,),
    validate_static_inputs=_validate_adaptive_avg_pool2d,
)
def torch_adaptive_avg_pool2d(x, output_size, out=None):
    # An empty output dimension yields an empty result; short-circuit before the
    # upstream decomposition, which divides input shape by output_size (% 0 crash).
    if 0 in tuple(output_size):
        return x.new_empty((*x.shape[:-2], *output_size))
    return torch.ops.aten._adaptive_avg_pool2d(x, output_size)


def _validate_adaptive_avg_pool2d_backward(grad_output, self, out=None):
    """Reject grad_output whose rank doesn't match the input, matching CPU."""
    if grad_output.ndim != self.ndim:
        raise RuntimeError(
            f"Expected dimensions {self.ndim} for grad_output but got dimensions "
            f"{grad_output.ndim}"
        )


@register_aten(
    ["aten::_adaptive_avg_pool2d_backward", "aten::_adaptive_avg_pool2d_backward.out"],
    output_params=("out",),
    validate_static_inputs=_validate_adaptive_avg_pool2d_backward,
)
def torch_adaptive_avg_pool2d_backward(grad_output, self, out=None):
    return torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, self)
