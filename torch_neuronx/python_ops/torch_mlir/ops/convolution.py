"""Convolution operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::convolution"], static_argnums=(3, 4, 5, 6, 7, 8), uses_preprocessing=True)
def torch_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    if bias is None:
        # Create graph without input bias parameter
        def conv_no_bias(inp, wgt):
            return (
                torch.nn.functional.conv1d(inp, wgt, None, stride, padding, dilation, groups)
                if inp.dim() == 3
                else torch.nn.functional.conv2d(inp, wgt, None, stride, padding, dilation, groups)
                if inp.dim() == 4
                else torch.nn.functional.conv3d(inp, wgt, None, stride, padding, dilation, groups)
            )

        return conv_no_bias, (input, weight), {}
    else:
        # Create graph with input bias parameter
        def conv_with_bias(inp, wgt, b):
            return torch.convolution(
                inp, wgt, b, stride, padding, dilation, transposed, output_padding, groups
            )

        return conv_with_bias, (input, weight, bias), {}


@register_aten(
    ["aten::convolution_backward"],
    static_argnums=(3, 4, 5, 6, 7, 8, 9, 10),
)
def torch_convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    **kwargs,
):
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        **kwargs,
    )
