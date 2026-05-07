"""Convolution operations."""

import torch

from ..operation_registry import register_aten


def _expand_param_if_needed(param, num_spatial_dims):
    """Expand parameter to match number of spatial dimensions if needed."""
    if len(param) < num_spatial_dims:
        return param * num_spatial_dims
    return param


def _validate_convolution_inputs(
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Common validation for convolution operations using PyTorch-style checks."""
    k = input.ndim
    weight_dim = weight.ndim
    dim = weight_dim - 2

    if dim <= 0:
        raise RuntimeError("weight should have at least three dimensions")

    if groups <= 0:
        raise RuntimeError("non-positive groups is not supported")

    if any(p < 0 for p in padding):
        raise RuntimeError("negative padding is not supported")

    if transposed and any(p < 0 for p in output_padding):
        raise RuntimeError("negative output_padding is not supported")

    if any(s <= 0 for s in stride):
        raise RuntimeError("non-positive stride is not supported")

    if any(d <= 0 for d in dilation):
        raise RuntimeError("dilation should be greater than zero")

    if weight_dim != k:
        raise RuntimeError(
            f"Expected {weight_dim}-dimensional input for {weight_dim}-dimensional weight "
            f"{list(weight.shape)}, but got {k}-dimensional input of size "
            f"{list(input.shape)} instead"
        )

    if weight.shape[0] < groups:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be at least {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if weight.shape[0] % groups != 0:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be divisible by {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if not transposed:
        expected_input_channels = weight.shape[1] * groups
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given groups={groups}, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )

        for i in range(2, k):
            input_size = input.shape[i] + 2 * padding[i - 2]
            kernel_size = dilation[i - 2] * (weight.shape[i] - 1) + 1
            if input_size < kernel_size:
                raise RuntimeError(
                    f"Calculated padded input size per channel: ({input_size}). "
                    f"Kernel size: ({kernel_size}). Kernel size can't be greater than "
                    f"actual input size"
                )
    else:
        expected_input_channels = weight.shape[0]
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given transposed=True, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )


@register_aten(["aten::convolution"], static_argnums=(3, 4, 5, 6, 7, 8), uses_preprocessing=True)
def torch_convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    num_spatial_dims = weight.ndim - 2
    padding = _expand_param_if_needed(padding, num_spatial_dims)
    stride = _expand_param_if_needed(stride, num_spatial_dims)
    dilation = _expand_param_if_needed(dilation, num_spatial_dims)
    output_padding = _expand_param_if_needed(output_padding, num_spatial_dims)

    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )

    def conv_kernel(inp, wgt, bias, stride, padding, dilation, transposed, output_padding, groups):
        if transposed:
            if inp.dim() == 3:
                return torch.nn.functional.conv_transpose1d(
                    inp, wgt, bias, stride, padding, output_padding, groups, dilation
                )
            elif inp.dim() == 4:
                return torch.nn.functional.conv_transpose2d(
                    inp, wgt, bias, stride, padding, output_padding, groups, dilation
                )
            else:
                return torch.nn.functional.conv_transpose3d(
                    inp, wgt, bias, stride, padding, output_padding, groups, dilation
                )
        else:
            if inp.dim() == 3:
                return torch.nn.functional.conv1d(inp, wgt, bias, stride, padding, dilation, groups)
            elif inp.dim() == 4:
                return torch.nn.functional.conv2d(inp, wgt, bias, stride, padding, dilation, groups)
            else:
                return torch.nn.functional.conv3d(inp, wgt, bias, stride, padding, dilation, groups)

    return (
        conv_kernel,
        (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups),
        {},
    )


@register_aten(["aten::convolution_backward"], static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
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
    )
