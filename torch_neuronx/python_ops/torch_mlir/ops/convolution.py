"""Convolution operations."""

import torch
import torch.nn.functional as functional

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


def _validate_convolution_backward_inputs(
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
):
    """Validate inputs for convolution backward operation."""
    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )
    if grad_output.ndim != input.ndim:
        raise RuntimeError(
            f"Expected input and grad_output to have the same number of dimensions, "
            f"got: {input.ndim} and {grad_output.ndim}"
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


def _tuple(x: list | tuple | int, ndim: int):
    """Convert to tuple of length ndim."""
    if isinstance(x, (list, tuple)):  # noqa: UP038
        return tuple(int(v) for v in x) if len(x) else (0,) * ndim
    return (int(x),) * ndim


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
    dtype = input.dtype
    ndim = input.ndim - 2  # spatial dimensions (1, 2, or 3)

    stride = _tuple(stride, ndim)
    padding = _tuple(padding, ndim)
    dilation = _tuple(dilation, ndim)
    output_padding = _tuple(output_padding, ndim)

    _validate_convolution_backward_inputs(
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
    )

    n, cin = input.shape[:2]
    cout = weight.shape[0] if not transposed else weight.shape[1]
    spatial_in = input.shape[2:]
    spatial_out = grad_output.shape[2:]
    kernel_size = weight.shape[2:]

    need_gi = bool(output_mask[0]) if len(output_mask) > 0 else True
    need_gw = bool(output_mask[1]) if len(output_mask) > 1 else True
    need_gb = bool(output_mask[2]) if len(output_mask) > 2 else True

    grad_input = grad_weight = grad_bias = None

    # Select conv functions based on ndim
    conv_fn = [functional.conv1d, functional.conv2d, functional.conv3d][ndim - 1]
    conv_transpose_fn = [
        functional.conv_transpose1d,
        functional.conv_transpose2d,
        functional.conv_transpose3d,
    ][ndim - 1]

    # --- grad_input ---
    if need_gi:
        if not transposed:
            # Compute output_padding to match input spatial size
            # output_size = (input_size - 1) * stride - 2 * padding +
            # dilation * (kernel - 1) + 1 + output_padding
            grad_input_output_padding = tuple(
                max(
                    0,
                    spatial_in[i]
                    - (
                        (spatial_out[i] - 1) * stride[i]
                        - 2 * padding[i]
                        + dilation[i] * (kernel_size[i] - 1)
                        + 1
                    ),
                )
                for i in range(ndim)
            )
            grad_input = conv_transpose_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=grad_input_output_padding,
                groups=groups,
                dilation=dilation,
            ).to(dtype)
        else:
            # transposed conv: grad_input = conv(grad_output, weight)
            grad_input = conv_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype)

    # --- grad_bias ---
    if need_gb and bias_sizes is not None and len(bias_sizes) > 0:
        reduce_dims = (0, *tuple(range(2, input.ndim)))
        grad_bias = grad_output.sum(dim=reduce_dims).to(dtype)

    # --- grad_weight ---
    if need_gw:
        cin_per_group = cin // groups
        cout_per_group = cout // groups
        is_depthwise = groups == cin and cin == cout and cin_per_group == 1

        if not transposed:
            if is_depthwise:
                # Depthwise convolution: use grouped conv trick for efficiency
                # Reshape: [N, C, *spatial] -> [C, N, *spatial] -> [1, C*N, *spatial]
                x_big = input.transpose(0, 1).reshape(1, cin * n, *spatial_in)
                k_big = grad_output.transpose(0, 1).reshape(cin * n, 1, *spatial_out).contiguous()

                dw_big = conv_fn(
                    x_big, k_big, bias=None, stride=1, padding=padding, dilation=1, groups=cin * n
                )

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_big = dw_big[:, :, :: dilation[0]]
                    dw_big = dw_big[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                # Reshape to [C, N, *K], sum over N -> [C, *K], then unsqueeze -> [C, 1, *K]
                dw_cnk = dw_big.reshape(cin, n, *kernel_size)
                grad_weight = dw_cnk.sum(dim=1).unsqueeze(1).to(dtype)
            else:
                # General grouped convolution
                dw_groups = []
                for g in range(groups):
                    x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                    dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                    x_t = x_g.transpose(0, 1).contiguous()
                    dy_t = dy_g.transpose(0, 1).contiguous()

                    dw_g = conv_fn(x_t, dy_t, bias=None, stride=1, padding=padding, dilation=stride)

                    # Subsample by dilation and slice to kernel_size
                    if ndim == 1:
                        dw_g = dw_g[:, :, :: dilation[0]]
                        dw_g = dw_g[:, :, : kernel_size[0]]
                    elif ndim == 2:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                    else:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                    dw_groups.append(dw_g.transpose(0, 1))

                grad_weight = torch.cat(dw_groups, dim=0).to(dtype)
        else:
            # Transposed conv: weight is [Cin, Cout/groups, *kernel_size]
            dw_groups = []
            for g in range(groups):
                x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                x_t = x_g.transpose(0, 1).contiguous()
                dy_t = dy_g.transpose(0, 1).contiguous()

                dw_g = conv_fn(dy_t, x_t, bias=None, stride=1, padding=padding, dilation=stride)

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_g = dw_g[:, :, :: dilation[0]]
                    dw_g = dw_g[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                dw_groups.append(dw_g.transpose(0, 1))

            grad_weight = torch.cat(dw_groups, dim=0).to(dtype)

    return grad_input, grad_weight, grad_bias
