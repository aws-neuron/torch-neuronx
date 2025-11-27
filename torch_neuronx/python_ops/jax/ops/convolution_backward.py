import jax
import jax.numpy as jnp

from torch_neuronx.python_ops.jax.jax_impl import _aten_convolution
from torch_neuronx.python_ops.jax.operation_registry import register_aten


def expand_param_if_needed(param, num_spatial_dims):
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
    # Reference: https://tiny.amazon.com/45i1y0ma/githpytopytoblobmainatensrc
    k = input.ndim
    weight_dim = weight.ndim
    dim = weight_dim - 2

    if dim <= 0:
        raise RuntimeError("weight should have at least three dimensions")

    if groups <= 0:
        raise RuntimeError("non-positive groups is not supported")

    # Check for negative padding
    if any(p < 0 for p in padding):
        raise RuntimeError("negative padding is not supported")

    # Check for negative output padding
    if transposed and any(p < 0 for p in output_padding):
        raise RuntimeError("negative output_padding is not supported")

    # Check for non-positive stride
    if any(s <= 0 for s in stride):
        raise RuntimeError("non-positive stride is not supported")

    # Check for negative dilation
    if any(d <= 0 for d in dilation):
        raise RuntimeError("dilation should be greater than zero")

    # Check input and weight dimensions match
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

    if input.dtype != weight.dtype:
        raise RuntimeError(
            f"Input type ({input.dtype}) and weight type ({weight.dtype}) should be the same"
        )

    if not transposed:
        # Check input channels match weight
        expected_input_channels = weight.shape[1] * groups
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given groups={groups}, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )

        # Check kernel size vs input size for each spatial dimension
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
    """Validate inputs for convolution backward operation"""
    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )
    if grad_output.ndim != input.ndim:
        raise RuntimeError(
            f"Expected input and grad_output to have the same number of dimensions, "
            f"got: {input.ndim} and {grad_output.ndim}"
        )


@register_aten(
    ["aten::convolution_backward"],
    operation_type="conv",
    static_argnums=(3, 4, 5, 6, 7, 8, 9, 10),
    output_params=(),
)
def _aten_convolution_backward(
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
    output_mask,  # (grad_input, grad_weight, grad_bias)
):
    num_spatial_dims = weight.ndim - 2
    padding = expand_param_if_needed(padding, num_spatial_dims)

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

    want_ginp, want_gw, want_gb = tuple(output_mask)

    # Check for empty tensors and handle them before VJP computation
    input_has_empty_dim = any(dim == 0 for dim in input.shape)
    weight_has_empty_dim = any(dim == 0 for dim in weight.shape)
    grad_output_has_empty_dim = any(dim == 0 for dim in grad_output.shape)

    if input_has_empty_dim or weight_has_empty_dim or grad_output_has_empty_dim:
        # Return zero gradients with correct shapes for empty tensor case
        grad_input = jnp.zeros_like(input) if want_ginp else None
        grad_weight = jnp.zeros_like(weight) if want_gw else None
        grad_bias = None
        if want_gb and bias_sizes is not None:
            # Use grad_output.dtype instead of input.dtype for consistency
            grad_bias = jnp.zeros(bias_sizes, dtype=grad_output.dtype)
        return grad_input, grad_weight, grad_bias

    # Match your forward's batch handling
    num_shape_dim = weight.ndim - 1
    batch_dims = input.shape[:-num_shape_dim]
    input_ = input.reshape((-1, *input.shape[-num_shape_dim:]))
    grad_output_ = grad_output.reshape((-1, *grad_output.shape[-num_shape_dim:]))

    # Forward closure (bias=None; bias grad handled separately)
    def fwd(inp, w):
        return _aten_convolution(
            inp, w, None, stride, padding, dilation, transposed, output_padding, groups
        )

    grad_input = None
    grad_weight = None
    grad_bias = None

    # Use VJP to get both dL/dx and dL/dw correctly for all cases
    # (groups, dilation, transposed, etc.)
    if want_ginp or want_gw:
        (_, pullback) = jax.vjp(fwd, input_, weight)
        ginp_, gw_ = pullback(grad_output_)
        if want_ginp:
            grad_input = ginp_.reshape((*batch_dims, *ginp_.shape[-num_shape_dim:]))
        if want_gw:
            grad_weight = gw_

    # Bias grad = sum over N and spatial dims (keep channel axis = 1)
    if want_gb and (bias_sizes is not None):
        reduce_axes = tuple(ax for ax in range(grad_output_.ndim) if ax != 1)
        grad_bias = jnp.sum(grad_output_, axis=reduce_axes)

    return grad_input, grad_weight, grad_bias
