"""Normalization operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::native_layer_norm", "aten::native_layer_norm.out"],
    static_argnums=(1, 4),
    output_params=("out0", "out1", "out2"),
)
def torch_native_layer_norm(
    input, normalized_shape, weight=None, bias=None, eps=1e-5, out0=None, out1=None, out2=None
):
    return torch.native_layer_norm(input, normalized_shape, weight, bias, eps)


@register_aten(
    ["aten::native_layer_norm_backward", "aten::native_layer_norm_backward.out"],
    static_argnums=(2, 7),
    output_params=("out0", "out1", "out2"),
)
def torch_native_layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight,
    bias,
    output_mask,
    out0=None,
    out1=None,
    out2=None,
):
    return torch.ops.aten.native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    )


@register_aten(["aten::batch_norm_backward_elemt", "aten::batch_norm_backward_elemt.out"])
def torch_batch_norm_backward_elemt(
    grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count, out=None
):
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.reshape(shape)
    invstd_reshaped = invstd.reshape(shape)
    weight_reshaped = weight.reshape(shape) if weight is not None else 1.0
    sum_dy_reshaped = sum_dy.reshape(shape)
    sum_dy_xmu_reshaped = sum_dy_xmu.reshape(shape)
    total_count = count.sum() if torch.is_tensor(count) else count
    xmu = input - mean_reshaped
    k = sum_dy_xmu_reshaped * invstd_reshaped * invstd_reshaped / total_count
    grad_input = (
        (grad_output - sum_dy_reshaped / total_count - xmu * k) * invstd_reshaped * weight_reshaped
    )
    return grad_input


@register_aten(
    ["aten::batch_norm_stats", "aten::batch_norm_stats.out"],
    static_argnames=("eps",),
    output_params=("out0", "out1"),
)
def torch_batch_norm_stats(input, eps, out0=None, out1=None):
    """Compute batch normalization statistics (mean and inverse std)."""

    # Reduction axes: all except channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Calculate mean and variance
    mean = torch.mean(input, dim=reduction_axes, keepdim=False)
    var = torch.var(input, dim=reduction_axes, unbiased=False, keepdim=False)

    # Calculate inverse standard deviation
    invstd = torch.rsqrt(var + eps)

    return mean, invstd


@register_aten(["aten::batch_norm_elemt", "aten::batch_norm_elemt.out"])
def torch_batch_norm_elemt(input, weight, bias, mean, invstd, eps, out=None):
    """Element-wise batch normalization."""
    # Reshape statistics to broadcast with input (add dimensions for spatial dims)
    shape = [1] * input.ndim
    shape[1] = -1  # Channel dimension

    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)
    weight_reshaped = weight.view(shape) if weight is not None else 1.0
    bias_reshaped = bias.view(shape) if bias is not None else 0.0

    # Apply batch normalization: (input - mean) * invstd * weight + bias
    normalized = (input - mean_reshaped) * invstd_reshaped
    return normalized * weight_reshaped + bias_reshaped


@register_aten(["aten::batch_norm_gather_stats_with_counts"])
def torch_batch_norm_gather_stats_with_counts(
    input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
):
    """Gather batch normalization statistics across devices with counts."""
    # Force all parameters to be used to prevent optimization
    fake_sum = (
        torch.sum(input_tensor * 0)
        + torch.sum(running_mean * 0)
        + torch.sum(running_var * 0)
        + momentum * 0
    )

    # Convert to tensors and ensure float32
    means = means.float()
    invstds = invstds.float()
    counts = counts.float()

    # Mask devices with count=0
    mask = counts > 0
    counts = counts * mask
    means = means * mask.unsqueeze(-1)
    invstds = invstds * mask.unsqueeze(-1)

    # Avoid division by zero if all counts are zero
    total_count = torch.sum(counts)
    total_count = torch.where(total_count == 0, 1.0, total_count)

    # Calculate global mean
    global_mean = torch.sum(means * counts.unsqueeze(-1), dim=0) / total_count

    # Calculate global variance
    var_i = 1.0 / (invstds**2) - eps
    global_var = (
        torch.sum(counts.unsqueeze(-1) * (var_i + (means - global_mean) ** 2), dim=0) / total_count
    )

    # Calculate global inverse standard deviation
    global_invstd = 1.0 / torch.sqrt(global_var + eps)

    return global_mean + fake_sum, global_invstd + fake_sum


@register_aten(
    ["aten::batch_norm_backward_reduce", "aten::batch_norm_backward_reduce.out"],
    output_params=("out0", "out1", "out2", "out3"),
)
def torch_batch_norm_backward_reduce(
    grad_output,
    input,
    mean,
    invstd,
    weight,
    input_g,
    weight_g,
    bias_g,
    out0=None,
    out1=None,
    out2=None,
    out3=None,
):
    """Backward pass reduction for batch normalization."""
    # Reduction axes: all except channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Reshape statistics for broadcasting
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)

    # Compute normalized input
    xmu = input - mean_reshaped
    xhat = xmu * invstd_reshaped

    # Compute sum_dy and sum_dy_xmu
    sum_dy = torch.sum(grad_output, dim=reduction_axes, keepdim=False)
    sum_dy_xmu = torch.sum(grad_output * xmu, dim=reduction_axes, keepdim=False)

    # Compute gradients (always compute, return based on flags)
    grad_weight = torch.sum(grad_output * xhat, dim=reduction_axes, keepdim=False)
    grad_bias = sum_dy.clone()

    return sum_dy, sum_dy_xmu, grad_weight, grad_bias


@register_aten(["aten::_fused_rms_norm"], static_argnums=(1, 3))
def torch_fused_rms_norm(input, normalized_shape, weight=None, eps=None):
    return torch.ops.aten._fused_rms_norm(input, normalized_shape, weight, eps)


@register_aten(["aten::_fused_rms_norm_backward"], static_argnums=(2, 5))
def torch_fused_rms_norm_backward(grad_out, input, normalized_shape, rstd, weight, output_mask):
    return torch.ops.aten._fused_rms_norm_backward(
        grad_out, input, normalized_shape, rstd, weight, output_mask
    )


@register_aten(
    ["aten::native_group_norm", "aten::native_group_norm.out"],
    static_argnums=(3, 4, 5, 6, 7),
    output_params=("out0", "out1", "out2"),
)
def torch_native_group_norm(
    input,
    weight=None,
    bias=None,
    n=0,
    c=0,
    h_x_w=0,
    group=1,
    eps=1e-5,
    out0=None,
    out1=None,
    out2=None,
):
    return torch.ops.aten.native_group_norm(input, weight, bias, n, c, h_x_w, group, eps)


@register_aten(
    ["aten::native_group_norm_backward", "aten::native_group_norm_backward.out"],
    static_argnums=(5, 6, 7, 8, 9),
    output_params=("out0", "out1", "out2"),
)
def torch_native_group_norm_backward(
    grad_out,
    input,
    mean,
    rstd,
    weight,
    n,
    c,
    h_x_w,
    group,
    output_mask,
    out0=None,
    out1=None,
    out2=None,
):
    return torch.ops.aten.native_group_norm_backward(
        grad_out, input, mean, rstd, weight, n, c, h_x_w, group, output_mask
    )


@register_aten(
    ["aten::_native_batch_norm_legit_functional"],
    static_argnums=(5, 6, 7),
    output_params=("out0", "out1", "out2", "out3", "out4"),
)
def torch_native_batch_norm_legit_functional(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    out0=None,
    out1=None,
    out2=None,
    out3=None,
    out4=None,
):
    return torch.ops.aten._native_batch_norm_legit_functional(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


def _validate_batch_norm_out(input, *args, out=None, **kwargs):
    """Reject an out tensor whose dtype doesn't match the input (unsafe cast)."""
    if out is not None and out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out.dtype} instead"
        )


@register_aten(
    ["aten::_native_batch_norm_legit", "aten::_native_batch_norm_legit.out"],
    static_argnums=(5, 6, 7),
    output_params=("out", "save_mean", "save_invstd"),
    validate_static_inputs=_validate_batch_norm_out,
)
def torch_native_batch_norm_legit(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    out=None,
    save_mean=None,
    save_invstd=None,
):
    return torch.ops.aten._native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


@register_aten(
    [
        "aten::_native_batch_norm_legit.no_stats",
        "aten::_native_batch_norm_legit.no_stats_out",
    ],
    static_argnums=(3, 4, 5),
    output_params=("out", "save_mean", "save_invstd"),
    validate_static_inputs=_validate_batch_norm_out,
)
def torch_native_batch_norm_legit_no_stats(
    input,
    weight,
    bias,
    training,
    momentum,
    eps,
    out=None,
    save_mean=None,
    save_invstd=None,
):
    return torch.ops.aten._native_batch_norm_legit.no_stats(
        input, weight, bias, training, momentum, eps
    )


@register_aten(
    [
        "aten::_native_batch_norm_legit_no_training",
        "aten::_native_batch_norm_legit_no_training.out",
    ],
    static_argnums=(5, 6),
    output_params=("out0", "out1", "out2"),
)
def torch_native_batch_norm_legit_no_training(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    momentum,
    eps,
    out0=None,
    out1=None,
    out2=None,
):
    return torch.ops.aten._native_batch_norm_legit_no_training(
        input, weight, bias, running_mean, running_var, momentum, eps
    )


@register_aten(
    ["aten::native_batch_norm", "aten::native_batch_norm.out"],
    static_argnums=(5, 6, 7),
    output_params=("out", "save_mean", "save_invstd"),
)
def torch_native_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    out=None,
    save_mean=None,
    save_invstd=None,
):
    return torch.ops.aten.native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


@register_aten(
    ["aten::native_batch_norm_backward", "aten::native_batch_norm_backward.out"],
    static_argnums=(7, 8, 9),
    output_params=("out0", "out1", "out2"),
)
def torch_native_batch_norm_backward(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    training,
    eps,
    output_mask,
    out0=None,
    out1=None,
    out2=None,
):
    return torch.ops.aten.native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
        output_mask,
    )
