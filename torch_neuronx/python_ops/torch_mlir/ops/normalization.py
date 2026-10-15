"""Normalization operations."""

import torch

from ..operation_registry import register_aten
from ..output_layout import channels_last_strides, suggest_output_memory_format


@register_aten(
    ["aten::native_layer_norm", "aten::native_layer_norm.out"],
    static_argnums=(1, 4),
    output_params=("out0", "out1", "out2"),
)
def torch_native_layer_norm(
    input, normalized_shape, weight=None, bias=None, eps=1e-5, out0=None, out1=None, out2=None
):
    """Apply native layer normalization.

    Args:
        input (torch.Tensor): Input tensor.
        normalized_shape (Sequence[int]): Shape of the dimensions to normalize.
        weight (torch.Tensor | None, optional): Per-element scale. Defaults to ``None``.
        bias (torch.Tensor | None, optional): Per-element bias. Defaults to ``None``.
        eps (float, optional): Value added to the variance for numerical stability.
            Defaults to ``1e-5``.
        out0 (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Mean destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, mean, and rstd.
    """
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
    """Compute native layer-normalization gradients.

    Args:
        grad_out (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        normalized_shape (Sequence[int]): Shape of the normalized dimensions.
        mean (torch.Tensor): Mean saved by the forward operation.
        rstd (torch.Tensor): Reciprocal standard deviation saved by the forward operation.
        weight (torch.Tensor | None): Forward scale tensor, or ``None``.
        bias (torch.Tensor | None): Forward bias tensor, or ``None``.
        output_mask (Sequence[bool]): Selects the gradients to return.
        out0 (torch.Tensor | None, optional): Input-gradient destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Weight-gradient destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Bias-gradient destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor | None, ...]: Input, weight, and bias gradients; ``None`` when masked.
    """
    return torch.ops.aten.native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    )


@register_aten(["aten::batch_norm_backward_elemt", "aten::batch_norm_backward_elemt.out"])
def torch_batch_norm_backward_elemt(
    grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count, out=None
):
    """Compute the element-wise batch-normalization input gradient.

    Args:
        grad_output (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        mean (torch.Tensor): Per-channel mean.
        invstd (torch.Tensor): Per-channel reciprocal standard deviation.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        sum_dy (torch.Tensor): Per-channel sum of output gradients.
        sum_dy_xmu (torch.Tensor): Per-channel sum of gradients times centered inputs.
        count (torch.Tensor or Number): Number of samples represented by the statistics.
        out (torch.Tensor | None, optional): Output destination. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``input``.
    """
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
    """Compute per-channel batch-normalization statistics.

    Args:
        input (torch.Tensor): Input tensor in channel-first layout.
        eps (float): Value added to the variance for numerical stability.
        out0 (torch.Tensor | None, optional): Mean destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Per-channel mean and rstd.
    """
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
    """Apply element-wise batch normalization from supplied statistics.

    Args:
        input (torch.Tensor): Input tensor in channel-first layout.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        mean (torch.Tensor): Per-channel mean.
        invstd (torch.Tensor): Per-channel reciprocal standard deviation.
        eps (float): Unused epsilon retained for schema compatibility.
        out (torch.Tensor | None, optional): Output destination. Defaults to ``None``.

    Returns:
        torch.Tensor: Normalized, scaled, and biased input.
    """
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
    """Combine batch-normalization statistics using device sample counts.

    Args:
        input_tensor (torch.Tensor): Input retained as a zero-valued graph dependency.
        means (torch.Tensor): Per-device channel means.
        invstds (torch.Tensor): Per-device reciprocal standard deviations.
        running_mean (torch.Tensor): Running mean retained as a graph dependency.
        running_var (torch.Tensor): Running variance retained as a graph dependency.
        momentum (Number): Momentum retained as a graph dependency.
        eps (float): Epsilon used to recover and normalize variance.
        counts (torch.Tensor): Per-device sample counts.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Global mean and reciprocal standard deviation.

    Note:
        Only ``means``, ``invstds``, and ``counts`` convert to float32. Running
        statistics must be tensors and are not updated. Zero-count statistics are
        masked; zero times infinity in the variance term can produce ``NaN`` outputs.
    """
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
    """Reduce batch-normalization gradients across non-channel dimensions.

    Args:
        grad_output (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        mean (torch.Tensor): Per-channel mean.
        invstd (torch.Tensor): Per-channel reciprocal standard deviation.
        weight (torch.Tensor | None): Unused scale tensor retained for the schema.
        input_g (bool): Unused input-gradient selection flag.
        weight_g (bool): Unused weight-gradient selection flag.
        bias_g (bool): Unused bias-gradient selection flag.
        out0 (torch.Tensor | None, optional): Gradient-sum destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Centered-sum destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Weight-gradient destination. Defaults to ``None``.
        out3 (torch.Tensor | None, optional): Bias-gradient destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, ...]: Four tensors: two sums, weight gradient, and bias gradient.
    """
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
    """Apply fused root-mean-square normalization.

    Args:
        input (torch.Tensor): Input tensor.
        normalized_shape (Sequence[int]): Shape of the dimensions to normalize.
        weight (torch.Tensor | None, optional): Per-element scale. Defaults to ``None``.
        eps (float | None, optional): Value added for numerical stability. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized output and reciprocal root mean square.
    """
    return torch.ops.aten._fused_rms_norm(input, normalized_shape, weight, eps)


@register_aten(["aten::_fused_rms_norm_backward"], static_argnums=(2, 5))
def torch_fused_rms_norm_backward(grad_out, input, normalized_shape, rstd, weight, output_mask):
    """Compute fused root-mean-square-normalization gradients.

    Args:
        grad_out (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        normalized_shape (Sequence[int]): Shape of the normalized dimensions.
        rstd (torch.Tensor): Reciprocal root mean square saved by the forward operation.
        weight (torch.Tensor | None): Forward scale tensor, or ``None``.
        output_mask (Sequence[bool]): Selects the gradients to return.

    Returns:
        tuple[torch.Tensor | None, ...]: Input and weight gradients; ``None`` when masked.
    """
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
    """Apply native group normalization.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None, optional): Per-channel scale. Defaults to ``None``.
        bias (torch.Tensor | None, optional): Per-channel bias. Defaults to ``None``.
        n (int, optional): Batch size. Defaults to ``0``.
        c (int, optional): Channel count. Defaults to ``0``.
        h_x_w (int, optional): Product of spatial dimensions. Defaults to ``0``.
        group (int, optional): Number of channel groups. Defaults to ``1``.
        eps (float, optional): Value added to the variance for numerical stability.
            Defaults to ``1e-5``.
        out0 (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Mean destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, mean, and rstd.
    """
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
    """Compute native group-normalization gradients.

    Args:
        grad_out (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        mean (torch.Tensor): Group means saved by the forward operation.
        rstd (torch.Tensor): Saved group reciprocal standard deviations.
        weight (torch.Tensor | None): Forward scale tensor, or ``None``.
        n (int): Batch size.
        c (int): Channel count.
        h_x_w (int): Product of spatial dimensions.
        group (int): Number of channel groups.
        output_mask (Sequence[bool]): Selects the gradients to return.
        out0 (torch.Tensor | None, optional): Input-gradient destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Weight-gradient destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Bias-gradient destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor | None, ...]: Input, weight, and bias gradients; ``None`` when masked.
    """
    return torch.ops.aten.native_group_norm_backward(
        grad_out, input, mean, rstd, weight, n, c, h_x_w, group, output_mask
    )


def _set_batch_norm_output_strides(lowering_result, logical_args, logical_kwargs):
    """Preserve a dense channels-last input layout on the normalized output."""
    if not logical_args or not lowering_result.output_specs:
        return
    if any(
        isinstance(value, torch.Tensor)
        for name, value in logical_kwargs.items()
        if name == "out" or name.startswith("out") or name in ("save_mean", "save_invstd")
    ):
        return
    input = logical_args[0]
    if not isinstance(input, torch.Tensor):
        return

    memory_format = suggest_output_memory_format(input)
    spec = lowering_result.output_specs[0]
    if not isinstance(spec, tuple) or len(spec) < 2:
        return
    shape = spec[0]
    target_strides = channels_last_strides(shape, memory_format)
    if target_strides is None:
        return
    lowering_result.output_specs[0] = (shape, spec[1], target_strides)


@register_aten(
    ["aten::_native_batch_norm_legit_functional"],
    static_argnums=(5, 6, 7),
    output_params=("out0", "out1", "out2", "out3", "out4"),
    lowering_result_transform=_set_batch_norm_output_strides,
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
    """Apply functional native batch normalization with running statistics.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        running_mean (torch.Tensor): Running channel means.
        running_var (torch.Tensor): Running channel variances.
        training (bool): Whether to use batch statistics.
        momentum (float): Running-statistics update factor.
        eps (float): Value added to the variance for numerical stability.
        out0 (torch.Tensor | None, optional): Registered output destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Registered saved-mean destination.
            Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Registered saved-rstd destination.
            Defaults to ``None``.
        out3 (torch.Tensor | None, optional): Registered updated-mean destination.
            Defaults to ``None``.
        out4 (torch.Tensor | None, optional): Updated-variance destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, ...]: Output, saved mean and rstd, and updated running
            mean and variance; the saved-stat tensors are empty in inference mode.
    """
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
    lowering_result_transform=_set_batch_norm_output_strides,
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
    """Apply native batch normalization with running statistics.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        running_mean (torch.Tensor): Running channel means updated by the operation.
        running_var (torch.Tensor): Running channel variances updated by the operation.
        training (bool): Whether to use batch statistics.
        momentum (float): Running-statistics update factor.
        eps (float): Value added to the variance for numerical stability.
        out (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        save_mean (torch.Tensor | None, optional): Saved-mean destination. Defaults to ``None``.
        save_invstd (torch.Tensor | None, optional): Saved-rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, saved mean, and saved rstd.

    Raises:
        RuntimeError: If ``out`` has a different dtype from ``input``.
    """
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
    lowering_result_transform=_set_batch_norm_output_strides,
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
    """Apply native batch normalization without running-statistics tensors.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        training (bool): Whether to use batch statistics.
        momentum (float): Running-statistics update factor retained by the schema.
        eps (float): Value added to the variance for numerical stability.
        out (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        save_mean (torch.Tensor | None, optional): Saved-mean destination. Defaults to ``None``.
        save_invstd (torch.Tensor | None, optional): Saved-rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, saved mean, and saved rstd.

    Raises:
        RuntimeError: If ``out`` has a different dtype from ``input``.
    """
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
    lowering_result_transform=_set_batch_norm_output_strides,
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
    """Apply native batch normalization in inference mode.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        running_mean (torch.Tensor): Running channel means.
        running_var (torch.Tensor): Running channel variances.
        momentum (float): Momentum argument forwarded to the ATen operation.
        eps (float): Value added to the variance for numerical stability.
        out0 (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Empty saved-mean destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Empty saved-rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output followed by empty
            saved-mean and saved-rstd tensors.
    """
    return torch.ops.aten._native_batch_norm_legit_no_training(
        input, weight, bias, running_mean, running_var, momentum, eps
    )


@register_aten(
    ["aten::native_batch_norm", "aten::native_batch_norm.out"],
    static_argnums=(5, 6, 7),
    output_params=("out", "save_mean", "save_invstd"),
    lowering_result_transform=_set_batch_norm_output_strides,
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
    """Apply native batch normalization.

    Args:
        input (torch.Tensor): Input tensor.
        weight (torch.Tensor | None): Per-channel scale, or ``None``.
        bias (torch.Tensor | None): Per-channel bias, or ``None``.
        running_mean (torch.Tensor | None): Running channel means, or ``None``.
        running_var (torch.Tensor | None): Running channel variances, or ``None``.
        training (bool): Whether to use batch statistics.
        momentum (float): Running-statistics update factor.
        eps (float): Value added to the variance for numerical stability.
        out (torch.Tensor | None, optional): Output destination. Defaults to ``None``.
        save_mean (torch.Tensor | None, optional): Saved-mean destination. Defaults to ``None``.
        save_invstd (torch.Tensor | None, optional): Saved-rstd destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, saved mean, and saved
            rstd; the saved-stat tensors are empty when ``training`` is ``False``.
    """
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
    """Compute native batch-normalization gradients.

    Args:
        grad_out (torch.Tensor): Gradient of the normalized output.
        input (torch.Tensor): Original input tensor.
        weight (torch.Tensor | None): Forward scale tensor, or ``None``.
        running_mean (torch.Tensor | None): Running channel means, or ``None``.
        running_var (torch.Tensor | None): Running channel variances, or ``None``.
        save_mean (torch.Tensor | None): Mean saved by the forward operation, or ``None``.
        save_invstd (torch.Tensor | None): Saved reciprocal standard deviation, or ``None``.
        training (bool): Whether the forward operation used batch statistics.
        eps (float): Value added to the variance for numerical stability.
        output_mask (Sequence[bool]): Selects the gradients to return.
        out0 (torch.Tensor | None, optional): Input-gradient destination. Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Weight-gradient destination. Defaults to ``None``.
        out2 (torch.Tensor | None, optional): Bias-gradient destination. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor | None, ...]: Input, weight, and bias gradients; ``None`` when masked.
    """
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


@register_aten(
    ["aten::_weight_norm_interface", "aten::_weight_norm_interface.out"],
    static_argnums=(2,),
    output_params=("out0", "out1"),
)
def torch_weight_norm_interface(v, g, dim=0, out0=None, out1=None):
    """Compute normalized weights and their reduction norms.

    Args:
        v (torch.Tensor): Unnormalized weight tensor.
        g (torch.Tensor): Magnitude tensor retained along ``dim``.
        dim (int, optional): Dimension retained by the norm. Defaults to ``0``.
        out0 (torch.Tensor | None, optional): Weight destination handled by registration.
            Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Norm destination handled by registration.
            Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized weight and saved norm.
    """
    return torch.ops.aten._weight_norm_interface(v, g, dim)


@register_aten(
    ["aten::_weight_norm_interface_backward", "aten::_weight_norm_interface_backward.out"],
    static_argnums=(4,),
    output_params=("out0", "out1"),
)
def torch_weight_norm_interface_backward(
    grad_w, saved_v, saved_g, saved_norms, dim, out0=None, out1=None
):
    """Compute gradients for the fused weight-normalization interface.

    Args:
        grad_w (torch.Tensor): Gradient of the normalized weight.
        saved_v (torch.Tensor): Unnormalized weight saved by the forward operation.
        saved_g (torch.Tensor): Magnitude tensor saved by the forward operation.
        saved_norms (torch.Tensor): Reduction norms saved by the forward operation.
        dim (int): Dimension retained by the norm.
        out0 (torch.Tensor | None, optional): Weight-gradient destination handled by registration.
            Defaults to ``None``.
        out1 (torch.Tensor | None, optional): Magnitude-gradient destination handled by
            registration. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gradients for ``saved_v`` and ``saved_g``.
    """
    return torch.ops.aten._weight_norm_interface_backward(
        grad_w, saved_v, saved_g, saved_norms, dim
    )
