import torch

# Operation configurations for scalable handling
OP_CONFIGS = {
    "batch_norm_gather_stats_with_counts": {
        "outputs": [
            {"shape_fn": lambda inp: (inp.shape[-1],), "dtype_fn": lambda inp: inp.dtype},
            {"shape_fn": lambda inp: (inp.shape[-1],), "dtype_fn": lambda inp: inp.dtype},
        ],
        "input_index": 1,
        "is_single": False,
    },
    "batch_norm_stats": {
        "outputs": [
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
        ],
        "input_index": 0,
        "is_single": False,
    },
    "batch_norm_elemt": {
        "outputs": [{"shape_fn": lambda inp: inp.shape, "dtype_fn": lambda inp: inp.dtype}],
        "input_index": 0,
        "is_single": True,
    },
    "batch_norm_backward_reduce": {
        "outputs": [
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
            {"shape_fn": lambda inp: (inp.shape[1],), "dtype_fn": lambda inp: inp.dtype},
        ],
        "input_index": 0,
        "is_single": False,
    },
    "batch_norm_backward_elemt": {
        "outputs": [{"shape_fn": lambda inp: inp.shape, "dtype_fn": lambda inp: inp.dtype}],
        "input_index": 0,
        "is_single": True,
    },
}


def _meta_impl_template(op_name: str, *args):
    config = OP_CONFIGS[op_name]
    input_tensor_for_shape_dtype = args[config["input_index"]]
    output_tensors = []
    for output_config in config["outputs"]:
        output_shape = output_config["shape_fn"](input_tensor_for_shape_dtype)
        output_dtype = output_config["dtype_fn"](input_tensor_for_shape_dtype)
        output_tensors.append(torch.empty(output_shape, dtype=output_dtype, device="meta"))

    if config["is_single"]:
        return output_tensors[0]
    else:
        return tuple(output_tensors)


@torch.library.impl("aten::batch_norm_gather_stats_with_counts", "Meta")
def batch_norm_gather_stats_with_counts_meta_impl(
    input: torch.Tensor,
    means: torch.Tensor,
    invstds: torch.Tensor,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    momentum: float,
    eps: float,
    counts: torch.Tensor,
):
    """Create meta outputs for batch normalization statistics gathering.

    Args:
        input (torch.Tensor): Input argument, which is ignored when constructing outputs.
        means (torch.Tensor): Tensor whose final dimension and dtype define both outputs.
        invstds (torch.Tensor): Inverse standard deviations, which are ignored.
        running_mean (torch.Tensor or None): Running mean, which is ignored.
        running_var (torch.Tensor or None): Running variance, which is ignored.
        momentum (float): Momentum, which is ignored.
        eps (float): Epsilon, which is ignored.
        counts (torch.Tensor): Counts, which are ignored.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Meta tensors matching the final dimension and
            dtype of ``means``.
    """
    return _meta_impl_template(
        "batch_norm_gather_stats_with_counts",
        input,
        means,
        invstds,
        running_mean,
        running_var,
        momentum,
        eps,
        counts,
    )


@torch.library.impl("aten::batch_norm_stats", "Meta")
def batch_norm_stats_meta_impl(
    input: torch.Tensor,
    eps: float,
):
    """Create meta outputs for batch normalization statistics.

    Args:
        input (torch.Tensor): Tensor whose channel dimension and dtype define both outputs.
        eps (float): Epsilon, which is ignored.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Channel-shaped meta tensors with the input dtype.
    """
    return _meta_impl_template("batch_norm_stats", input, eps)


@torch.library.impl("aten::batch_norm_elemt", "Meta")
def batch_norm_elemt_meta_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    eps: float,
):
    """Create the meta output for elementwise batch normalization.

    Args:
        input (torch.Tensor): Tensor whose shape and dtype define the output.
        weight (torch.Tensor or None): Weight, which is ignored.
        bias (torch.Tensor or None): Bias, which is ignored.
        mean (torch.Tensor): Mean, which is ignored.
        invstd (torch.Tensor): Inverse standard deviation, which is ignored.
        eps (float): Epsilon, which is ignored.

    Returns:
        torch.Tensor: Meta tensor matching the input shape and dtype.
    """
    return _meta_impl_template("batch_norm_elemt", input, weight, bias, mean, invstd, eps)


@torch.library.impl("aten::batch_norm_backward_reduce", "Meta")
def batch_norm_backward_reduce_meta_impl(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    training: bool,
    eps: float,
):
    """Create meta outputs for the batch normalization backward reduction.

    Args:
        grad_output (torch.Tensor): Tensor whose channel dimension and dtype define the outputs.
        input (torch.Tensor): Forward input, which is ignored.
        mean (torch.Tensor): Mean, which is ignored.
        invstd (torch.Tensor): Inverse standard deviation, which is ignored.
        weight (torch.Tensor or None): Weight, which is ignored.
        bias (torch.Tensor): Annotated as a tensor, but ATen passes the ``input_g``
            gradient-request boolean in this position. Ignored either way.
        training (bool): ATen's ``weight_g`` gradient-request flag, which is ignored.
        eps (float): Annotated as a float, but ATen passes the ``bias_g``
            gradient-request boolean in this position. Ignored either way.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Channel-shaped
            meta tensors with the gradient dtype.
    """
    return _meta_impl_template(
        "batch_norm_backward_reduce", grad_output, input, mean, invstd, weight, bias, training, eps
    )


@torch.library.impl("aten::batch_norm_backward_elemt", "Meta")
def batch_norm_backward_elemt_meta_impl(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    weight: torch.Tensor | None,
    sum_dy: torch.Tensor,
    sum_dy_xmu: torch.Tensor,
    count: torch.Tensor,
):
    """Create the meta output for elementwise batch normalization backward.

    Args:
        grad_output (torch.Tensor): Tensor whose shape and dtype define the output.
        input (torch.Tensor): Forward input, which is ignored.
        mean (torch.Tensor): Mean, which is ignored.
        invstd (torch.Tensor): Inverse standard deviation, which is ignored.
        weight (torch.Tensor or None): Weight, which is ignored.
        sum_dy (torch.Tensor): Gradient sum, which is ignored.
        sum_dy_xmu (torch.Tensor): Centered-gradient sum, which is ignored.
        count (torch.Tensor): Element count, which is ignored.

    Returns:
        torch.Tensor: Meta tensor matching the gradient output shape and dtype.
    """
    return _meta_impl_template(
        "batch_norm_backward_elemt",
        grad_output,
        input,
        mean,
        invstd,
        weight,
        sum_dy,
        sum_dy_xmu,
        count,
    )
