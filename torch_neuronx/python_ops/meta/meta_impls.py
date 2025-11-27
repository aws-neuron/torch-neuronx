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
