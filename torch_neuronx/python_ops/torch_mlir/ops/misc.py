"""Miscellaneous operations that don't fit into other categories."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::embedding"], static_argnums=(2, 3, 4))
def torch_embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return torch.embedding(
        weights, indices, padding_idx=padding_idx, scale_grad_by_freq=False, sparse=sparse
    )


@register_aten(
    ["aten::embedding_dense_backward", "aten::embedding_dense_backward.out"],
    static_argnums=(2, 3, 4),
)
def torch_embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out=None
):
    if indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "Expected tensor for argument #2 'indices' to have one of the following "
            "scalar types: Long, Int; but got torch.FloatTensor instead "
            "(while checking arguments for embedding_backward)"
        )
    return torch.ops.aten.embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_aten(["aten::_softmax", "aten::_softmax.out"], static_argnums=(1, 2))
def torch_softmax_internal(x, dim, half_to_float, out=None):
    if half_to_float:
        x = x.float()
    return torch.softmax(x, dim)


@register_aten(["aten::_log_softmax", "aten::_log_softmax.out"], static_argnums=(1, 2))
def torch_log_softmax_internal(x, dim, half_to_float, out=None):
    if half_to_float:
        x = x.float()
    return torch.log_softmax(x, dim)


@register_aten(["aten::softmax"], static_argnums=(1,), static_argnames=("dtype",))
def torch_softmax(x, dim, dtype=None):
    return torch.softmax(x, dim, dtype=dtype)


@register_aten(["aten::log_softmax"], static_argnums=(1,), static_argnames=("dtype",))
def torch_log_softmax(x, dim, dtype=None):
    return torch.log_softmax(x, dim, dtype=dtype)


@register_aten(
    ["aten::_softmax_backward_data", "aten::_softmax_backward_data.out"], static_argnums=(2, 3)
)
def torch_softmax_backward_data(grad_output, output, dim, input_dtype, out=None):
    return torch.ops.aten._softmax_backward_data(grad_output, output, dim, input_dtype)


@register_aten(
    ["aten::_log_softmax_backward_data", "aten::_log_softmax_backward_data.out"],
    static_argnums=(2, 3),
)
def torch_log_softmax_backward_data(grad_output, output, dim, input_dtype, out=None):
    return torch.ops.aten._log_softmax_backward_data(grad_output, output, dim, input_dtype)


@register_aten(
    [
        "aten::lerp.Scalar",
        "aten::lerp.Scalar_out",
        "aten::lerp.Tensor",
        "aten::lerp.Tensor_out",
        "aten::lerp_.Scalar",
        "aten::lerp_.Tensor",
    ]
)
def torch_lerp(start, end, weight, out=None):
    return torch.ops.aten.lerp(start, end, weight)


@register_aten(
    [
        "aten::histc",
        "aten::histc.out",
    ],
    static_argnames=("bins", "range_min", "range_max"),
    uses_preprocessing=True,
)
def torch_histc(
    input: torch.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0, out=None
) -> torch.Tensor:
    """Fast histogram using linear interpolation like PyTorch."""
    if out is not None and input.dtype != out.dtype:
        raise RuntimeError(
            "torch.histogram: input tensor and hist tensor should have the same dtype, "
            f"but got input {input.dtype} and hist {out.dtype}"
        )

    # Handle auto range
    if min == 0.0 and max == 0.0 and input.numel() > 0:
        min = input.min().item()
        max = input.max().item()

    # Handle special case where min == max
    range_min = min - 0.5 if min == max else min
    range_max = max + 0.5 if min == max else max

    def efficient_histc(input, bins, range_min, range_max, out=None):
        data = input.flatten()
        # Linear interpolation
        bin_idx = ((data - range_min) * bins / (range_max - range_min)).to(torch.int32)
        # Clamp to valid range and handle edge cases
        in_range = (data >= range_min) & (data <= range_max)
        bin_idx = torch.where(bin_idx >= bins, bins - 1, bin_idx)
        bin_idx = torch.where(bin_idx < 0, 0, bin_idx)
        # Vectorized counting without scatter
        bin_indices = torch.arange(bins, device=data.device)
        matches = bin_idx.unsqueeze(1) == bin_indices.unsqueeze(0)
        valid_matches = matches & in_range.unsqueeze(1)
        counts = valid_matches.sum(dim=0, dtype=input.dtype)
        return counts

    return (
        efficient_histc,
        (input,),
        {"bins": bins, "range_min": range_min, "range_max": range_max, "out": out},
    )


@register_aten(["aten::threshold", "aten::threshold.out"], static_argnums=(1, 2))
def torch_threshold(self, threshold, value):
    return torch.nn.functional.threshold(self, threshold, value)


@register_aten(
    ["aten::threshold_backward", "aten::threshold_backward.grad_input"], static_argnums=(2,)
)
def torch_threshold_backward(grad_output, self, threshold):
    return torch.ops.aten.threshold_backward(grad_output, self, threshold)
