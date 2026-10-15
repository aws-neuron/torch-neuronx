"""Miscellaneous operations that don't fit into other categories."""

import torch

from ..operation_registry import Preprocessing, register_aten
from ..validation import check_lerp_dtype, validate_inplace_broadcast
from .unary import _validate_float_promoting_out


@register_aten(["aten::embedding"], static_argnums=(2, 3, 4))
def torch_embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """Look up embedding rows for integer indices.

    Args:
        weights (torch.Tensor): Embedding table.
        indices (torch.Tensor): int32 or int64 indices into ``weights``.
        padding_idx (int, optional): Index whose embedding does not receive gradients.
            Defaults to ``-1``.
        scale_grad_by_freq (bool, optional): Accepted but ignored; frequency scaling is
            always disabled. Defaults to ``False``.
        sparse (bool, optional): Whether gradients with respect to ``weights`` are sparse.
            Defaults to ``False``.

    Returns:
        torch.Tensor: Embeddings selected by ``indices``.

    Raises:
        RuntimeError: If ``indices`` is not int32 or int64.
    """
    if indices.dtype not in (torch.long, torch.int):
        raise RuntimeError(
            f"Expected tensor for argument #1 'indices' to have one of the following "
            f"scalar types: Long, Int; but got {indices.dtype} instead"
        )
    return torch.embedding(
        weights, indices, padding_idx=padding_idx, scale_grad_by_freq=False, sparse=sparse
    )


def _validate_embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out=None
):
    """Validate embedding_dense_backward dtypes on raw args."""
    if indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "Expected tensor for argument #2 'indices' to have one of the following "
            "scalar types: Long, Int; but got torch.FloatTensor instead "
            "(while checking arguments for embedding_backward)"
        )


@register_aten(
    ["aten::embedding_dense_backward", "aten::embedding_dense_backward.out"],
    static_argnums=(2, 3, 4),
    validate_static_inputs=_validate_embedding_dense_backward,
)
def torch_embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out=None
):
    """Compute dense embedding gradients with respect to the weight table.

    Args:
        grad_output (torch.Tensor): Gradient of the embedding output.
        indices (torch.Tensor): int32 or int64 embedding indices.
        num_weights (int): Number of rows in the embedding table.
        padding_idx (int): Index excluded from gradient accumulation.
        scale_grad_by_freq (bool): Scale gradients by index frequency when ``True``.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Dense gradient for the embedding weights.

    Raises:
        RuntimeError: If ``indices`` is not int32 or int64.
    """
    return torch.ops.aten.embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_aten(["aten::_softmax", "aten::_softmax.out"], static_argnums=(1, 2))
def torch_softmax_internal(x, dim, half_to_float, out=None):
    """Compute the internal softmax overload.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension over which to normalize.
        half_to_float (bool): Convert the input to float32 before normalization.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Softmax probabilities.
    """
    if half_to_float:
        x = x.float()
    return torch.softmax(x, dim)


@register_aten(["aten::_log_softmax", "aten::_log_softmax.out"], static_argnums=(1, 2))
def torch_log_softmax_internal(x, dim, half_to_float, out=None):
    """Compute the internal log-softmax overload.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension over which to normalize.
        half_to_float (bool): Convert the input to float32 before normalization.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Log-softmax values.
    """
    if half_to_float:
        x = x.float()
    return torch.log_softmax(x, dim)


@register_aten(["aten::softmax"], static_argnums=(1,), static_argnames=("dtype",))
def torch_softmax(x, dim, dtype=None):
    """Compute softmax probabilities along a dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension over which to normalize.
        dtype (torch.dtype, optional): Computation and result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Softmax probabilities.
    """
    return torch.softmax(x, dim, dtype=dtype)


@register_aten(["aten::log_softmax"], static_argnums=(1,), static_argnames=("dtype",))
def torch_log_softmax(x, dim, dtype=None):
    """Compute log-softmax values along a dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension over which to normalize.
        dtype (torch.dtype, optional): Computation and result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Log-softmax values.
    """
    return torch.log_softmax(x, dim, dtype=dtype)


@register_aten(
    ["aten::_softmax_backward_data", "aten::_softmax_backward_data.out"],
    static_argnums=(2, 3),
    output_params=("grad_input",),
)
def torch_softmax_backward_data(grad_output, output, dim, input_dtype, grad_input=None):
    """Compute the softmax backward pass from output values.

    Args:
        grad_output (torch.Tensor): Gradient of the softmax output.
        output (torch.Tensor): Softmax output from the forward pass.
        dim (int): Normalized dimension.
        input_dtype (torch.dtype): Dtype of the forward input.
        grad_input (torch.Tensor, optional): Output for the gradient-input overload.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to the softmax input.
    """
    return torch.ops.aten._softmax_backward_data(grad_output, output, dim, input_dtype)


@register_aten(
    ["aten::_log_softmax_backward_data", "aten::_log_softmax_backward_data.out"],
    static_argnums=(2, 3),
    output_params=("out",),
)
def torch_log_softmax_backward_data(grad_output, output, dim, input_dtype, out=None):
    """Compute the log-softmax backward pass from output values.

    Args:
        grad_output (torch.Tensor): Gradient of the log-softmax output.
        output (torch.Tensor): Log-softmax output from the forward pass.
        dim (int): Normalized dimension.
        input_dtype (torch.dtype): Dtype of the forward input.
        out (torch.Tensor, optional): Output tensor for the ``out`` overload. Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to the log-softmax input.
    """
    return torch.ops.aten._log_softmax_backward_data(grad_output, output, dim, input_dtype)


@register_aten(
    [
        "aten::lerp.Scalar",
        "aten::lerp.Scalar_out",
        "aten::lerp.Tensor",
        "aten::lerp.Tensor_out",
    ],
    validate_static_inputs=_validate_float_promoting_out,
)
@register_aten(
    ["aten::lerp_.Scalar", "aten::lerp_.Tensor"],
    validate_static_inputs=validate_inplace_broadcast,
)
def torch_lerp(start, end, weight, out=None):
    """Linearly interpolate between two tensors.

    Args:
        start (torch.Tensor): Starting tensor.
        end (torch.Tensor): Ending tensor.
        weight (torch.Tensor or number): Interpolation weight.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Interpolated values.

    Raises:
        NotImplementedError: If ``start`` is a Boolean tensor.
        RuntimeError: If an in-place overload would expand ``start`` through broadcasting.

    Note:
        Registered ``lerp_`` overloads update ``start`` in place.
    """
    check_lerp_dtype(start, weight)
    return torch.ops.aten.lerp(start, end, weight)


def _validate_histc(
    input: torch.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0, out=None
):
    """Validate histc dtypes on raw args."""
    # ATen rejects a non-positive bin count and an inverted range before it
    # builds the histogram.
    if bins is not None and bins <= 0:
        raise RuntimeError(f"torch.histogram(): bins must be > 0, but got {bins} for dimension 0")
    if max is not None and min is not None and max < min:
        raise RuntimeError("torch.histc: max must be larger than min")
    if out is not None and input.dtype != out.dtype:
        raise RuntimeError(
            "torch.histogram: input tensor and hist tensor should have the same dtype, "
            f"but got input {input.dtype} and hist {out.dtype}"
        )


@register_aten(
    [
        "aten::histc",
        "aten::histc.out",
    ],
    # Both lists are needed because they index different signatures: argnums the
    # ATen schema histc(self, bins, min, max), argnames the compute fn's kwargs
    # (range_min/range_max are the adjusted values, hence the different spelling).
    # Without argnums, C++ PreprocessArgs tensorizes the positional min/max into
    # runtime inputs the traced graph never declared, and the launch is rejected
    # ("Unable to find input tensor input1 in input set").
    static_argnums=(1, 2, 3),
    static_argnames=("bins", "range_min", "range_max"),
    # Auto-range (min == max == 0) derives bin edges from min()/max(), so the
    # lowering cannot be cached.
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    validate_static_inputs=_validate_histc,
)
def torch_histc(
    input: torch.Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0, out=None
) -> torch.Tensor:
    """Count tensor values in equal-width histogram bins.

    Args:
        input (torch.Tensor): Values to count.
        bins (int, optional): Number of histogram bins. Defaults to ``100``.
        min (float, optional): Lower boundary. Defaults to ``0.0``.
        max (float, optional): Upper boundary. Defaults to ``0.0``.
        out (torch.Tensor, optional): Output tensor used for dtype validation. Defaults to ``None``.

    Returns:
        tuple: Lowering callable, positional arguments, and adjusted static arguments.

    Raises:
        RuntimeError: If ``bins`` is nonpositive, ``max`` is below ``min``, or ``input`` and
            ``out`` have different dtypes.

    Note:
        For nonempty input, equal boundaries are replaced by its minimum and maximum; a still-equal
        range expands by 0.5. Automatic range selection relowers every call.
    """
    # Handle auto range — CPU auto-ranges whenever min == max (not just 0==0)
    if min == max and input.numel() > 0:
        min = input.min().item()
        max = input.max().item()

    # Handle special case where min == max
    range_min = min - 0.5 if min == max else min
    range_max = max + 0.5 if min == max else max

    def efficient_histc(input, bins, range_min, range_max, out=None):
        """Count flattened input values within a fixed histogram range.

        Args:
            input (torch.Tensor): Values to count.
            bins (int): Number of histogram bins.
            range_min (float): Inclusive lower range boundary.
            range_max (float): Inclusive upper range boundary.
            out (torch.Tensor, optional): Accepted but ignored. Defaults to ``None``.

        Returns:
            torch.Tensor: Per-bin counts in the input dtype.
        """
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
def torch_threshold(self, threshold, value, out=None):
    """Replace values at or below a threshold.

    Args:
        self (torch.Tensor): Input tensor.
        threshold (number): Boundary at or below which values are replaced.
        value (number): Replacement value.
        out (torch.Tensor, optional): Unused output placeholder. Defaults to ``None``.

    Returns:
        torch.Tensor: Thresholded tensor.
    """
    return torch.nn.functional.threshold(self, threshold, value)


@register_aten(
    ["aten::threshold_backward", "aten::threshold_backward.grad_input"],
    static_argnums=(2,),
    output_params=("grad_input",),
)
def torch_threshold_backward(grad_output, self, threshold, grad_input=None):
    """Compute the threshold-operation backward pass.

    Args:
        grad_output (torch.Tensor): Gradient of the threshold output.
        self (torch.Tensor): Input from the forward pass.
        threshold (number): Forward threshold boundary.
        grad_input (torch.Tensor, optional): Output for the gradient-input overload.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Gradient with respect to ``self``.
    """
    return torch.ops.aten.threshold_backward(grad_output, self, threshold)
