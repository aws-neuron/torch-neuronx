import torch


def rms_norm_neuron(input, normalized_shape, weight=None, eps=None):
    # eps=None is intentional: the downstream _fused_rms_norm kernel resolves it
    # based on the actual computation dtype after any device-level downcasting
    # (e.g. float64 -> float32 on neuron). Resolving eps in Python would use the
    # original input dtype, producing a value too small for the actual precision.
    """Apply RMS normalization and preserve the input dtype.

    Args:
        input (torch.Tensor): Input tensor to normalize.
        normalized_shape (Sequence[int]): Trailing dimensions to normalize.
        weight (torch.Tensor, optional): Elementwise scaling tensor. Defaults to ``None``.
        eps (float | None, optional): Stability epsilon; ``None`` defers selection to the fused
            kernel. Defaults to ``None``.

    Returns:
        torch.Tensor: Normalized output cast back to the input dtype.

    Note:
        Half or bfloat16 input with a differently typed weight computes in float32. Deferring
        ``eps=None`` lets the kernel use the actual post-downcast computation dtype.
    """
    original_dtype = input.dtype

    if (
        weight is not None
        and weight.dtype != input.dtype
        and input.dtype in (torch.float16, torch.bfloat16)
    ):
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
    result = torch.ops.aten._fused_rms_norm(input, normalized_shape, weight, eps)
    output = result[0].to(original_dtype)
    return output
