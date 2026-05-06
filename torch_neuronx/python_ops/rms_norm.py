import torch


def rms_norm_neuron(input, normalized_shape, weight=None, eps=None):
    # eps=None is intentional: the downstream _fused_rms_norm kernel resolves it
    # based on the actual computation dtype after any device-level downcasting
    # (e.g. float64 -> float32 on neuron). Resolving eps in Python would use the
    # original input dtype, producing a value too small for the actual precision.
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
