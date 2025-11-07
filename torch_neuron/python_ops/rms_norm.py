import torch


def rms_norm_neuron(input, normalized_shape, weight=None, eps=1e-5):
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
