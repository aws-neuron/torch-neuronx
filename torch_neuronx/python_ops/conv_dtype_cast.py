"""
Auto-cast convolution input dtype to match weight dtype on Neuron.

PyTorch's Conv1d/Conv2d/Conv3d modules require input, weight, and bias to
have the same dtype. When models are cast to bfloat16 via model.to(torch.bfloat16),
fp32 intermediate tensors (common in diffusion noise scheduling) cause:

    RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same

This module registers a global forward pre-hook on all Conv modules that
auto-casts the input to match the weight dtype, matching the behavior that
users expect when migrating from XLA_USE_BF16 to explicit bf16 casting.

Only floating-point dtype mismatches are auto-cast (e.g., fp32 -> bf16).
Integer dtype mismatches are left unchanged to preserve PyTorch error behavior.
"""

import torch
import torch.nn as nn


def _conv_dtype_cast_pre_hook(module: nn.Module, args: tuple) -> tuple:
    """Cast conv input to match weight dtype if they differ."""
    if not args:
        return args

    input_tensor = args[0]
    if not isinstance(input_tensor, torch.Tensor):
        return args

    weight = module.weight
    if (
        input_tensor.dtype != weight.dtype
        and input_tensor.is_floating_point()
        and weight.is_floating_point()
        and hasattr(input_tensor, "device")
        and str(input_tensor.device).startswith("neuron")
    ):
        return (input_tensor.to(weight.dtype),) + args[1:]
    return args


_HOOK_HANDLE = None


def apply_conv_dtype_cast_hook():
    """Register a global forward pre-hook on all Conv modules.

    This ensures that fp32 inputs to bf16 Conv layers are auto-cast,
    preventing the "Input type (float) and bias type (c10::BFloat16)
    should be the same" error on Neuron.

    Safe to call multiple times -- only registers the hook once.
    """
    global _HOOK_HANDLE
    if _HOOK_HANDLE is not None:
        return

    _HOOK_HANDLE = nn.modules.module.register_module_forward_pre_hook(
        _conv_dtype_cast_pre_hook
    )


def remove_conv_dtype_cast_hook():
    """Remove the global conv dtype cast hook."""
    global _HOOK_HANDLE
    if _HOOK_HANDLE is not None:
        _HOOK_HANDLE.remove()
        _HOOK_HANDLE = None
