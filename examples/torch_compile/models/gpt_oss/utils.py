"""Utility functions for MoE implementation."""

from collections.abc import Callable

import torch

# Token group alignment size for grouped_mm operations
# For bf16: 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements)
TOKEN_GROUP_ALIGN_SIZE_M = 8


def _round_up(x: int, multiple: int) -> int:
    """Round x up to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def _permute_simple(x: torch.Tensor, num_tokens_per_expert: torch.Tensor, num_experts: int):
    """
    Simplified permutation that pads each expert's token group to alignment size.
    This is a pure PyTorch implementation without custom kernels.
    """
    # Calculate padding needed per expert
    padded_tokens_per_expert = torch.tensor(
        [_round_up(int(n.item()), TOKEN_GROUP_ALIGN_SIZE_M) for n in num_tokens_per_expert],
        dtype=torch.long,
        device=x.device,
    )

    total_padded = padded_tokens_per_expert.sum().item()

    # Create padded tensor
    x_padded = torch.zeros((total_padded, x.shape[-1]), dtype=x.dtype, device=x.device)

    # Build permutation indices by sorting tokens by expert assignment
    # This groups tokens by expert while maintaining padding
    input_offset = 0
    output_offset = 0

    for expert_idx in range(num_experts):
        n_tokens = int(num_tokens_per_expert[expert_idx].item())
        n_padded = int(padded_tokens_per_expert[expert_idx].item())

        if n_tokens > 0:
            x_padded[output_offset : output_offset + n_tokens] = x[
                input_offset : input_offset + n_tokens
            ]

        input_offset += n_tokens
        output_offset += n_padded

    return x.shape, x_padded, padded_tokens_per_expert


def _unpermute_simple(
    out: torch.Tensor, input_shape: tuple, num_tokens_per_expert: torch.Tensor, num_experts: int
):
    """
    Simplified unpermutation that removes padding added during permutation.
    """
    # Calculate padding per expert
    padded_tokens_per_expert = torch.tensor(
        [_round_up(int(n.item()), TOKEN_GROUP_ALIGN_SIZE_M) for n in num_tokens_per_expert],
        dtype=torch.long,
        device=out.device,
    )

    # Create output tensor
    out_unpermuted = torch.zeros(input_shape, dtype=out.dtype, device=out.device)

    # Unpack tokens from padded groups
    input_offset = 0
    output_offset = 0

    for expert_idx in range(num_experts):
        n_tokens = int(num_tokens_per_expert[expert_idx].item())
        n_padded = int(padded_tokens_per_expert[expert_idx].item())

        if n_tokens > 0:
            out_unpermuted[output_offset : output_offset + n_tokens] = out[
                input_offset : input_offset + n_tokens
            ]

        input_offset += n_padded
        output_offset += n_tokens

    return out_unpermuted


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    Wrapper to pad token groups for grouped_mm operations.

    This ensures that the number of tokens each expert gets is a multiple of
    TOKEN_GROUP_ALIGN_SIZE_M, which is required for efficient grouped_mm execution.

    This is a simplified pure PyTorch implementation without custom kernels.

    Args:
        func: The function to wrap (typically _run_experts_grouped_mm)

    Returns:
        Wrapped function that handles padding and unpermutation
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = w1.shape[0]

        # Permute and pad input
        input_shape, x_padded, padded_num_tokens = _permute_simple(
            x, num_tokens_per_expert, num_experts
        )

        # Run the expert computation with padded inputs
        out = func(w1, w2, w3, x_padded, padded_num_tokens)

        # Unpermute and remove padding
        out = _unpermute_simple(out, input_shape, num_tokens_per_expert, num_experts)

        return out

    return wrapper
