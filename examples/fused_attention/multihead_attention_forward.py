#!/usr/bin/env python3
"""
Simple example showing how to use PyTorch's MultiheadAttention on Neuron devices.
This calls Neuron's flash attention kernel.
"""

import torch
import torch.nn as nn

import torch_neuronx


def main():
    # Configuration
    batch_size = 2
    seq_len = 2048  # Must be multiple of 512 for Neuron flash attention
    embed_dim = 1024
    num_heads = 8

    # Create MultiheadAttention module on Neuron
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,  # Use (batch, seq, feature) format
    ).to("neuron")

    mha.eval()  # turn on inference mode inside the module

    # Disable gradient computation for inference (required for fast path)
    with torch.no_grad():
        # Create input tensors
        query = torch.randn(batch_size, seq_len, embed_dim, device="neuron")
        key = torch.randn(batch_size, seq_len, embed_dim, device="neuron")
        value = torch.randn(batch_size, seq_len, embed_dim, device="neuron")

        # Run attention (need_weights=False for fast path)
        output, _ = mha(query, key, value, need_weights=False)

        print(f"Input shape: {query.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Device: {output.device}")

        # For self-attention, use same tensor for Q, K, V:
        x = torch.randn(batch_size, seq_len, embed_dim, device="neuron")
        output, _ = mha(x, x, x, need_weights=False)
        print(f"\nSelf-attention output shape: {output.shape}")


if __name__ == "__main__":
    main()
