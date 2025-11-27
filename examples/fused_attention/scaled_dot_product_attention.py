"""Example demonstrating scaled_dot_product_attention on Neuron

This example shows how to use F.scaled_dot_product_attention directly
for both forward and backward passes on Neuron.
"""

import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx


def run_attention_example(device):
    """Run scaled_dot_product_attention example with forward and backward passes"""

    # Model configuration
    batch_size = 2
    num_heads = 8
    seq_len = 2048
    head_dim = 128  # 1024 / 8 = 128

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print()

    # Create Q, K, V tensors directly in the shape expected by scaled_dot_product_attention
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True)

    print("Input tensors:")
    print(f"  Query shape: {q.shape}")
    print(f"  Key shape: {k.shape}")
    print(f"  Value shape: {v.shape}")

    # Forward pass using scaled_dot_product_attention
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    print("\nForward pass:")
    print(f"  Output shape: {output.shape}")

    # Compute loss and backward pass
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    print("\nBackward pass:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Query gradient shape: {q.grad.shape}")
    print(f"  Key gradient shape: {k.grad.shape}")
    print(f"  Value gradient shape: {v.grad.shape}")
    print("  Gradients computed successfully")


def main():
    # Check if Neuron device is available
    if not torch.neuron.is_available():
        print("\nError: Neuron device not available.")
        print("Please run this example on an instance with Neuron device.")
        return

    device = torch.device("neuron", 0)
    print(f"\nUsing device: {device}")

    # Run the attention example
    run_attention_example(device)


if __name__ == "__main__":
    main()
