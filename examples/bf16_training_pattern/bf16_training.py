#!/usr/bin/env python3
"""
Example: correct bf16 training pattern on Neuron.

Demonstrates how to handle fp32 intermediate tensors (like noise scheduling
sigma values) when the model is in bf16. The key is to cast mixed-precision
intermediates back to the model's dtype before passing to bf16 layers.
"""
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import torch.nn as nn
import torch_neuronx


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(16, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 16)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return self.head(x)


def add_noise(original, noise, sigma):
    """Flow-matching noise schedule. sigma is fp32, inputs may be bf16."""
    while sigma.ndim < original.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1 - sigma) * original + sigma * noise


def main():
    device = torch.device("neuron")
    model = SimpleConvModel().to(torch.bfloat16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Device: {device}")

    for step in range(5):
        latents = torch.randn(1, 16, 1, 8, 8, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(latents)

        timestep = torch.randint(0, 1000, (1,), device=device)
        sigma = timestep.float() / 1000.0  # fp32

        noisy = add_noise(latents, noise, sigma)
        # noisy is fp32 due to type promotion -- cast it back!
        noisy = noisy.to(latents.dtype)

        pred = model(noisy)
        loss = pred.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step + 1}: loss={loss.item():.4f}")

    print("\nBF16 training completed successfully!")


if __name__ == "__main__":
    main()
