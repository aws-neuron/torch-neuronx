import argparse

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torch_neuronx


class FakeDataset:
    """Fake dataset that generates random token IDs"""

    def __init__(self, seq_length=2048, vocab_size=50257):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __iter__(self):
        while True:
            # Generate random token IDs
            yield {"input_ids": torch.randint(0, self.vocab_size, (self.seq_length,))}

    def __getitem__(self, idx):
        # For indexing support
        return {"input_ids": torch.randint(0, self.vocab_size, (self.seq_length,))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2, help="Number of training iterations")
    args = parser.parse_args()

    # Custom GPT-2 config: 4 layers, 8 heads, head_dim=128, hidden_size=1024
    config = GPT2Config(
        n_layer=4,
        n_head=8,
        n_embd=1024,  # 8 heads * 128 head_dim = 1024
        vocab_size=50257,
        n_positions=2048,
    )

    # Create model and move to neuron
    model = GPT2LMHeadModel(config).to("neuron")
    # Set loss_type to avoid warning
    model.loss_type = "ForCausalLM"

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Use fake dataset
    dataset = FakeDataset(seq_length=2048, vocab_size=50257)

    # Training loop
    for i, sample in enumerate(dataset):
        if i >= args.iterations:
            break

        # Get input_ids and create attention_mask
        input_ids = sample["input_ids"].unsqueeze(0).to("neuron")  # Add batch dimension
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        print(f"Iteration {i + 1}/{args.iterations} - Forward complete, Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print(f"Iteration {i + 1}/{args.iterations} - Backward complete")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i + 1}/{args.iterations} - Optimizer step complete\n")


if __name__ == "__main__":
    main()
