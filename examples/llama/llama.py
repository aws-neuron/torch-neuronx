import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from self_attention import create_causal_mask
from transformer_block import RMSNorm, TransformerStack

import torch_neuronx  # Register neuron device


class LlamaModel(nn.Module):
    """
    Complete Llama-style transformer model with embeddings and output layers.
    Architecture:
    Token Embeddings -> Positional Embeddings -> Transformer Blocks -> RMSNorm -> Output Projection
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        max_seq_length=2048,
        dropout=0.0,
        tie_embeddings=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learnable)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer = TransformerStack(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_rms_norm=True,
        )

        # Final normalization
        self.ln_f = RMSNorm(d_model)

        # Output projection
        if tie_embeddings:
            # Tie input and output embeddings (common in language models)
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

        # Output projection
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_length, device)

        # Pass through transformer blocks
        hidden_states = self.transformer(hidden_states, mask=attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Output projection
        if self.lm_head is None:
            # Use tied embeddings
            logits = torch_functional.linear(hidden_states, self.token_embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, top_p=0.9):
        """Simple generation method for demonstration"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                logits = self.forward(input_ids)

                # Focus on last token
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch_functional.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

                # Sample from the distribution
                probs = torch_functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check if we've hit max sequence length
                if input_ids.shape[1] >= self.max_seq_length:
                    break

        return input_ids


def create_llama_config(size="small"):
    """Create configuration for different model sizes"""
    configs = {
        "small": {
            "vocab_size": 32000,
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "d_ff": 2048,
            "max_seq_length": 512,
        },
        "medium": {
            "vocab_size": 32000,
            "d_model": 1024,
            "n_layers": 12,
            "n_heads": 16,
            "d_ff": 4096,
            "max_seq_length": 1024,
        },
        "large": {
            "vocab_size": 32000,
            "d_model": 2048,
            "n_layers": 24,
            "n_heads": 32,
            "d_ff": 8192,
            "max_seq_length": 2048,
        },
    }
    return configs.get(size, configs["small"])


def demo_llama_model():
    """Demonstrate the complete Llama model"""

    # Use neuron device
    device = torch.device("neuron")

    # Get configuration
    config = create_llama_config("small")

    print("=== Llama Model Demo ===\n")
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create model
    model = LlamaModel(**config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}\n")

    # Parameter breakdown
    print("Parameter breakdown:")
    embedding_params = sum(p.numel() for p in model.token_embeddings.parameters())
    pos_embedding_params = sum(p.numel() for p in model.position_embeddings.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    ln_f_params = sum(p.numel() for p in model.ln_f.parameters())

    print(f"  Token embeddings: {embedding_params:,}")
    print(f"  Position embeddings: {pos_embedding_params:,}")
    print(f"  Transformer blocks: {transformer_params:,}")
    print(f"  Final layer norm: {ln_f_params:,}")
    if model.lm_head is not None:
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        print(f"  Output projection: {lm_head_params:,}")
    else:
        print("  Output projection: (tied with embeddings)")
    print()

    # Test forward pass
    print("=== Forward Pass Test ===")
    batch_size = 2
    seq_length = 20

    # Create random input tokens
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length), device=device)
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Output vocabulary dimension: {logits.shape[-1]}\n")

    # Test language modeling loss
    print("=== Language Modeling Loss ===")
    # Create target tokens (shifted input)
    target_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length), device=device)

    # Compute cross-entropy loss
    loss = torch_functional.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
    print(f"Loss: {loss.item():.4f}")

    # Test gradient flow
    print("\n=== Gradient Flow Test ===")
    loss.backward()

    # Check some gradients
    for name, param in model.named_parameters():
        if param.grad is not None and "embeddings" in name:
            grad_norm = param.grad.norm().item()
            print(f"{name}: shape {param.shape}, grad norm {grad_norm:.4f}")

    # Test generation
    print("\n=== Generation Test ===")
    model.eval()

    # Start with a single token
    prompt = torch.tensor([[1]], device=device)  # Token ID 1 as prompt
    print(f"Prompt shape: {prompt.shape}")

    # Generate sequence
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated token IDs: {generated[0].tolist()}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    demo_llama_model()
