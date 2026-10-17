"""
Simple MoE (Mixture of Experts) Model Example for PyTorch Neuron
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from self_attention import MultiHeadAttention, create_causal_mask

import torch_neuronx  # Register neuron device


class SimpleMoERouter(nn.Module):
    """Select top-k experts and compute normalized routing weights.

    Args:
        d_model (int): Hidden-state feature dimension.
        num_experts (int): Number of available experts.
        top_k (int, optional): Number of experts selected per token. Defaults to ``2``.
    """

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, hidden_states):
        # Router logits: [batch, seq_len, num_experts]
        """Route each token to its highest-scoring experts.

        Args:
            hidden_states (torch.Tensor): States shaped ``(batch, sequence, d_model)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized top-k weights and expert indices.
        """
        router_logits = self.router(hidden_states)

        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        routing_weights = f.softmax(top_k_logits, dim=-1)
        return routing_weights, top_k_indices


class SimpleMoEExpert(nn.Module):
    """Implement one gated-free feed-forward expert.

    Args:
        d_model (int): Input and output feature dimension.
        d_ff (int): Intermediate feed-forward dimension.
        dropout (float, optional): Intermediate activation dropout probability. Defaults to ``0.1``.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Simple FFN: w2(dropout(silu(w1(x))))
        """Apply the expert feed-forward network.

        Args:
            x (torch.Tensor): Input whose final dimension is ``d_model``.

        Returns:
            torch.Tensor: Expert output with the same shape as ``x``.
        """
        return self.w2(self.dropout(f.silu(self.w1(x))))


class SimpleMoELayer(nn.Module):
    """Route tokens through a collection of feed-forward experts.

    Args:
        d_model (int): Hidden-state feature dimension.
        num_experts (int, optional): Number of experts. Defaults to ``8``.
        top_k (int, optional): Number of experts selected per token. Defaults to ``2``.
        d_ff (int, optional): Expert intermediate dimension. Falsy values use
            four times ``d_model``. Defaults to ``None``.
        dropout (float, optional): Expert dropout probability. Defaults to ``0.1``.
    """

    def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        d_ff = d_ff or 4 * d_model

        # Router
        self.router = SimpleMoERouter(d_model, num_experts, top_k)

        # Experts
        self.experts = nn.ModuleList(
            [SimpleMoEExpert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, hidden_states):
        """Sum the unweighted outputs of the selected experts.

        Args:
            hidden_states (torch.Tensor): States shaped ``(batch, sequence, d_model)``.

        Returns:
            torch.Tensor: Expert outputs with the same shape as ``hidden_states``.

        Note:
            The implementation uses routing indices but does not apply routing weights.
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Get routing decisions
        routing_weights, expert_indices = self.router(hidden_states)

        # Initialize output
        final_output = torch.zeros_like(hidden_states)

        # Process each expert (simplified approach - not optimized)
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if expert_mask.any():
                expert_input = hidden_states[expert_mask]

                # Process through expert (this should work on Neuron)
                expert_output = self.experts[expert_idx](expert_input)

                # Add back to final output
                final_output[expert_mask] += expert_output

        return final_output


class SimpleMoETransformerBlock(nn.Module):
    """Combine self-attention and an MoE feed-forward residual block.

    Args:
        d_model (int): Hidden-state feature dimension.
        n_heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to ``8``.
        top_k (int, optional): Number of experts selected per token. Defaults to ``2``.
        dropout (float, optional): Attention and expert dropout probability. Defaults to ``0.1``.
    """

    def __init__(self, d_model, n_heads, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SimpleMoELayer(d_model, num_experts, top_k, dropout=dropout)

    def forward(self, x, attention_mask=None):
        """Apply attention and MoE residual transformations.

        Args:
            x (torch.Tensor): Hidden states shaped ``(batch, sequence, d_model)``.
            attention_mask (torch.Tensor, optional): Additive attention mask. A causal
                mask is created when omitted. Defaults to ``None``.

        Returns:
            torch.Tensor: Transformed hidden states with the same shape as ``x``.
        """
        batch_size, seq_length, d_model = x.shape
        device = x.device

        # Self-attention with residual
        attn_input = self.ln1(x)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_length, device)

        attn_output = self.attention(attn_input, attention_mask)
        x = x + attn_output

        # MoE FFN with residual
        moe_input = self.ln2(x)
        moe_output = self.moe(moe_input)
        x = x + moe_output

        return x


class SimpleMoEModel(nn.Module):
    """Build a decoder-only language model with MoE transformer blocks.

    Args:
        vocab_size (int, optional): Token vocabulary size. Defaults to ``32000``.
        d_model (int, optional): Hidden-state feature dimension. Defaults to ``512``.
        n_layers (int, optional): Number of transformer blocks. Defaults to ``6``.
        n_heads (int, optional): Number of attention heads. Defaults to ``8``.
        num_experts (int, optional): Experts per MoE layer. Defaults to ``8``.
        top_k (int, optional): Experts selected per token. Defaults to ``2``.
        max_seq_length (int, optional): Number of positional embeddings. Defaults to ``512``.
        dropout (float, optional): Embedding, attention, and expert dropout probability.
            Defaults to ``0.1``.
    """

    def __init__(
        self,
        vocab_size=32000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        num_experts=8,
        top_k=2,
        max_seq_length=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks with MoE
        self.blocks = nn.ModuleList(
            [
                SimpleMoETransformerBlock(d_model, n_heads, num_experts, top_k, dropout)
                for _ in range(n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """Compute token logits for an input batch.

        Args:
            input_ids (torch.Tensor): Token IDs shaped ``(batch, sequence)``.
            attention_mask (torch.Tensor, optional): Additive attention mask passed to
                every transformer block. Defaults to ``None``.

        Returns:
            torch.Tensor: Logits shaped ``(batch, sequence, vocab_size)``.
        """
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

        # Process through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


def create_moe_config(size="small"):
    """Return a predefined MoE model configuration.

    Args:
        size (str, optional): Configuration name, either ``"small"`` or ``"medium"``. Unknown
            names fall back to ``"small"``. Defaults to ``"small"``.

    Returns:
        dict[str, int]: Model dimensions for the selected size.
    """
    configs = {
        "small": {
            "vocab_size": 32000,
            "d_model": 512,
            "n_layers": 4,
            "n_heads": 8,
            "num_experts": 4,
            "top_k": 2,
            "max_seq_length": 256,
        },
        "medium": {
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 6,
            "n_heads": 12,
            "num_experts": 8,
            "top_k": 2,
            "max_seq_length": 512,
        },
    }
    return configs.get(size, configs["small"])


def test_moe_model():
    """Run the small MoE language-model example on a Neuron device."""
    print("=== Simple MoE Model Test ===\n")

    # Use neuron device
    device = torch.device("neuron")

    # Get configuration
    config = create_moe_config("small")
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create model
    model = SimpleMoEModel(**config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Test forward pass
    print("\n=== Forward Pass Test ===")
    batch_size = 2
    seq_length = 32

    # Create input
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length), device=device)
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"Output vocabulary dimension: {logits.shape[-1]}\n")

    # Test language modeling loss
    print("=== Language Modeling Loss ===")
    # Create target tokens (shifted input)
    target_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length), device=device)

    # Compute cross-entropy loss
    loss = f.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test the model
    test_moe_model()
