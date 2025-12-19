from dataclasses import dataclass, field
from typing import Literal

from torch import nn


@dataclass
class MoEArgs:
    """Arguments for Mixture of Experts (MoE) layers."""

    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    top_k: int = 1
    use_grouped_mm: bool = True
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False


@dataclass
class GptOssModelArgs:
    """
    Data class for defining model arguments and hyperparameters.
    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        norm_eps (float): Epsilon used for RMSNorm.
        moe_args (MoEArgs): Arguments for Mixture of Experts (MoE) layers.
        swiglu_limit (float): SwiGLU activation limit.
        head_dim (int): Dimension of each attention head.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key-value heads.
        sliding_window_size (int): Size of the sliding attention window.
        attn_mask_type (str): Type of basic attention mask.
        use_flex_attn (bool): Whether to use FlexAttention. Only supports True.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
    """

    max_batch_size: int = 8
    max_seq_len: int = 131072
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 201088
    dim: int = 2880
    moe_inter_dim: int = 2880
    n_layers: int = 24
    norm_eps: float = 1e-5  # eps used for RMSNorm
    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    swiglu_limit: float = 7.0
    # Multi-Head Latent Attention (MLA)
    head_dim: int = 64
    n_heads: int = 64
    n_kv_heads: int = 8
    sliding_window_size: int = 128
    attn_mask_type: str = "causal"
    use_flex_attn: bool = True  # NOTE: gpt-oss only support FlexAttention
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 150000.0
    rope_factor: float = 32
    beta_fast: int = 32
    beta_slow: int = 1

    def update_from_config(self, **kwargs) -> None:
        """Update configuration from provided arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
