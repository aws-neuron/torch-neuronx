from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 transformer block architecture."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    vocab_size: int = 151936

    @classmethod
    def get_preset(cls, preset_name: str):
        """Get preset configurations for Qwen3 model sizes."""
        presets = {
            "qwen3-8b-1lyr": cls(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                intermediate_size=12288,
                head_dim=128,
                num_hidden_layers=1,
                vocab_size=151936,
            ),
            # NOT actual tensor parallelism due to lack of collectives support.
            # This setup divides model dimensions to simulate per TP rank compute workload
            "qwen3-8b-4lyrs-tp4": cls(
                hidden_size=4096,
                num_attention_heads=8,  # 32 / 4 (heads per TP rank)
                num_key_value_heads=2,  # 8 / 4 (KV heads per TP rank)
                intermediate_size=3072,  # 12288 / 4 (FFN size per TP rank)
                head_dim=128,
                num_hidden_layers=4,
                vocab_size=37984,  # 151936 / 4 (vocab size per TP rank)
            ),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        return presets[preset_name]
