"""Qwen3-specific block and performance calculations."""

import torch
from block_def.qwen3 import Qwen3Config
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model


def create_qwen3_block(
    head_dim,
    hidden_size,
    num_attention_heads,
    intermediate_size,
    num_key_value_heads,
    num_hidden_layers,
    device,
    dtype,
):
    """Create and configure a Qwen3 transformer block."""
    config = Qwen3Config(
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
    )
    # requires transformers>=4.51.0
    model_config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True, **config.__dict__
    )
    block = Qwen3Model(model_config)
    block = block.to(device, dtype=dtype) if dtype is not None else block.to(device)
    block.eval()
    return block


def count_qwen3_flops(
    head_dim,
    hidden_size,
    intermediate_size,
    num_attention_heads,
    num_key_value_heads,
    num_hidden_layers,
    seq_len,
    batch_size,
):
    """Count FLOPs for a Qwen3 transformer block with GQA."""

    # Attention FLOPs
    q_proj_flops = 2 * batch_size * seq_len * hidden_size * (num_attention_heads * head_dim)

    kv_proj_flops = 2 * 2 * batch_size * seq_len * hidden_size * (num_key_value_heads * head_dim)

    attn_flops = 4 * batch_size * num_attention_heads * seq_len * seq_len * head_dim
    softmax_flops = 2 * 3 * num_attention_heads * (seq_len**2)
    attn_flops += softmax_flops

    out_proj_flops = 2 * batch_size * seq_len * (num_attention_heads * head_dim) * hidden_size

    # MLP FLOPs
    gate_proj_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
    up_proj_flops = 2 * batch_size * seq_len * hidden_size * intermediate_size
    down_proj_flops = 2 * batch_size * seq_len * intermediate_size * hidden_size

    mlp_flops = gate_proj_flops + up_proj_flops + down_proj_flops

    single_layer_flops = q_proj_flops + kv_proj_flops + attn_flops + out_proj_flops + mlp_flops
    total_flops = single_layer_flops * num_hidden_layers
    return total_flops


def get_default_config():
    """Get default configuration for Qwen3 blocks."""
    return Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
        head_dim=128,
        num_hidden_layers=1,
        vocab_size=151936,
    )


def get_preset_config(preset_name):
    """Get preset configuration for specific Qwen3 model sizes."""
    return Qwen3Config.get_preset(preset_name)


def list_presets():
    """List available Qwen3 configuration presets."""
    return ["qwen3-8b-1lyr", "qwen3-8b-4lyrs-tp4"]


def get_presets():
    """Return list of available presets for dynamic loading."""
    return list_presets()


def create_config(preset_name, batch_size, seq_len):
    """Create complete config from preset and runtime parameters."""
    base_config = Qwen3Config.get_preset(preset_name)
    return {
        "d_model": base_config.hidden_size,  # Use d_model for input compatibility
        "hidden_size": base_config.hidden_size,
        "intermediate_size": base_config.intermediate_size,
        "num_attention_heads": base_config.num_attention_heads,
        "num_key_value_heads": base_config.num_key_value_heads,
        "head_dim": base_config.head_dim,
        "num_hidden_layers": base_config.num_hidden_layers,
        "vocab_size": base_config.vocab_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create block from config dictionary."""
    return create_qwen3_block(
        config["head_dim"],
        config["hidden_size"],
        config["num_attention_heads"],
        config["intermediate_size"],
        config["num_key_value_heads"],
        config["num_hidden_layers"],
        device,
        dtype,
    )


def count_flops(config):
    """Count FLOPs for given configuration."""
    return count_qwen3_flops(
        config["head_dim"],
        config["hidden_size"],
        config["intermediate_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["num_hidden_layers"],
        config["seq_len"],
        config["batch_size"],
    )


def run_block(block, input_tensor, **kwargs):
    """Execute Qwen3 block with standard signature."""
    return block(input_tensor, **kwargs)
