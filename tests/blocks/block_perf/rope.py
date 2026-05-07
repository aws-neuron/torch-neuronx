from block_def.rope import RoPEBlock, RoPEBlockHF


def create_rope_block(config, device, dtype=None):
    implementation = config.get("implementation", "torchtitan")

    if implementation == "torchtitan":
        block = RoPEBlock(config["head_dim"], config["max_seq_len"], config["rope_theta"])
    elif implementation == "hf":
        block = RoPEBlockHF(config["head_dim"], config["max_seq_len"], config["rope_theta"])
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    return block.to(device, dtype=dtype) if dtype else block.to(device)


def count_rope_flops(head_dim, seq_len, batch_size, num_attention_heads, num_key_value_heads):
    """
    Count FLOPs for RoPE application on query and key tensors.

    For each tensor (xq, xk):
    - Element-wise multiply with cos: batch_size * seq_len * num_heads * head_dim FLOPs
    - Element-wise multiply with sin: batch_size * seq_len * num_heads * head_dim FLOPs
    - Element-wise add: batch_size * seq_len * num_heads * head_dim FLOPs
    Total per tensor: 3 * batch_size * seq_len * num_heads * head_dim
    For both xq and xk: 6 * batch_size * seq_len * num_heads * head_dim
    """
    q_flops = 3 * batch_size * seq_len * num_attention_heads * head_dim
    k_flops = 3 * batch_size * seq_len * num_key_value_heads * head_dim
    return q_flops + k_flops


def get_presets():
    return [
        "rope-gqa-32-8-torchtitan",
        "rope-gqa-8-2-torchtitan",
        "rope-gqa-32-8-hf",
        "rope-gqa-8-2-hf",
    ]


def create_config(preset_name, batch_size, seq_len):
    # Parse preset name to extract implementation
    if preset_name.endswith("-torchtitan"):
        implementation = "torchtitan"
        base_preset = preset_name.replace("-torchtitan", "")
    elif preset_name.endswith("-hf"):
        implementation = "hf"
        base_preset = preset_name.replace("-hf", "")
    else:
        # Default to torchtitan for backward compatibility
        implementation = "torchtitan"
        base_preset = preset_name

    presets = {
        "rope-gqa-32-8": {
            "head_dim": 128,
            "max_seq_len": 4096,
            "rope_theta": 1_000_000.0,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        "rope-gqa-8-2": {
            "head_dim": 128,
            "max_seq_len": 4096,
            "rope_theta": 1_000_000.0,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
    }

    config = presets[base_preset].copy()
    config.update(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "implementation": implementation,
        }
    )
    return config


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    return create_rope_block(config, device, dtype)


def count_flops(config):
    return count_rope_flops(
        config["head_dim"],
        config["seq_len"],
        config["batch_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
    )


def run_block(block, input_tensor, **kwargs):
    """Execute RoPE block with tuple inputs."""
    return block(*input_tensor)
