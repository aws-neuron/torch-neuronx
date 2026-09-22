"""block-perf adapter for the Mixtral MoE dynamic-kernel benchmark (single device)."""

from block_def.mixtral_moe import MixtralConfig
from block_def.moe_dynamic import MoEDynamicBlock


def create_config(preset_name, batch_size, seq_len):
    base = MixtralConfig.get_preset(preset_name)
    return {
        "hidden_size": base.hidden_size,
        "num_attention_heads": base.num_attention_heads,
        "num_key_value_heads": base.num_key_value_heads,
        "head_dim": base.head_dim,
        "intermediate_size": base.intermediate_size,
        "num_experts": base.num_experts,
        "num_experts_per_tok": base.num_experts_per_tok,
        "max_seq_len": max(base.max_seq_len, seq_len),
        "rope_theta": base.rope_theta,
        "norm_eps": base.norm_eps,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    mc = MixtralConfig(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        intermediate_size=config["intermediate_size"],
        num_experts=config["num_experts"],
        num_experts_per_tok=config["num_experts_per_tok"],
        max_seq_len=config["max_seq_len"],
        rope_theta=config["rope_theta"],
        norm_eps=config["norm_eps"],
    )
    block = MoEDynamicBlock(mc)
    block = block.to(device, dtype=dtype) if dtype else block.to(device)
    block.eval()
    return block


def count_flops(config, active_tokens=None):
    """Forward matmul FLOPs (1 FMA = 2). ``active_tokens`` overrides the MoE-MLP
    token count (default top_k*T)."""
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["hidden_size"]
    nh = config["num_attention_heads"]
    nkv = config["num_key_value_heads"]
    hd = config["head_dim"]
    fe = config["intermediate_size"]
    ne = config["num_experts"]
    k = config["num_experts_per_tok"]

    qkv = 2 * b * s * h * (nh * hd + 2 * nkv * hd)
    attn = 4 * b * nh * s * s * hd
    out_proj = 2 * b * s * (nh * hd) * h
    router = 2 * b * s * h * ne
    routed = active_tokens if active_tokens is not None else (k * b * s)
    moe_mlp = 3 * 2 * routed * h * fe  # 3 SwiGLU matmuls over `routed` tokens
    return qkv + attn + out_proj + router + moe_mlp


def get_presets():
    return ["mixtral-8x7b", "mixtral-tiny"]


def run_block(block, input_tensor, **kwargs):
    return block(input_tensor)
