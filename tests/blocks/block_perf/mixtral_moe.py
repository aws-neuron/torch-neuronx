"""Mixtral MoE block performance benchmark.

Mixtral-8x7B: 8 SwiGLU experts, top-2 routing, GQA attention.
No TP support yet — single device only.
# TODO: Add expert parallelism (EP) with AllToAll for multi-device benchmarking.
"""

from block_def.mixtral_moe import MixtralConfig, MixtralTransformerBlock


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
        "max_seq_len": base.max_seq_len,
        "rope_theta": base.rope_theta,
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
        max_seq_len=config.get("max_seq_len", 4096),
        rope_theta=config.get("rope_theta", 1000000.0),
    )
    block = MixtralTransformerBlock(mc)
    block = block.to(device, dtype=dtype) if dtype else block.to(device)
    block.eval()
    return block


def count_flops(config):
    """Count forward-pass FLOPs for a single Mixtral block.

    Uses the convention that 1 FMA = 2 FLOPs. Counts dominant matmul-based ops
    only; norms, activations, softmax, and elementwise ops are excluded.

    Components:
    - Attention (GQA): Q/K/V projections + dot-product attention + output projection
    - Router: linear projection H -> num_experts
    - MoE MLP: top_k active experts x 3 SwiGLU matmuls per expert
    """
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    hidden_size = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    f_expert = config["intermediate_size"]
    num_experts = config["num_experts"]
    k = config["num_experts_per_tok"]

    # Attention: Q + K + V projections
    qkv_flops = (
        2 * batch_size * seq_len * hidden_size * (n_heads * head_dim + 2 * n_kv_heads * head_dim)
    )
    # Attention: QK^T and scores@V
    attn_flops = 4 * batch_size * n_heads * seq_len * seq_len * head_dim
    # Attention: output projection
    out_proj_flops = 2 * batch_size * seq_len * (n_heads * head_dim) * hidden_size
    # Router: H -> num_experts
    router_flops = 2 * batch_size * seq_len * hidden_size * num_experts
    # MoE MLP: top_k active experts x 3 SwiGLU matmuls (w1, w2, w3)
    moe_mlp_flops = k * 3 * 2 * batch_size * seq_len * hidden_size * f_expert

    return qkv_flops + attn_flops + out_proj_flops + router_flops + moe_mlp_flops


def get_presets():
    return ["mixtral-8x7b"]


def run_block(block, input_tensor, **kwargs):
    result = block(input_tensor, **kwargs)
    return result[0] if isinstance(result, tuple) else result
