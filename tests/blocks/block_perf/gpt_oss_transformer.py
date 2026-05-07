"""GPT-OSS Transformer block perf test with TP/FSDP support.

TP plan adapted from: NeuronTorchTitan/torchtitan/experiments/gpt_oss/infra/parallelize.py
"""

import re

import torch
from block_def.gpt_oss import GptOssModelArgs, GptOssTransformerBlock
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def _get_tp_plan():
    """TP plan from NeuronTorchTitan gpt_oss/infra/parallelize.py."""
    return {
        "block.attention_norm": SequenceParallel(),
        "block.attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "block.attention.wq": ColwiseParallel(use_local_output=False),
        "block.attention.wk": ColwiseParallel(use_local_output=False),
        "block.attention.wv": ColwiseParallel(use_local_output=False),
        "block.attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "block.ffn_norm": SequenceParallel(),
        "block.feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "block.feed_forward.w1": ColwiseParallel(),
        "block.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "block.feed_forward.w3": ColwiseParallel(),
    }


def _apply_fsdp(block, dp_mesh):
    """Apply FSDP to the block."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=True)


def _apply_parallelism(block, tp_mesh, dp_mesh, parallelism_type):
    """Apply parallelism based on type."""
    if "tp" in parallelism_type:
        parallelize_module(block, tp_mesh, _get_tp_plan())
    if "fsdp" in parallelism_type:
        _apply_fsdp(block, dp_mesh)


# Extracts parallelism suffix from preset names. Matches:
#   -singlecore  : single core execution
#   -tp_fsdp     : tensor parallel + FSDP combined
#   -tp or -tp4  : tensor parallel (optional degree, e.g., tp4, tp8)
#   -fsdp or -fsdp4 : FSDP (optional degree, e.g., fsdp2, fsdp16)
PARALLELISM_PATTERN = re.compile(r"-(singlecore|tp_fsdp|tp\d*|fsdp\d*)$")


def create_config(preset_name, batch_size, seq_len):
    """Create config from preset."""
    match = PARALLELISM_PATTERN.search(preset_name)
    singlecore = match and match.group(1) == "singlecore"
    parallelism = "none" if (not match or singlecore) else match.group(1)
    base_preset = PARALLELISM_PATTERN.sub("", preset_name)

    args = GptOssModelArgs.from_preset(base_preset)

    config = {
        "hidden_size": args.dim,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "hidden_dim": args.hidden_dim,
        "max_seq_len": args.max_seq_len,
        "rope_theta": args.rope_theta,
        "norm_eps": args.norm_eps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parallelism": parallelism,
    }

    return config


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create GPT-OSS block with optional parallelization."""
    args = GptOssModelArgs(
        dim=config["hidden_size"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        head_dim=config["head_dim"],
        hidden_dim=config["hidden_dim"],
        max_seq_len=config["max_seq_len"],
        rope_theta=config["rope_theta"],
        norm_eps=config["norm_eps"],
    )

    block = GptOssTransformerBlock(args)
    block = block.to(device, dtype=dtype) if dtype else block.to(device)
    block.eval()

    parallelism = config.get("parallelism", "none")
    if parallelism != "none" and (tp_mesh is not None or dp_mesh is not None):
        _apply_parallelism(block, tp_mesh, dp_mesh, parallelism)

    return block


def count_flops(config):
    """Count FLOPs for GPT-OSS block (GQA attention + SwiGLU FFN)."""
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    dim = config["hidden_size"]
    hidden_dim = config["hidden_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]

    # QKV projections
    q_flops = 2 * batch_size * seq_len * dim * (n_heads * head_dim)
    kv_flops = 2 * 2 * batch_size * seq_len * dim * (n_kv_heads * head_dim)

    # Attention matmuls (QK^T and softmax@V)
    attn_flops = 4 * batch_size * n_heads * seq_len * seq_len * head_dim

    # Output projection
    out_proj_flops = 2 * batch_size * seq_len * (n_heads * head_dim) * dim

    # SwiGLU FFN: w1, w3 (up), w2 (down)
    ff_flops = 3 * 2 * batch_size * seq_len * dim * hidden_dim

    return q_flops + kv_flops + attn_flops + out_proj_flops + ff_flops


def get_presets():
    """Return available presets."""
    return [
        "gpt-oss-20b",
        "gpt-oss-20b-tp",
        "gpt-oss-20b-fsdp",
        "gpt-oss-20b-tp_fsdp",
        "gpt-oss-20b-singlecore",  # Pure compute baseline, no distributed
    ]


def run_block(block, input_tensor, **kwargs):
    """Execute block."""
    return block(input_tensor)
