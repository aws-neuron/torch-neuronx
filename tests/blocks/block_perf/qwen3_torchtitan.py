# block_perf/qwen3_torchtitan.py
"""Qwen3 TorchTitan with parallelism variants."""

import torch
from block_def.qwen3_torchtitan import Qwen3ModelArgs, Qwen3TransformerBlock
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def _get_tp_plan():
    """Get TP parallelization plan."""
    return {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate(), None),
            desired_input_layouts=(Replicate(), Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        "attention.q_norm": SequenceParallel(sequence_dim=2),
        "attention.k_norm": SequenceParallel(sequence_dim=2),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }


def _apply_fsdp(block, dp_mesh):
    """Apply FSDP to the block."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    fully_shard(
        block.block,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )


def _apply_parallelism(block, tp_mesh, dp_mesh, parallelism_type):
    """Apply parallelism based on type."""
    if parallelism_type == "tp_fsdp":
        parallelize_module(block.block, tp_mesh, _get_tp_plan())
        _apply_fsdp(block, dp_mesh)
    elif parallelism_type == "tp":
        parallelize_module(block.block, tp_mesh, _get_tp_plan())
    elif parallelism_type == "fsdp" and dp_mesh:
        _apply_fsdp(block, dp_mesh)
    elif parallelism_type == "none":
        # No parallelism applied for single device
        pass


def create_config(preset_name, batch_size, seq_len):
    """Create config from preset."""
    config = {
        "hidden_size": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 12288,
        "n_layers": 1,
        "head_dim": 128,
        "vocab_size": 16384,
        "max_seq_len": 4096,
        "rope_theta": 1000000,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    if "tp_fsdp" in preset_name:
        config["parallelism"] = "tp_fsdp"
    elif "tp" in preset_name:
        config["parallelism"] = "tp"
    elif "fsdp" in preset_name:
        config["parallelism"] = "fsdp"
    else:
        config["parallelism"] = "none"

    return config


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create Qwen3 block with parallelization."""
    model_args = Qwen3ModelArgs(
        dim=config["hidden_size"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        head_dim=config["head_dim"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        rope_theta=config["rope_theta"],
    )

    block = Qwen3TransformerBlock(model_args)
    block = block.to(device, dtype=dtype) if dtype else block.to(device)
    block.eval()

    # Only apply parallelism if tp_mesh or dp_mesh is provided (similar to llama)
    parallelism = config.get("parallelism", "none")
    if parallelism != "none" and (tp_mesh is not None or dp_mesh is not None):
        _apply_parallelism(block, tp_mesh, dp_mesh, parallelism)

    return block


def count_flops(config):
    """Count FLOPs for Qwen3 block."""
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    dim = config["hidden_size"]
    hidden_dim = config["hidden_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]

    qkv_flops = 2 * batch_size * seq_len * dim * (n_heads * head_dim + 2 * n_kv_heads * head_dim)
    attn_flops = 4 * batch_size * n_heads * seq_len * seq_len * head_dim
    out_proj_flops = 2 * batch_size * seq_len * (n_heads * head_dim) * dim
    ff_flops = 3 * 2 * batch_size * seq_len * dim * hidden_dim

    return qkv_flops + attn_flops + out_proj_flops + ff_flops


def get_presets():
    """Return available presets."""
    return [
        "qwen3-8b",
        "qwen3-8b-tp",
        "qwen3-8b-fsdp",
        "qwen3-8b-tp_fsdp",
        "qwen3-8b-singlecore",  # Pure compute baseline, no distributed
    ]


def run_block(block, input_tensor, **kwargs):
    """Execute block."""
    return block(input_tensor)
