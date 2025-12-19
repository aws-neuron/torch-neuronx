"""RMSNorm-specific block and performance calculations."""

import torch
import torch.distributed as dist
import torch.nn as nn
from block_def.rms_norm import Qwen3RMSNorm, RMSNorm
from torch.distributed.tensor.parallel import SequenceParallel, parallelize_module


def create_rms_norm_block(d_model, implementation, device, dtype=None):
    """
    Create and configure an RMSNorm block.

    Args:
        d_model: Model dimension size
        implementation: Implementation type ('custom', 'torch', 'qwen3', 'qwen3_no_upcast')
        device: Device to place model on
        dtype: Data type for model weights (optional)

    Returns:
        Configured block on specified device with specified dtype
    """
    if implementation == "custom":
        block = RMSNorm(d_model)
    elif implementation == "torch":
        block = nn.RMSNorm(d_model, eps=1e-6)
    elif implementation == "qwen3":
        block = Qwen3RMSNorm(d_model, upcast=True)
    elif implementation == "qwen3_no_upcast":
        block = Qwen3RMSNorm(d_model, upcast=False)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    block = block.to(device, dtype=dtype) if dtype is not None else block.to(device)
    block.eval()
    return block


def count_rms_norm_flops(d_model, seq_len, batch_size):
    """
    Count FLOPs for a single forward pass of RMSNorm.

    RMSNorm operations:
    - x.pow(2): B * S * D FLOPs
    - mean(-1): B * S FLOPs (reduction)
    - rsqrt: B * S FLOPs
    - multiply by inv_rms: B * S * D FLOPs
    - multiply by weight: B * S * D FLOPs

    Args:
        d_model: Model dimension size
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Total FLOPs for one forward pass
    """
    # Approximate FLOP count for RMSNorm
    # pow(2) + mean + rsqrt + 2 multiplications
    flops = batch_size * seq_len * d_model * 3 + batch_size * seq_len * 2
    return flops


def get_presets():
    """Return list of available implementations as presets."""
    return ["custom", "torch", "qwen3", "qwen3_no_upcast", "rms_norm_tp"]


def create_config(preset_name, batch_size, seq_len):
    """Create complete config from preset and runtime parameters.

    Args:
        preset_name: Implementation name (used as preset)
        batch_size: Batch size for benchmarking
        seq_len: Sequence length for benchmarking

    Returns:
        Dictionary with complete configuration
    """
    return {
        "hidden_size": 4096,
        "implementation": preset_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create block from config dictionary.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        dtype: Data type for model weights (optional)
        tp_mesh: Device mesh for tensor parallelism (optional)

    Returns:
        Configured block on specified device with specified dtype
    """
    implementation = config.get("implementation", "custom")

    if implementation == "rms_norm_tp":
        block = nn.RMSNorm(config["hidden_size"], eps=1e-6)
        block = block.to(device, dtype=dtype) if dtype is not None else block.to(device)

        if tp_mesh and dist.is_initialized():
            parallelize_module(block, tp_mesh, SequenceParallel())
        return block

    # No TP - use original implementations
    return create_rms_norm_block(config["hidden_size"], implementation, device, dtype)


def count_flops(config):
    """Count FLOPs for given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Total FLOPs for one forward pass
    """
    return count_rms_norm_flops(
        config["hidden_size"],
        config["seq_len"],
        config["batch_size"],
    )


def run_block(block, input_tensor, **kwargs):
    """Execute RMSNorm block without kwargs."""
    return block(input_tensor)
