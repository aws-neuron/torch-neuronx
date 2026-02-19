"""Llama-specific block and performance calculations."""

import torch
from block_def.llama import LlamaConfig, LlamaTransformerBlock
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def create_llama_block(
    hidden_size,
    num_attention_heads,
    intermediate_size,
    device,
    dtype=None,
    max_seq_len=4096,
    rope_theta=1_000_000.0,
):
    """
    Create and configure a Llama transformer block with TorchTitan RoPE.

    Args:
        hidden_size: Embedding dimension
        num_attention_heads: Number of attention heads
        intermediate_size: Feed-forward dimension
        device: Device to place model on
        dtype: Data type for model weights (optional)
        max_seq_len: Maximum sequence length for RoPE cache (default: 4096)
        rope_theta: RoPE theta parameter (default: 1_000_000.0)

    Returns:
        Configured block on specified device with specified dtype
    """
    block = LlamaTransformerBlock(
        hidden_size, num_attention_heads, intermediate_size, max_seq_len, rope_theta
    )
    block = block.to(device, dtype=dtype) if dtype is not None else block.to(device)
    block.eval()
    return block


def count_llama_flops(hidden_size, intermediate_size, num_attention_heads, seq_len, batch_size):
    """
    Count FLOPs for a single forward pass of a Llama transformer block.

    Notes on what is counted:
    - Uses the common convention that 1 FMA = 2 FLOPs.
    - Counts the dominant matmul-based ops only (projections, attention matmuls,
      and MLP). Light-weight ops like norm, activations, elementwise, reshapes,
      and softmax are excluded as they contribute comparatively little FLOPs.
    - Backward pass FLOPs are NOT included here; this is forward-only.

    Components:
    - Q, K, V projections: 3 linear layers of shape (B*S, hidden_size) x (hidden_size, hidden_size)
      => 3 * 2 * B * S * hidden_size * hidden_size FLOPs
    - Attention matmuls: (Q @ K^T) and (softmax(QK^T) @ V) each cost
      2 * B * num_attention_heads * S * S * head_dim FLOPs
      => total 4 * B * num_attention_heads * S * S * head_dim FLOPs
    - Output projection: 1 linear layer (B*S, hidden_size) x (hidden_size, hidden_size)
      => 2 * B * S * hidden_size * hidden_size FLOPs
    - Feed-forward (SwiGLU): 3 linear layers
        w1, w3: (hidden_size->intermediate_size)
        w2: (intermediate_size->hidden_size)
      => 3 * 2 * B * S * hidden_size * intermediate_size FLOPs

    Args:
        hidden_size: Embedding dimension
        intermediate_size: Feed-forward dimension
        num_attention_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Total FLOPs for one forward pass (forward-only)
    """

    # Attention FLOPs
    # Q, K, V projections: 3 * (2 * batch * seq_len * hidden_size * hidden_size)
    qkv_flops = 3 * 2 * batch_size * seq_len * hidden_size * hidden_size

    # Attention matmuls include both QK^T and (softmax(QK^T) @ V):
    # each costs 2 * B * num_attention_heads * S * S * head_dim
    head_dim = hidden_size // num_attention_heads
    attn_flops = 4 * batch_size * num_attention_heads * seq_len * seq_len * head_dim

    # Output projection: 2 * batch * seq_len * hidden_size * hidden_size
    out_proj_flops = 2 * batch_size * seq_len * hidden_size * hidden_size

    # Feed-forward FLOPs (SwiGLU has 3 linear layers)
    # w1: 2 * batch * seq_len * hidden_size * intermediate_size
    # w3: 2 * batch * seq_len * hidden_size * intermediate_size
    # w2: 2 * batch * seq_len * intermediate_size * hidden_size
    ff_flops = 3 * 2 * batch_size * seq_len * hidden_size * intermediate_size

    total_flops = qkv_flops + attn_flops + out_proj_flops + ff_flops

    return total_flops


def get_default_config():
    """Get default configuration for Llama blocks."""
    return LlamaConfig.get_default()


def get_preset_config(preset_name):
    """Get preset configuration for specific Llama model sizes."""
    return LlamaConfig.get_preset(preset_name)


def list_presets():
    """List available Llama configuration presets."""
    return ["llama-7b", "llama-13b", "llama-30b", "llama-70b"]


def get_presets():
    """Return list of available presets for dynamic loading."""
    return list_presets()


def create_config(preset_name, batch_size, seq_len):
    """Create complete config from preset and runtime parameters.

    Args:
        preset_name: Name of the preset configuration
        batch_size: Batch size for benchmarking
        seq_len: Sequence length for benchmarking

    Returns:
        Dictionary with complete configuration
    """
    base_config = LlamaConfig.get_preset(preset_name)
    return {
        "hidden_size": base_config.hidden_size,
        "intermediate_size": base_config.intermediate_size,
        "head_dim": base_config.head_dim,
        "num_attention_heads": base_config.num_attention_heads,
        "max_seq_len": base_config.max_seq_len,
        "rope_theta": base_config.rope_theta,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def create_block(config, device, dtype=None, tp_mesh=None, dp_mesh=None):
    """Create block from config dictionary.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        dtype: Data type for model weights (optional)
        tp_mesh: Tensor parallel mesh (optional)

    Returns:
        Configured block on specified device with specified dtype
    """
    block = create_llama_block(
        config["hidden_size"],
        config["num_attention_heads"],
        config["intermediate_size"],
        device,
        dtype,
        config.get("max_seq_len", 4096),
        config.get("rope_theta", 1_000_000.0),
    )
    if tp_mesh:
        return parallelize_block(block, tp_mesh)
    return block


def count_flops(config):
    """Count FLOPs for given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Total FLOPs for one forward pass
    """
    return count_llama_flops(
        config["hidden_size"],
        config["intermediate_size"],
        config["num_attention_heads"],
        config["seq_len"],
        config["batch_size"],
    )


def run_block(block, input_tensor, **kwargs):
    """Execute Llama block with standard signature."""
    return block(input_tensor, **kwargs)


def parallelize_block(transformer_block, tp_mesh):
    layer_tp_plan = {
        "ln1": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1)),
            desired_input_layouts=(Replicate()),
        ),
        "attention.q_proj": ColwiseParallel(use_local_output=False),
        "attention.k_proj": ColwiseParallel(use_local_output=False),
        "attention.v_proj": ColwiseParallel(use_local_output=False),
        "attention.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "ln2": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1)),
            desired_input_layouts=(Replicate()),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    # Custom parallelization plan for the model
    parallelize_module(
        module=transformer_block, device_mesh=tp_mesh, parallelize_plan=layer_tp_plan
    )
    # After parallelize_module, manually distribute the buffer
    if hasattr(transformer_block.attention, "rope_cache"):
        transformer_block.attention.rope_cache = distribute_tensor(
            transformer_block.attention.rope_cache,
            device_mesh=tp_mesh,
            placements=[Replicate()],
        )
    return transformer_block
