"""
GPT-OSS Example with Tensor Parallelism

Run with torchrun:
    torchrun --nproc_per_node=64 gpt_oss_example.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from models.gpt_oss import (
    GptOssModel,
    GptOssModelArgs,
    MoEArgs,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

torch.set_default_dtype(torch.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEQ_LEN = 1024
BATCH_SIZE = 1
MAX_NEW_TOKENS = 50
PROMPT_LEN = 10
torch._dynamo.config.automatic_dynamic_shapes = False


MODEL_CONFIG = GptOssModelArgs(
    dim=2816,  # Adjusted for RoPE
    n_layers=36,
    n_heads=64,
    n_kv_heads=64,  # To enable TP up to 64
    head_dim=64,
    vocab_size=201088,
    max_seq_len=1024,
    moe_inter_dim=2816,
    sliding_window_size=1024,
    rope_theta=150000.0,
    moe_args=MoEArgs(
        num_experts=4, use_grouped_mm=False
    ),  # No expert parallelism yet, so reduced num_experts
)


def apply_tp_to_gptoss(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = False,
) -> nn.Module:
    """
    Apply tensor parallelism to GPT-OSS model using DTensor.

    This follows a similar strategy to Llama3 TP but adapted for GPT-OSS:
    - Token embeddings: RowwiseParallel (shard output along sequence dim)
    - Attention projections (wq, wk, wv): ColwiseParallel
    - Attention output (wo): RowwiseParallel (shard output along sequence dim)
    - MoE layers: Simplified approach for now
    - Output projection: ColwiseParallel

    Args:
        model: GPT-OSS Transformer model
        tp_mesh: Device mesh for tensor parallel dimension
        loss_parallel: If True, keep loss computation parallel (shard output)

    Returns:
        Model with DTensor-sharded weights
    """
    logger.debug("Applying tensor parallelism to GPT-OSS model...")

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    logger.debug("Applied TP to embeddings, norm, and output projection")

    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "moe": PrepareModuleInputOutput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
                use_local_input=True,
                output_layouts=(Partial(),),
                desired_output_layouts=(Shard(1),),
            ),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        logger.debug(f"Applied TP to layer {layer_id}")

    logger.debug(f"Applied Tensor Parallelism to all {len(model.layers)} transformer blocks")

    return model


def materialize_model_empty(model, device, rank=0):
    """Materialize meta tensors to empty real tensors (uninitialized).

    Call model.init_weights() after this to properly initialize.
    """
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            local_tensor = param._local_tensor
            if local_tensor.is_meta:
                local_real = torch.empty(
                    local_tensor.shape, dtype=local_tensor.dtype, device=device
                )
                new_param = DTensor.from_local(
                    local_real,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
            else:
                new_param = param
        else:
            if param.is_meta:
                new_param = torch.empty(param.shape, dtype=param.dtype, device=device)
            else:
                new_param = param

        *path, param_name = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        parent._parameters[param_name] = new_param

    for name, buffer in list(model.named_buffers()):
        if buffer is not None and buffer.is_meta:
            if isinstance(buffer, DTensor):
                local_tensor = buffer._local_tensor
                local_real = torch.zeros(
                    local_tensor.shape, dtype=local_tensor.dtype, device=device
                )
                new_buffer = DTensor.from_local(
                    local_real,
                    device_mesh=buffer.device_mesh,
                    placements=buffer.placements,
                    run_check=False,
                )
            else:
                new_buffer = torch.zeros_like(buffer, device=device)
            *path, buffer_name = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            parent._buffers[buffer_name] = new_buffer


def generate_tokens(model, tokens, attention_mask, max_new_tokens, max_seq_len, device, rank):
    """
    Greedy token generation with fixed-length padding.

    Tokens and attention_mask are always max_seq_len in size.
    We track `current_len` to know where the valid tokens end.
    """
    current_len = attention_mask.sum(dim=-1).item()
    generated_token_ids = []

    for i in range(max_new_tokens):
        if current_len >= max_seq_len:
            if rank == 0:
                logger.debug(f"Reached max sequence length ({max_seq_len}), stopping generation")
            break

        with torch.no_grad():
            logits = model(tokens, attention_mask)

        next_token_logits = logits[:, current_len - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        generated_token_ids.append(next_token.item())

        tokens[:, current_len] = next_token.to(tokens.dtype)
        attention_mask[:, current_len] = True
        current_len += 1

        if (i + 1) % 10 == 0 and rank == 0:
            logger.debug(f"Generated {i + 1}/{max_new_tokens} tokens (seq len: {current_len})")

    return tokens, attention_mask, generated_token_ids


def setup_distributed():
    """Initialize distributed with torchrun environment."""
    dist.init_process_group("neuron")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.debug(f"Rank {rank}/{world_size}: Initialized")
    return rank, world_size


def main():
    rank, world_size = setup_distributed()

    logger.debug(f"=== Rank {rank}/{world_size}: GPT-OSS Example ===")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    max_seq_len = MODEL_CONFIG.max_seq_len

    # Phase 1: Create model on meta device (no memory allocated)
    logger.debug(f"Rank {rank}: Creating model on meta device...")
    with torch.device("meta"):
        model = GptOssModel(MODEL_CONFIG)

    # Phase 2: Apply TP (still no memory allocated)
    logger.debug(f"Rank {rank}: Applying tensor parallelism...")
    model = apply_tp_to_gptoss(model, device_mesh)

    # Phase 3: Materialize empty tensors to device
    logger.debug(f"Rank {rank}: Materializing tensors to {device}...")
    materialize_model_empty(model, device=device, rank=rank)

    # Phase 4: Initialize weights
    logger.debug(f"Rank {rank}: Initializing weights...")
    if hasattr(model, "init_weights"):
        model.init_weights()
        logger.debug(f"Rank {rank}: Called model.init_weights()")
    else:
        logger.warning(f"Rank {rank}: No init_weights() method found")

    model.eval()

    torch.manual_seed(42)

    tokens = torch.zeros(BATCH_SIZE, max_seq_len, dtype=torch.int32, device=device)
    prompt_tokens = torch.randint(
        0, MODEL_CONFIG.vocab_size, (BATCH_SIZE, PROMPT_LEN), dtype=torch.int32, device=device
    )
    tokens[:, :PROMPT_LEN] = prompt_tokens

    attention_mask = torch.zeros(BATCH_SIZE, max_seq_len, dtype=torch.bool, device=device)
    attention_mask[:, :PROMPT_LEN] = True

    # Compile
    logger.debug(f"Rank {rank}: Compiling model...")

    logger.debug(f"Rank {rank}: Starting Neuron compilation...")
    compile_start = time.time()

    compiled_model = torch.compile(model, backend="neuron", fullgraph=True)

    compile_time = time.time() - compile_start
    logger.debug(f"Rank {rank}: Compilation setup: {compile_time:.2f}s")

    # Warmup
    logger.debug(f"Rank {rank}: Running warmup...")
    with torch.no_grad():
        output = compiled_model(tokens, attention_mask)
    logger.debug(f"Rank {rank}: Warmup complete, output shape: {output.shape}")

    # Generate tokens
    logger.debug(f"Rank {rank}: Generating {MAX_NEW_TOKENS} tokens...")
    final_tokens, final_mask, generated_ids = generate_tokens(
        compiled_model, tokens, attention_mask, MAX_NEW_TOKENS, max_seq_len, device, rank
    )

    final_len = final_mask.sum(dim=-1).item()
    logger.info(f"Generated: {final_tokens}, Final length: {final_len}")

    if rank == 0:
        logger.info(f"Generated token IDs: {generated_ids}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
