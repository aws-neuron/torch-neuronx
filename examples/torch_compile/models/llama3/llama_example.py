"""
Test TorchTitan Llama3 8B generation

"""

import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from models.llama3 import Model, TransformerModelArgs
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import AutoConfig

torch.set_default_dtype(torch.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_LEN = 128
BATCH_SIZE = 1
MAX_NEW_TOKENS = 50
HF_MODEL_ID = "meta-llama/Llama-3.1-8B"


def apply_tp_to_llama3(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = False,
) -> nn.Module:
    """
    Apply tensor parallelism to Llama 3 model using DTensor.

    This follows TorchTitan's proven TP strategy:
    - Token embeddings: RowwiseParallel (shard output along sequence dim)
    - Attention projections (wq, wk, wv): ColwiseParallel
    - Attention output (wo): RowwiseParallel (shard output along sequence dim)
    - FFN layers (w1, w3): ColwiseParallel
    - FFN output (w2): RowwiseParallel (shard output along sequence dim)
    - Output projection: ColwiseParallel

    Args:
        model: Llama 3 Transformer model
        tp_mesh: Device mesh for tensor parallel dimension
        loss_parallel: If True, keep loss computation parallel (shard output)

    Returns:
        Model with DTensor-sharded weights

    Reference:
        TorchTitan's apply_tp() in models/llama3/infra/parallelize.py
    """
    logger.debug("Applying tensor parallelism to Llama 3 model...")

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
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        logger.debug(f"Applied TP to layer {layer_id}")

    logger.debug(f"Applied Tensor Parallelism to all {len(model.layers)} transformer blocks")

    return model


def hf_config_to_transformer_args(hf_config):
    """Convert HuggingFace config to TransformerModelArgs."""
    dim = hf_config.hidden_size
    base_hidden = int(2 * 4 * dim / 3)
    ffn_dim_multiplier = hf_config.intermediate_size / base_hidden if base_hidden > 0 else 1.0
    return TransformerModelArgs(
        dim=dim,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=32,  # allow TP up to 32
        vocab_size=hf_config.vocab_size,
        multiple_of=256,
        ffn_dim_multiplier=ffn_dim_multiplier,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        max_seq_len=1024,
    )


def load_model_args_from_hf(rank=0):
    """Load model args from HuggingFace."""
    logger.debug(f"Rank {rank}: Loading config from HuggingFace: {HF_MODEL_ID}")
    hf_config = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    args = hf_config_to_transformer_args(hf_config)
    logger.debug(
        f"Rank {rank}: Config loaded - dim={args.dim}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}"
    )
    return args


def materialize_model_dtensor(model, device, rank=0):
    """Materialize meta tensors to real tensors while preserving DTensor structure."""
    torch.manual_seed(42 + rank)

    def materialize_param(param, name):
        if isinstance(param, DTensor):
            local_tensor = param._local_tensor
            if local_tensor.is_meta:
                local_real = torch.empty(
                    local_tensor.shape, dtype=local_tensor.dtype, device=device
                )
                nn.init.normal_(local_real, mean=0.0, std=0.02)
                return DTensor.from_local(
                    local_real,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
            return param
        else:
            if param.is_meta:
                real_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
                nn.init.normal_(real_tensor, mean=0.0, std=0.02)
                return real_tensor
            return param

    for name, param in list(model.named_parameters()):
        new_param = materialize_param(param, name)
        *path, param_name = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        parent._parameters[param_name] = new_param

    for name, buffer in list(model.named_buffers()):
        if buffer is not None and buffer.is_meta:
            if isinstance(buffer, DTensor):
                local_tensor = buffer._local_tensor
                local_real = torch.empty(
                    local_tensor.shape, dtype=local_tensor.dtype, device=device
                )
                local_real.zero_()
                new_buffer = DTensor.from_local(
                    local_real,
                    device_mesh=buffer.device_mesh,
                    placements=buffer.placements,
                    run_check=False,
                )
            else:
                new_buffer = torch.empty_like(buffer, device=device)
                new_buffer.zero_()
            *path, buffer_name = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            parent._buffers[buffer_name] = new_buffer


def generate_tokens(model, tokens, current_len, max_new_tokens, max_seq_len, device, rank):
    """
    Greedy token generation with fixed-length padding.
    """
    generated_token_ids = []

    for i in range(max_new_tokens):
        if current_len >= max_seq_len:
            if rank == 0:
                logger.debug(f"Reached max sequence length ({max_seq_len}), stopping generation")
            break

        with torch.no_grad():
            logits = model(tokens)

        next_token_logits = logits[:, current_len - 1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        generated_token_ids.append(next_token.item())

        tokens[:, current_len] = next_token.to(tokens.dtype)
        current_len += 1

        if (i + 1) % 10 == 0 and rank == 0:
            logger.debug(f"Generated {i + 1}/{max_new_tokens} tokens (seq len: {current_len})")

    return tokens, current_len, generated_token_ids


def setup_distributed():
    """Initialize distributed with torchrun environment."""
    dist.init_process_group("neuron")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.debug(f"Rank {rank}/{world_size}: Initialized")
    return rank, world_size


def main():
    rank, world_size = setup_distributed()

    logger.debug(f"=== Rank {rank}/{world_size}: Llama3 8B Example ===")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Load config from HuggingFace
    model_args = load_model_args_from_hf(rank)
    max_seq_len = model_args.max_seq_len

    # Verify TP compatibility
    assert model_args.dim % world_size == 0, "dim not divisible by world_size"
    assert model_args.n_heads % world_size == 0, "n_heads not divisible by world_size"
    if model_args.n_kv_heads:
        assert model_args.n_kv_heads % world_size == 0, "n_kv_heads not divisible by world_size"

    # Create model on meta device
    logger.debug(f"Rank {rank}: Creating model on meta device...")
    with torch.device("meta"):
        model = Model(model_args)

    # Apply TP
    logger.debug(f"Rank {rank}: Applying tensor parallelism...")
    model = apply_tp_to_llama3(model, device_mesh)

    # Materialize weights
    logger.debug(f"Rank {rank}: Materializing weights to {device}...")
    materialize_model_dtensor(model, device=device, rank=rank)
    model.eval()

    # Prepare inputs - padded to max_seq_len
    torch.manual_seed(42 + rank)
    tokens = torch.zeros(BATCH_SIZE, max_seq_len, dtype=torch.int32, device=device)
    prompt_tokens = torch.randint(
        0, model_args.vocab_size, (BATCH_SIZE, PROMPT_LEN), dtype=torch.int32, device=device
    )
    tokens[:, :PROMPT_LEN] = prompt_tokens

    logger.debug(f"Rank {rank}: Input shape: {tokens.shape}, prompt length: {PROMPT_LEN}")

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
        output = compiled_model(tokens)
    logger.debug(f"Rank {rank}: Warmup complete, output shape: {output.shape}")

    # Generate tokens
    logger.debug(f"Rank {rank}: Generating {MAX_NEW_TOKENS} tokens...")
    final_tokens, final_len, generated_ids = generate_tokens(
        compiled_model, tokens, PROMPT_LEN, MAX_NEW_TOKENS, max_seq_len, device, rank
    )

    logger.info(f"Generated: {final_tokens}, Final length: {final_len}")

    if rank == 0:
        logger.info(f"Generated token IDs: {generated_ids}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
