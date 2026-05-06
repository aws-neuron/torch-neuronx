"""
Test Qwen2 from HuggingFace with tensor parallelism on Neuron devices

This test validates compilation and execution of Huggingface Qwen2 model
on Neuron devices using the generate API
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default constants
BATCH_SIZE = 1
MAX_NEW_TOKENS = 16
MAX_SEQ_LEN = 128
DEFAULT_MODEL_SIZE = "7B"
DEFAULT_PROMPT = "The future of artificial intelligence is"

# Model configuration mapping
MODEL_CONFIGS = {
    "0.5B": {
        "hf_model_id": "Qwen/Qwen2-0.5B",
        "tp_degree": 1,
    },
    "7B": {
        "hf_model_id": "Qwen/Qwen2-7B",
        "tp_degree": 4,
    },
}

torch._dynamo.config.cache_size_limit = 64
torch.set_default_dtype(torch.float32)


def get_model_id(model_size: str) -> str:
    """Get HuggingFace model ID based on size."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_size]["hf_model_id"]


def get_tp_degree(model_size: str) -> int:
    """Get tensor parallelism degree for model size."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_size]["tp_degree"]


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load the HuggingFace tokenizer."""
    logger.info(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    logger.info(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    return tokenizer


def apply_qwen2_tp(model, tp_mesh):
    """
    Apply tensor parallelism to HuggingFace Qwen2 model.

    Qwen2 Architecture:
    - Attention: q_proj, k_proj, v_proj (Colwise), o_proj (Rowwise)
    - MLP (SwiGLU): gate_proj, up_proj (Colwise), down_proj (Rowwise)
    """
    layer_tp_plan = {
        # Self Attention Block
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        # MLP Block (SwiGLU)
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    for layer in model.model.layers:
        parallelize_module(layer, tp_mesh, layer_tp_plan)

    return model


def create_qwen2_model_with_tp(
    model_id: str,
    world_size: int,
    device_mesh: DeviceMesh,
    device: torch.device,
    rank: int,
) -> nn.Module:
    """
    Create Qwen2 model: CPU (with weights) -> TP -> move to device.

    This approach:
    1. Loads model with real weights to CPU
    2. Applies TP (shards the weights according to the plan)
    3. Moves to target device
    """

    logger.info(f"Rank {rank}: Loading model from {model_id}...")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    logger.info(f"Rank {rank}: Model loaded in {time.time() - load_start:.2f}s")

    if world_size > 1:
        logger.info(f"Rank {rank}: Applying tensor parallelism...")
        tp_start = time.time()
        model = apply_qwen2_tp(model, device_mesh)
        logger.info(f"Rank {rank}: TP application took {time.time() - tp_start:.3f}s")
    else:
        logger.info(f"Rank {rank}: Running without tensor parallelism (single device)")

    logger.info(f"Rank {rank}: Moving to {device}...")
    move_start = time.time()
    model = model.to(device)
    logger.info(f"Rank {rank}: Move to device took {time.time() - move_start:.3f}s")

    return model


def tokenize_prompt(
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Tokenize a prompt and pad to fixed length.

    Returns:
        Tuple of (input_ids, attention_mask, prompt_length)
    """
    logger.info(f"Tokenizing prompt: '{prompt}'")

    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(encoded)

    if prompt_length > max_seq_len:
        logger.warning(
            f"Prompt length {prompt_length} exceeds max_seq_len {max_seq_len}, truncating"
        )
        encoded = encoded[:max_seq_len]
        prompt_length = max_seq_len

    logger.info(f"Tokenized to {prompt_length} tokens (will pad to {max_seq_len})")
    logger.info(f"Token IDs: {encoded}")

    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    padding_length = max_seq_len - prompt_length
    padded_ids = encoded + [pad_token_id] * padding_length

    attention_mask = [1] * prompt_length + [0] * padding_length

    input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

    return input_ids, attention_mask, prompt_length


def run_qwen2_generation(**kwargs):
    """Test Qwen2 model generation."""
    model_size = kwargs.get("model_size", DEFAULT_MODEL_SIZE)
    max_seq_len = kwargs.get("max_seq_len", MAX_SEQ_LEN)
    prompt = kwargs.get("prompt", DEFAULT_PROMPT)

    model_id = get_model_id(model_size)
    expected_tp_degree = get_tp_degree(model_size)

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != expected_tp_degree:
        raise ValueError(
            f"Model size {model_size} requires TP degree {expected_tp_degree}, "
            f"but got world size {world_size}. "
            f"Use: torchrun --nproc-per-node {expected_tp_degree} run_qwen2.py "
            f"--model-size {model_size}"
        )

    logger.info(f"=== Rank {rank}/{world_size}: Testing Qwen2 {model_size} generation ===")

    device = torch.neuron.current_device()
    device_mesh = DeviceMesh("neuron", list(range(world_size))) if world_size > 1 else None

    tokenizer = load_tokenizer(model_id)
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    eos_token_id = tokenizer.eos_token_id

    logger.info(f"Rank {rank}: Using EOS token ID: {eos_token_id}, PAD token ID: {pad_token_id}")

    model = create_qwen2_model_with_tp(
        model_id,
        world_size,
        device_mesh,
        device,
        rank,
    )
    model.eval()

    input_ids, attention_mask, prompt_length = tokenize_prompt(
        tokenizer,
        prompt,
        device,
        max_seq_len=max_seq_len,
    )

    logger.info(f"Rank {rank}: Input shape: {input_ids.shape}, Prompt length: {prompt_length}")

    if rank == 0:
        logger.info(f"Rank {rank}: === PROMPT ===")
        logger.info(f"Rank {rank}: {prompt}")
        logger.info(f"Rank {rank}: ==============")

    # Compile model
    logger.info(f"Rank {rank}: Compiling model...")
    model.forward = torch.compile(model.forward, backend="neuron", fullgraph=True, dynamic=False)

    gen_config = GenerationConfig(
        max_length=max_seq_len,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    dist.barrier()

    # Generation
    logger.info(f"Rank {rank}: Starting generation...")
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    elapsed = time.time() - start_time

    output_ids_no_pad = output_ids[0]

    actual_length = prompt_length
    for i in range(prompt_length, len(output_ids_no_pad)):
        if output_ids_no_pad[i] == pad_token_id:
            break
        actual_length = i + 1

    output_ids_trimmed = output_ids_no_pad[:actual_length]
    output_text = tokenizer.decode(output_ids_trimmed, skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids_trimmed[prompt_length:], skip_special_tokens=True)

    num_new_tokens = actual_length - prompt_length

    if rank == 0:
        logger.info(f"Rank {rank}: === GENERATED OUTPUT ===")
        logger.info(f"Rank {rank}: Prompt: {prompt}")
        logger.info(f"Rank {rank}: Generated: {generated_text}")
        logger.info(f"Rank {rank}: Full text: {output_text}")
        logger.info(f"Rank {rank}: =========================")

        logger.info(f"Rank {rank}: Generation test PASSED")
        logger.info(f"Rank {rank}: Generated {num_new_tokens} new tokens in {elapsed:.2f}s")
        logger.info(f"Rank {rank}: Tokens per second: {num_new_tokens / elapsed:.2f}")

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Qwen2 generation with tensor parallelism on Neuron devices"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default=DEFAULT_MODEL_SIZE,
        help=f"Model size to use (default: {DEFAULT_MODEL_SIZE}). "
        f"0.5B runs on 1 device, 7B runs on 4 devices with TP.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum number of new tokens to generate (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for text generation (default: '{DEFAULT_PROMPT}')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_qwen2_generation(
        model_size=args.model_size,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        prompt=args.prompt,
    )
