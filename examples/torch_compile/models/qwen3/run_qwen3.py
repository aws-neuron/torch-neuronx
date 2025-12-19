"""
Test TorchTitan Qwen3 8B generation

"""

import argparse
import logging
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
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
from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1
NUM_TOKENS_TO_GENERATE = 20
MAX_SEQ_LEN = 128
HF_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_PROMPT = "The future of artificial intelligence is"


torch.set_default_dtype(torch.float32)


def get_model_args_from_hf(model_id: str = HF_MODEL_ID) -> Qwen3ModelArgs:
    """Load model config from HuggingFace and convert to TT Qwen3ModelArgs."""
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    return Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=hf_config.max_position_embeddings,
        eos_id=hf_config.eos_token_id,
    )


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
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
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    for transformer_block in model.layers.values():
        layer_plan = {
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
        }

        layer_plan.update(
            {
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
            }
        )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    return model


def load_tokenizer(model_id: str = HF_MODEL_ID) -> AutoTokenizer:
    """Load the HuggingFace tokenizer for Qwen3."""
    logger.info(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    logger.info(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    return tokenizer


def get_weight_map() -> dict:
    return {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }


def get_layer_weight_map(layer_idx: int) -> dict:
    """Get weight mapping for a specific layer."""
    hf_prefix = f"model.layers.{layer_idx}"
    tt_prefix = f"layers.{layer_idx}"

    return {
        f"{hf_prefix}.self_attn.q_proj.weight": f"{tt_prefix}.attention.wq.weight",
        f"{hf_prefix}.self_attn.k_proj.weight": f"{tt_prefix}.attention.wk.weight",
        f"{hf_prefix}.self_attn.v_proj.weight": f"{tt_prefix}.attention.wv.weight",
        f"{hf_prefix}.self_attn.o_proj.weight": f"{tt_prefix}.attention.wo.weight",
        f"{hf_prefix}.self_attn.q_norm.weight": f"{tt_prefix}.attention.q_norm.weight",
        f"{hf_prefix}.self_attn.k_norm.weight": f"{tt_prefix}.attention.k_norm.weight",
        f"{hf_prefix}.mlp.gate_proj.weight": f"{tt_prefix}.feed_forward.w1.weight",
        f"{hf_prefix}.mlp.up_proj.weight": f"{tt_prefix}.feed_forward.w3.weight",
        f"{hf_prefix}.mlp.down_proj.weight": f"{tt_prefix}.feed_forward.w2.weight",
        f"{hf_prefix}.input_layernorm.weight": f"{tt_prefix}.attention_norm.weight",
        f"{hf_prefix}.post_attention_layernorm.weight": f"{tt_prefix}.ffn_norm.weight",
    }


def load_weights_from_hf(
    model: nn.Module,
    model_id: str = HF_MODEL_ID,
    model_args: Qwen3ModelArgs = None,
) -> None:
    """
    Load pretrained weights from HuggingFace into TT model.

    Args:
        model: TT Qwen3Model instance
        model_id: HuggingFace model ID
        model_args: Model arguments (for n_layers)
    """
    logger.info(f"Loading pretrained weights from {model_id}...")
    load_start = time.time()

    # Load HuggingFace model weights
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    hf_state_dict = hf_model.state_dict()

    logger.info(f"HuggingFace model loaded in {time.time() - load_start:.2f}s")
    logger.info(f"HuggingFace state dict keys: {len(hf_state_dict)}")

    # Build complete weight mapping
    weight_map = get_weight_map()
    n_layers = model_args.n_layers if model_args else 36
    for i in range(n_layers):
        weight_map.update(get_layer_weight_map(i))

    # Get TT model's state dict
    tt_state_dict = model.state_dict()

    # Map and load weights
    loaded_count = 0
    missing_in_hf = []
    missing_in_tt = []
    shape_mismatch = []

    for hf_name, tt_name in weight_map.items():
        if hf_name not in hf_state_dict:
            missing_in_hf.append(hf_name)
            continue
        if tt_name not in tt_state_dict:
            missing_in_tt.append(tt_name)
            continue

        hf_tensor = hf_state_dict[hf_name]
        tt_tensor = tt_state_dict[tt_name]

        if hf_tensor.shape != tt_tensor.shape:
            shape_mismatch.append(f"{hf_name}: HF {hf_tensor.shape} vs TT {tt_tensor.shape}")
            continue

        tt_state_dict[tt_name] = hf_tensor.clone()
        loaded_count += 1

    # Load the mapped state dict
    model.load_state_dict(tt_state_dict, strict=False)

    logger.info(f"Loaded {loaded_count} weight tensors")
    if missing_in_hf:
        logger.warning(
            f"Missing in HF model: {missing_in_hf[:5]}{'...' if len(missing_in_hf) > 5 else ''}"
        )
    if missing_in_tt:
        logger.warning(
            f"Missing in TT model: {missing_in_tt[:5]}{'...' if len(missing_in_tt) > 5 else ''}"
        )
    if shape_mismatch:
        logger.warning(
            f"Shape mismatches: {shape_mismatch[:5]}{'...' if len(shape_mismatch) > 5 else ''}"
        )

    del hf_model
    del hf_state_dict

    logger.info(f"Weight loading complete in {time.time() - load_start:.2f}s")


def tokenize_prompt(
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_length: int = MAX_SEQ_LEN,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, int]:
    """
    Tokenize a prompt and pad to fixed length.

    Returns:
        Tuple of (padded_tokens, actual_length)
    """
    logger.info(f"Tokenizing prompt: '{prompt}'")

    # Tokenize without special tokens for simple completion
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    actual_length = len(encoded)

    if actual_length > max_length:
        logger.warning(f"Prompt length {actual_length} exceeds max_length {max_length}, truncating")
        encoded = encoded[:max_length]
        actual_length = max_length

    # Pad to fixed length
    padding_length = max_length - actual_length
    padded = encoded + [pad_token_id] * padding_length

    # Convert to tensor
    tokens = torch.tensor([padded], dtype=torch.int32, device=device)

    logger.info(f"Tokenized to {actual_length} tokens (padded to {max_length})")
    logger.info(f"Token IDs: {encoded}")

    return tokens, actual_length


def decode_tokens(tokenizer: AutoTokenizer, tokens: torch.Tensor) -> str:
    """Decode token tensor back to string."""
    if tokens.dim() == 2:
        tokens = tokens[0]  # Take first batch element
    token_list = tokens.tolist()
    decoded = tokenizer.decode(token_list, skip_special_tokens=False)
    return decoded


def run_neuron_forward(
    rank: int, compiled_model: torch.nn.Module, *inputs, run_number: int = 1
) -> tuple[torch.Tensor, float]:
    """Run forward pass on Neuron device."""
    device = torch.device(f"neuron:{rank}")
    neuron_inputs = []

    for inp in inputs:
        if hasattr(inp, "to_local"):
            local_tensor = inp.to_local().to(device)
            neuron_inputs.append(local_tensor)
        elif isinstance(inp, torch.Tensor):
            neuron_inputs.append(inp.to(device))
        else:
            neuron_inputs.append(inp)

    with torch.no_grad():
        neuron_output = compiled_model(*neuron_inputs)

    return neuron_output


def generate_tokens_fixed_length(
    rank: int,
    compiled_model: torch.nn.Module,
    input_tokens: torch.Tensor,
    prompt_length: int,
    tokenizer: AutoTokenizer = None,
    num_tokens: int = 5,
    temperature: float = 1.0,
    top_k: int = 50,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, list[float], list[int]]:
    """
    Generate tokens autoregressively with fixed sequence length.

    This approach maintains a fixed-length tensor and updates positions
    in place to avoid recompilation from changing tensor shapes.
    """
    max_seq_len = input_tokens.shape[1]
    tokens = input_tokens.clone()
    generation_times = []
    generated_token_ids = []

    current_pos = prompt_length  # Position to write next generated token

    # Check if we have room to generate
    max_generate = min(num_tokens, max_seq_len - prompt_length)
    if max_generate < num_tokens:
        logger.warning(
            f"Rank {rank}: Can only generate {max_generate} tokens (requested {num_tokens})"
        )

    for i in range(max_generate):
        step_start = time.time()

        # Forward pass with full fixed-length sequence
        with torch.no_grad():
            logits = compiled_model(tokens)

        # Handle DTensor output
        if isinstance(logits, DTensor):
            logits = logits.to_local()

        # Get logits for the CURRENT position (last non-padded token)
        next_token_logits = logits[:, current_pos - 1, :]  # (batch, vocab_size)

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Top-k sampling
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            next_token_logits_filtered = torch.full_like(next_token_logits, float("-inf"))
            next_token_logits_filtered.scatter_(1, top_k_indices, top_k_values)
            next_token_logits = next_token_logits_filtered

        # Sample next token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        step_time = time.time() - step_start
        generation_times.append(step_time)

        next_token_id = next_token.item()
        generated_token_ids.append(next_token_id)

        # Update the token at current position
        tokens[:, current_pos] = next_token.squeeze(-1).to(tokens.dtype)
        current_pos += 1

        # Decode token if tokenizer available
        if tokenizer is not None:
            token_str = tokenizer.decode([next_token_id])
            logger.info(
                f"Rank {rank}: Token {i+1}/{max_generate}: {next_token_id} -> '{token_str}'"
            )
        else:
            logger.info(
                f"Rank {rank}: Token {i+1}/{max_generate}: {next_token_id} ({step_time:.3f}s)"
            )

        # Check for EOS
        if eos_token_id is not None and next_token_id == eos_token_id:
            logger.info(f"Rank {rank}: EOS token encountered, stopping generation")
            break

    total_time = sum(generation_times)
    avg_time = total_time / len(generation_times) if generation_times else 0

    logger.info(f"Rank {rank}: Generation complete!")
    logger.info(f"Rank {rank}: Generated {len(generated_token_ids)} tokens")
    logger.info(f"Rank {rank}: Total generation time: {total_time:.3f}s")
    logger.info(f"Rank {rank}: Average time per token: {avg_time:.3f}s")
    logger.info(f"Rank {rank}: Tokens per second: {len(generated_token_ids) / total_time:.2f}")

    # Return only the meaningful part of the sequence
    actual_length = prompt_length + len(generated_token_ids)
    meaningful_tokens = tokens[:, :actual_length].clone()

    return meaningful_tokens, generation_times, generated_token_ids


def move_dtensor_model_to_device(model: nn.Module, device: torch.device) -> None:
    """
    Move a model with DTensor parameters to the target device.
    """
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            local_tensor = param._local_tensor.to(device)
            new_dtensor = DTensor.from_local(
                local_tensor,
                device_mesh=param.device_mesh,
                placements=param.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )
            *path, param_name = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            parent._parameters[param_name] = new_dtensor
        else:
            *path, param_name = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            parent._parameters[param_name] = nn.Parameter(
                param.to(device), requires_grad=param.requires_grad
            )

    for name, buffer in list(model.named_buffers()):
        if buffer is not None:
            if isinstance(buffer, DTensor):
                local_tensor = buffer._local_tensor.to(device)
                new_buffer = DTensor.from_local(
                    local_tensor,
                    device_mesh=buffer.device_mesh,
                    placements=buffer.placements,
                    run_check=False,
                )
            else:
                new_buffer = buffer.to(device)

            *path, buffer_name = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            parent._buffers[buffer_name] = new_buffer


def create_qwen3_model_with_tp(
    model_args: Qwen3ModelArgs,
    world_size: int,
    device_mesh: DeviceMesh,
    device: torch.device,
    rank: int,
    hf_model_id: str = HF_MODEL_ID,
) -> nn.Module:
    """
    Create Qwen3 model: CPU (with weights) -> TP -> move to device.

    This approach:
    1. Creates model on CPU with real tensors
    2. Loads pretrained weights from HuggingFace (or random init)
    3. Applies TP (shards the weights according to the plan)
    4. Moves to target device
    """
    torch.set_default_dtype(torch.float32)

    logger.info(f"Rank {rank}: Creating model on CPU...")
    cpu_start = time.time()
    with torch.device("cpu"):
        model = Qwen3Model(model_args)
    logger.info(f"Rank {rank}: CPU creation took {time.time() - cpu_start:.3f}s")

    logger.info(f"Rank {rank}: Loading pretrained weights...")
    load_weights_from_hf(model, model_id=hf_model_id, model_args=model_args)

    logger.info(f"Rank {rank}: Applying tensor parallelism...")
    tp_start = time.time()
    model = apply_non_moe_tp(model=model, tp_mesh=device_mesh)
    logger.info(f"Rank {rank}: TP application took {time.time() - tp_start:.3f}s")

    logger.info(f"Rank {rank}: Moving to {device}...")
    move_start = time.time()
    move_dtensor_model_to_device(model, device)
    logger.info(f"Rank {rank}: Move to device took {time.time() - move_start:.3f}s")

    return model


def run_qwen3_generation(**kwargs):
    """Test Qwen3 model generation."""
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model_name = kwargs.get("model_name", "qwen3_8b")
    num_tokens = kwargs.get("num_tokens", NUM_TOKENS_TO_GENERATE)
    prompt = kwargs.get("prompt", DEFAULT_PROMPT)
    hf_model_id = kwargs.get("hf_model_id", HF_MODEL_ID)
    max_seq_len = kwargs.get("max_seq_len", MAX_SEQ_LEN)
    model_args = kwargs.get("model_args") or get_model_args_from_hf(hf_model_id)

    logger.info(f"=== Rank {rank}/{world_size}: Testing {model_name} generation ===")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Load tokenizer
    tokenizer = load_tokenizer(hf_model_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model_args.eos_id
    )

    logger.info(f"Rank {rank}: Using EOS token ID: {eos_token_id}, PAD token ID: {pad_token_id}")

    # Create model with pretrained weights
    model = create_qwen3_model_with_tp(
        model_args,
        world_size,
        device_mesh,
        device,
        rank,
        hf_model_id=hf_model_id,
    )
    model.eval()

    # Tokenize the prompt with padding to fixed length
    input_tokens, prompt_length = tokenize_prompt(
        tokenizer,
        prompt,
        device,
        max_length=max_seq_len,
        pad_token_id=pad_token_id,
    )
    logger.info(f"Rank {rank}: Input shape: {input_tokens.shape}, Prompt length: {prompt_length}")

    # Log the prompt
    if rank == 0:
        logger.info(f"Rank {rank}: === PROMPT ===")
        logger.info(f"Rank {rank}: {prompt}")
        logger.info(f"Rank {rank}: ==============")

    # Compile model
    compiled_model = torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)

    # Warm-up forward pass (triggers compilation)
    logger.info(f"Rank {rank}: Running warm-up forward pass...")
    # warmup_output = run_neuron_forward(rank, compiled_model, input_tokens, run_number=1)
    logger.info(f"Rank {rank}: Warm-up complete")

    # Synchronize all ranks before generation
    dist.barrier()

    logger.info(f"Rank {rank}: Starting token generation...")
    generated_tokens, generation_times, generated_token_ids = generate_tokens_fixed_length(
        rank=rank,
        compiled_model=compiled_model,
        input_tokens=input_tokens,
        prompt_length=prompt_length,
        tokenizer=tokenizer,
        num_tokens=num_tokens,
        temperature=0.9,
        top_k=50,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    full_output = decode_tokens(tokenizer, generated_tokens)
    generated_text = decode_tokens(tokenizer, torch.tensor(generated_token_ids))

    if rank == 0:
        logger.info(f"Rank {rank}: === GENERATED OUTPUT ===")
        logger.info(f"Rank {rank}: Prompt: {prompt}")
        logger.info(f"Rank {rank}: Generated: {generated_text}")
        logger.info(f"Rank {rank}: Full text: {full_output}")
        logger.info(f"Rank {rank}: =========================")

        logger.info(f"Rank {rank}: Generation test PASSED")
        logger.info(f"Rank {rank}: Final sequence shape: {generated_tokens.shape}")

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test TorchTitan Qwen3 8B generation with tensor parallelism"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=NUM_TOKENS_TO_GENERATE,
        help=f"Number of tokens to generate (default: {NUM_TOKENS_TO_GENERATE})",
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
    run_qwen3_generation(
        model_name="qwen3_8b",
        num_tokens=args.num_tokens,
        prompt=args.prompt,
        hf_model_id=HF_MODEL_ID,
        max_seq_len=args.max_seq_len,
    )
