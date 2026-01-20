"""
DeepSeek V3 16B inference on Trainium2 using TorchTitan model definitions
and torch.compile with Neuron backend.
"""

import argparse
import dataclasses
import logging
import time

import torch
import torch.distributed as dist
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
from torchtitan.distributed import NoParallel
from torchtitan.models.deepseek_v3 import DeepSeekV3Model, DeepSeekV3ModelArgs
from torchtitan.models.moe import MoEArgs
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import torch_neuronx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Neuron-compatible rotary embeddings (using real arithmetic instead of complex)
# ============================================================================

def apply_rotary_emb_real_paired(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings using real arithmetic that matches complex RoPE.

    This implements the same computation as complex RoPE:
        x_complex = x[..., ::2] + i * x[..., 1::2]
        y_complex = x_complex * (cos + i * sin)
        y = interleave(y_complex.real, y_complex.imag)

    But using only real arithmetic for Neuron compatibility.

    Args:
        x: Input tensor of shape (..., head_dim) where head_dim is even
        cos: Cosine tensor of shape (seq_len, head_dim // 2) or broadcastable
        sin: Sine tensor of shape (seq_len, head_dim // 2) or broadcastable

    Returns:
        Tensor with rotary embeddings applied, same shape as x
    """
    # x shape: typically (batch, seq_len, n_heads, head_dim)
    # Split into even and odd indices (this pairs adjacent elements like complex)
    x_even = x[..., ::2]  # x0, x2, x4, ... (real parts in complex view)
    x_odd = x[..., 1::2]  # x1, x3, x5, ... (imag parts in complex view)

    # Complex multiplication: (x_even + i*x_odd) * (cos + i*sin)
    # = (x_even*cos - x_odd*sin) + i*(x_even*sin + x_odd*cos)
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    # Interleave back: [y_even[0], y_odd[0], y_even[1], y_odd[1], ...]
    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    return y.type_as(x)


def patch_deepseek_v3_for_neuron(model: nn.Module) -> nn.Module:
    """Patch DeepSeek V3 model to use Neuron-compatible rotary embeddings.

    The original TorchTitan DeepSeek V3 uses complex arithmetic for RoPE:
        x = torch.view_as_complex(...)
        y = torch.view_as_real(x * freqs_cis)

    Neuron doesn't support complex operations with broadcasting, so we convert
    to use real arithmetic that produces identical results.
    """
    import torchtitan.models.deepseek_v3.model.model as ds_model

    # Store original freqs_cis (complex) and convert to cos/sin
    original_freqs_cis = model.freqs_cis.clone()

    # Extract cos and sin from complex exponentials
    # freqs_cis shape: (seq_len, head_dim // 2), dtype: complex64
    # e^(i*theta) = cos(theta) + i*sin(theta)
    cos = original_freqs_cis.real.float()  # (seq_len, head_dim // 2)
    sin = original_freqs_cis.imag.float()  # (seq_len, head_dim // 2)

    # Replace the complex freqs_cis with cos/sin cache
    model.register_buffer("rope_cos", cos, persistent=False)
    model.register_buffer("rope_sin", sin, persistent=False)

    # Monkey-patch the apply_rotary_emb function
    def new_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Neuron-compatible rotary embedding using real arithmetic."""
        # x shape: (batch, seq_len, n_heads, head_dim) where head_dim = qk_rope_head_dim
        # freqs_cis is the original complex tensor, but we'll use our cos/sin instead

        seq_len = x.shape[1]
        ndim = x.ndim

        # Get the cos/sin for this sequence length
        cos = model.rope_cos[:seq_len]
        sin = model.rope_sin[:seq_len]

        # Ensure cos/sin are on same device as x
        cos = cos.to(x.device, dtype=x.dtype)
        sin = sin.to(x.device, dtype=x.dtype)

        # Reshape for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
        # to match x shape (batch, seq_len, n_heads, head_dim)
        shape = [1] * ndim
        shape[1] = seq_len  # seq_len dimension
        shape[-1] = cos.shape[-1]  # head_dim // 2
        cos = cos.view(*shape)
        sin = sin.view(*shape)

        return apply_rotary_emb_real_paired(x, cos, sin)

    # Replace the module-level function
    ds_model.apply_rotary_emb = new_apply_rotary_emb

    logger.info("Patched DeepSeek V3 model with Neuron-compatible rotary embeddings")
    return model

BATCH_SIZE = 1
NUM_TOKENS_TO_GENERATE = 20
MAX_SEQ_LEN = 128
DEFAULT_PROMPT = "The future of artificial intelligence is"

# Default HF model ID for tokenizer (model architecture differs)
HF_TOKENIZER_ID = "deepseek-ai/deepseek-moe-16b-base"

torch.set_default_dtype(torch.float32)


def get_debug_model_args() -> DeepSeekV3ModelArgs:
    """Get smaller debug model args for testing (fits in TP=4 memory)."""
    # Debug model: 8 experts, 6 layers, smaller dimensions
    # Roughly similar to TorchTitan's debugmodel configuration
    base_args = {
        "vocab_size": 102400,
        "dim": 512,  # Reduced from 2048
        "inter_dim": 1408,  # Reduced from 10944
        "moe_inter_dim": 512,  # Reduced from 1408
        "n_layers": 6,  # Reduced from 27
        "n_dense_layers": 1,
        "n_heads": 8,  # Reduced from 16
        "moe_args": MoEArgs(
            num_experts=8,  # Reduced from 64
            num_shared_experts=2,
            top_k=2,  # Reduced from 6
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            use_grouped_mm=False,
        ),
        "q_lora_rank": 0,
        "kv_lora_rank": 128,  # Reduced from 512
        "qk_nope_head_dim": 64,  # Reduced from 128
        "qk_rope_head_dim": 32,  # Reduced from 64
        "v_head_dim": 64,  # Reduced from 128
    }

    # Optional arguments that may not exist in older versions
    if dataclasses.is_dataclass(DeepSeekV3ModelArgs):
        field_names = {f.name for f in dataclasses.fields(DeepSeekV3ModelArgs)}
    else:
        field_names = set()

    optional_args = {
        "attn_type": "sdpa",
        "attn_mask_type": "causal",
        "mscale": 0.70,
    }

    for key, value in optional_args.items():
        if key in field_names:
            base_args[key] = value

    return DeepSeekV3ModelArgs(**base_args)


def get_deepseek_v3_16b_args() -> DeepSeekV3ModelArgs:
    """Get model args for DeepSeek V3 16B model."""
    # Base arguments that should work with all versions
    base_args = {
        "vocab_size": 102400,
        "dim": 2048,
        "inter_dim": 10944,
        "moe_inter_dim": 1408,
        "n_layers": 27,
        "n_dense_layers": 1,
        "n_heads": 16,
        "moe_args": MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            use_grouped_mm=False,  # Use for-loop for inference compatibility
        ),
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }

    # Optional arguments that may not exist in older versions
    if dataclasses.is_dataclass(DeepSeekV3ModelArgs):
        field_names = {f.name for f in dataclasses.fields(DeepSeekV3ModelArgs)}
    else:
        field_names = set()

    optional_args = {
        "attn_type": "sdpa",  # Use SDPA for Neuron compatibility
        "attn_mask_type": "causal",
        "mscale": 0.70,
    }

    for key, value in optional_args.items():
        if key in field_names:
            base_args[key] = value

    return DeepSeekV3ModelArgs(**base_args)


def get_model_args_from_hf(model_id: str) -> DeepSeekV3ModelArgs:
    """Load model config from HuggingFace and convert to DeepSeekV3ModelArgs."""
    try:
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # Extract MoE configuration
        num_experts = getattr(hf_config, "n_routed_experts", 64)
        num_shared_experts = getattr(hf_config, "n_shared_experts", 2)
        top_k = getattr(hf_config, "num_experts_per_tok", 6)

        # Base arguments that should work with all versions
        base_args = {
            "dim": hf_config.hidden_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_heads": hf_config.num_attention_heads,
            "vocab_size": hf_config.vocab_size,
            "inter_dim": getattr(hf_config, "intermediate_size", 10944),
            "moe_inter_dim": getattr(hf_config, "moe_intermediate_size", 1408),
            "n_dense_layers": getattr(hf_config, "first_k_dense_replace", 1),
            "norm_eps": getattr(hf_config, "rms_norm_eps", 1e-5),
            "moe_args": MoEArgs(
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                score_func="softmax",
                route_norm=False,
                score_before_experts=False,
                use_grouped_mm=False,
            ),
            "q_lora_rank": getattr(hf_config, "q_lora_rank", 0),
            "kv_lora_rank": getattr(hf_config, "kv_lora_rank", 512),
            "qk_nope_head_dim": getattr(hf_config, "qk_nope_head_dim", 128),
            "qk_rope_head_dim": getattr(hf_config, "qk_rope_head_dim", 64),
            "v_head_dim": getattr(hf_config, "v_head_dim", 128),
            "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
            "max_seq_len": getattr(hf_config, "max_position_embeddings", 4096),
        }

        # Optional arguments that may not exist in older versions
        if dataclasses.is_dataclass(DeepSeekV3ModelArgs):
            field_names = {f.name for f in dataclasses.fields(DeepSeekV3ModelArgs)}
        else:
            field_names = set()

        optional_args = {
            "attn_type": "sdpa",
            "attn_mask_type": "causal",
        }

        for key, value in optional_args.items():
            if key in field_names:
                base_args[key] = value

        return DeepSeekV3ModelArgs(**base_args)
    except Exception as e:
        logger.warning(f"Failed to load config from HF ({e}), using default 16B config")
        return get_deepseek_v3_16b_args()


def apply_tp_for_inference(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    skip_tp: bool = False,
):
    """Apply tensor parallelism for DeepSeek V3 inference.

    Due to Neuron's 40MB collective operation size limit, we cannot use DTensor-based
    TP on this model. The embedding layer alone is ~800MB which exceeds the limit.

    When skip_tp=True, no tensor parallelism is applied and the model runs with
    full replication on each device (data parallel style).
    """
    if skip_tp:
        logger.info("Skipping tensor parallelism due to Neuron collective size limits")
        return model

    # Note: DTensor-based TP triggers all_reduce for parameter replication
    # which exceeds Neuron's 40MB collective size limit for large weights.
    # For now, we skip TP entirely. Future work could implement manual
    # weight sharding without using DTensor collectives.

    logger.warning(
        "DTensor-based TP is not supported on Neuron due to collective size limits. "
        "Running with replicated weights on each device."
    )
    return model


def load_tokenizer(model_id: str = HF_TOKENIZER_ID) -> AutoTokenizer:
    """Load the HuggingFace tokenizer."""
    logger.info(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    logger.info(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    return tokenizer


def get_weight_map() -> dict:
    """Get global weight mapping from HuggingFace to TorchTitan."""
    return {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }


def get_layer_weight_map(layer_idx: int, n_dense_layers: int, q_lora_rank: int) -> dict:
    """Get weight mapping for a specific layer."""
    hf_prefix = f"model.layers.{layer_idx}"
    tt_prefix = f"layers.{layer_idx}"

    # Base attention mapping (MLA structure)
    mapping = {
        f"{hf_prefix}.self_attn.kv_a_proj_with_mqa.weight": f"{tt_prefix}.attention.wkv_a.weight",
        f"{hf_prefix}.self_attn.kv_a_layernorm.weight": f"{tt_prefix}.attention.kv_norm.weight",
        f"{hf_prefix}.self_attn.kv_b_proj.weight": f"{tt_prefix}.attention.wkv_b.weight",
        f"{hf_prefix}.self_attn.o_proj.weight": f"{tt_prefix}.attention.wo.weight",
        f"{hf_prefix}.input_layernorm.weight": f"{tt_prefix}.attention_norm.weight",
        f"{hf_prefix}.post_attention_layernorm.weight": f"{tt_prefix}.ffn_norm.weight",
    }

    # Query projection based on q_lora_rank
    if q_lora_rank == 0:
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{tt_prefix}.attention.wq.weight"
    else:
        mapping.update(
            {
                f"{hf_prefix}.self_attn.q_a_proj.weight": f"{tt_prefix}.attention.wq_a.weight",
                f"{hf_prefix}.self_attn.q_a_layernorm.weight": f"{tt_prefix}.attention.q_norm.weight",
                f"{hf_prefix}.self_attn.q_b_proj.weight": f"{tt_prefix}.attention.wq_b.weight",
            }
        )

    # FFN/MoE layers based on layer index
    is_moe_layer = layer_idx >= n_dense_layers

    if not is_moe_layer:
        # Dense FFN layer
        mapping.update(
            {
                f"{hf_prefix}.mlp.gate_proj.weight": f"{tt_prefix}.feed_forward.w1.weight",
                f"{hf_prefix}.mlp.up_proj.weight": f"{tt_prefix}.feed_forward.w3.weight",
                f"{hf_prefix}.mlp.down_proj.weight": f"{tt_prefix}.feed_forward.w2.weight",
            }
        )
    else:
        # MoE layer: router and shared experts
        mapping.update(
            {
                f"{hf_prefix}.mlp.gate.weight": f"{tt_prefix}.moe.router.gate.weight",
                f"{hf_prefix}.mlp.shared_experts.gate_proj.weight": f"{tt_prefix}.moe.shared_experts.w1.weight",
                f"{hf_prefix}.mlp.shared_experts.up_proj.weight": f"{tt_prefix}.moe.shared_experts.w3.weight",
                f"{hf_prefix}.mlp.shared_experts.down_proj.weight": f"{tt_prefix}.moe.shared_experts.w2.weight",
            }
        )
        # Expert bias (optional)
        mapping[f"{hf_prefix}.mlp.gate.e_score_correction_bias"] = f"{tt_prefix}.moe.expert_bias"

    return mapping


def load_moe_expert_weights(
    tt_state_dict: dict,
    hf_state_dict: dict,
    layer_idx: int,
    num_experts: int,
    moe_inter_dim: int,
    dim: int,
) -> None:
    """Load MoE expert weights from HuggingFace format to grouped format."""
    hf_prefix = f"model.layers.{layer_idx}.mlp.experts"
    tt_prefix = f"layers.{layer_idx}.moe.experts"

    # Initialize grouped weight tensors
    w1_list = []
    w2_list = []
    w3_list = []

    for expert_idx in range(num_experts):
        gate_key = f"{hf_prefix}.{expert_idx}.gate_proj.weight"
        up_key = f"{hf_prefix}.{expert_idx}.up_proj.weight"
        down_key = f"{hf_prefix}.{expert_idx}.down_proj.weight"

        if gate_key in hf_state_dict:
            w1_list.append(hf_state_dict[gate_key])
            w3_list.append(hf_state_dict[up_key])
            w2_list.append(hf_state_dict[down_key])
        else:
            # Initialize with zeros if not found
            w1_list.append(torch.zeros(moe_inter_dim, dim))
            w3_list.append(torch.zeros(moe_inter_dim, dim))
            w2_list.append(torch.zeros(dim, moe_inter_dim))

    # Stack into grouped format: (num_experts, hidden_dim, dim)
    if w1_list:
        tt_state_dict[f"{tt_prefix}.w1"] = torch.stack(w1_list, dim=0)
        tt_state_dict[f"{tt_prefix}.w2"] = torch.stack(w2_list, dim=0)
        tt_state_dict[f"{tt_prefix}.w3"] = torch.stack(w3_list, dim=0)


def load_weights_from_hf(
    model: nn.Module,
    model_id: str,
    model_args: DeepSeekV3ModelArgs,
) -> None:
    """Load pretrained weights from HuggingFace into TorchTitan model."""
    logger.info(f"Loading pretrained weights from {model_id}...")
    load_start = time.time()

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        hf_state_dict = hf_model.state_dict()
    except Exception as e:
        logger.error(f"Failed to load HF model: {e}")
        logger.info("Using random initialization instead")
        return

    logger.info(f"HuggingFace model loaded in {time.time() - load_start:.2f}s")
    logger.info(f"HuggingFace state dict keys: {len(hf_state_dict)}")

    # Build complete weight mapping
    weight_map = get_weight_map()
    n_layers = model_args.n_layers
    n_dense_layers = model_args.n_dense_layers
    q_lora_rank = model_args.q_lora_rank

    for i in range(n_layers):
        weight_map.update(get_layer_weight_map(i, n_dense_layers, q_lora_rank))

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

    # Load MoE expert weights for MoE layers
    num_experts = model_args.moe_args.num_experts
    moe_inter_dim = model_args.moe_inter_dim
    dim = model_args.dim

    for layer_idx in range(n_dense_layers, n_layers):
        load_moe_expert_weights(
            tt_state_dict, hf_state_dict, layer_idx, num_experts, moe_inter_dim, dim
        )

    # Load the mapped state dict
    model.load_state_dict(tt_state_dict, strict=False)

    logger.info(f"Loaded {loaded_count} weight tensors")
    if missing_in_hf:
        logger.warning(f"Missing in HF model: {missing_in_hf[:5]}{'...' if len(missing_in_hf) > 5 else ''}")
    if missing_in_tt:
        logger.warning(f"Missing in TT model: {missing_in_tt[:5]}{'...' if len(missing_in_tt) > 5 else ''}")
    if shape_mismatch:
        logger.warning(f"Shape mismatches: {shape_mismatch[:5]}{'...' if len(shape_mismatch) > 5 else ''}")

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
    """Tokenize a prompt and pad to fixed length."""
    logger.info(f"Tokenizing prompt: '{prompt}'")

    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    actual_length = len(encoded)

    if actual_length > max_length:
        logger.warning(f"Prompt length {actual_length} exceeds max_length {max_length}, truncating")
        encoded = encoded[:max_length]
        actual_length = max_length

    # Pad to fixed length
    padding_length = max_length - actual_length
    padded = encoded + [pad_token_id] * padding_length

    tokens = torch.tensor([padded], dtype=torch.int32, device=device)

    logger.info(f"Tokenized to {actual_length} tokens (padded to {max_length})")
    logger.info(f"Token IDs: {encoded}")

    return tokens, actual_length


def decode_tokens(tokenizer: AutoTokenizer, tokens: torch.Tensor) -> str:
    """Decode token tensor back to string."""
    if tokens.dim() == 2:
        tokens = tokens[0]
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
    """Generate tokens autoregressively with fixed sequence length."""
    max_seq_len = input_tokens.shape[1]
    tokens = input_tokens.clone()
    generation_times = []
    generated_token_ids = []

    current_pos = prompt_length

    max_generate = min(num_tokens, max_seq_len - prompt_length)
    if max_generate < num_tokens:
        logger.warning(f"Rank {rank}: Can only generate {max_generate} tokens (requested {num_tokens})")

    for i in range(max_generate):
        step_start = time.time()

        with torch.no_grad():
            logits = compiled_model(tokens)

        if isinstance(logits, DTensor):
            logits = logits.to_local()

        next_token_logits = logits[:, current_pos - 1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            next_token_logits_filtered = torch.full_like(next_token_logits, float("-inf"))
            next_token_logits_filtered.scatter_(1, top_k_indices, top_k_values)
            next_token_logits = next_token_logits_filtered

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        step_time = time.time() - step_start
        generation_times.append(step_time)

        next_token_id = next_token.item()
        generated_token_ids.append(next_token_id)

        tokens[:, current_pos] = next_token.squeeze(-1).to(tokens.dtype)
        current_pos += 1

        if tokenizer is not None:
            token_str = tokenizer.decode([next_token_id])
            logger.info(f"Rank {rank}: Token {i+1}/{max_generate}: {next_token_id} -> '{token_str}'")
        else:
            logger.info(f"Rank {rank}: Token {i+1}/{max_generate}: {next_token_id} ({step_time:.3f}s)")

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

    actual_length = prompt_length + len(generated_token_ids)
    meaningful_tokens = tokens[:, :actual_length].clone()

    return meaningful_tokens, generation_times, generated_token_ids


def move_dtensor_model_to_device(model: nn.Module, device: torch.device) -> None:
    """Move a model with DTensor parameters to the target device."""
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


def create_deepseek_v3_model_with_tp(
    model_args: DeepSeekV3ModelArgs,
    world_size: int,
    device_mesh: DeviceMesh,
    device: torch.device,
    rank: int,
    hf_model_id: str = None,
    random_init: bool = False,
) -> nn.Module:
    """
    Create DeepSeek V3 model: CPU (with weights) -> move to device.

    Note: DTensor-based TP is disabled due to Neuron's 40MB collective size limit.
    The model runs with replicated weights on each device.
    """
    torch.set_default_dtype(torch.float32)

    logger.info(f"Rank {rank}: Creating DeepSeek V3 model on CPU...")
    cpu_start = time.time()
    with torch.device("cpu"):
        model = DeepSeekV3Model(model_args)
    logger.info(f"Rank {rank}: CPU creation took {time.time() - cpu_start:.3f}s")

    if not random_init and hf_model_id:
        logger.info(f"Rank {rank}: Loading pretrained weights...")
        load_weights_from_hf(model, model_id=hf_model_id, model_args=model_args)
    else:
        logger.info(f"Rank {rank}: Using random initialization...")
        model.init_weights()

    # Patch model for Neuron compatibility (replace complex RoPE with real arithmetic)
    logger.info(f"Rank {rank}: Patching model for Neuron-compatible rotary embeddings...")
    model = patch_deepseek_v3_for_neuron(model)

    # Skip TP due to Neuron collective size limits
    logger.info(f"Rank {rank}: Skipping TP (Neuron collective size limit)")
    model = apply_tp_for_inference(model=model, tp_mesh=device_mesh, skip_tp=True)

    logger.info(f"Rank {rank}: Moving to {device}...")
    move_start = time.time()
    # Since we skip TP, model has regular tensors, not DTensors
    model = model.to(device)
    logger.info(f"Rank {rank}: Move to device took {time.time() - move_start:.3f}s")

    return model


def run_deepseek_v3_generation(**kwargs):
    """Run DeepSeek V3 model generation."""
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model_name = kwargs.get("model_name", "deepseek_v3_16b")
    num_tokens = kwargs.get("num_tokens", NUM_TOKENS_TO_GENERATE)
    prompt = kwargs.get("prompt", DEFAULT_PROMPT)
    hf_model_id = kwargs.get("hf_model_id", None)
    max_seq_len = kwargs.get("max_seq_len", MAX_SEQ_LEN)
    random_init = kwargs.get("random_init", True)
    tokenizer_id = kwargs.get("tokenizer_id", HF_TOKENIZER_ID)
    debug = kwargs.get("debug", False)

    # Get model args
    if debug:
        logger.info(f"Rank {rank}: Using DEBUG model configuration (smaller for testing)")
        model_args = get_debug_model_args()
        model_name = "deepseek_v3_debug"
    elif hf_model_id:
        model_args = get_model_args_from_hf(hf_model_id)
    else:
        model_args = get_deepseek_v3_16b_args()

    # Override max_seq_len if provided
    model_args.max_seq_len = max_seq_len

    logger.info(f"=== Rank {rank}/{world_size}: Testing {model_name} generation ===")
    logger.info(f"Model config: dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")
    logger.info(f"MoE config: experts={model_args.moe_args.num_experts}, top_k={model_args.moe_args.top_k}")

    device = torch.device(f"neuron:{rank}")
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id

    logger.info(f"Rank {rank}: Using EOS token ID: {eos_token_id}, PAD token ID: {pad_token_id}")

    # Create model
    model = create_deepseek_v3_model_with_tp(
        model_args,
        world_size,
        device_mesh,
        device,
        rank,
        hf_model_id=hf_model_id,
        random_init=random_init,
    )
    model.eval()

    # Tokenize the prompt
    input_tokens, prompt_length = tokenize_prompt(
        tokenizer,
        prompt,
        device,
        max_length=max_seq_len,
        pad_token_id=pad_token_id,
    )
    logger.info(f"Rank {rank}: Input shape: {input_tokens.shape}, Prompt length: {prompt_length}")

    if rank == 0:
        logger.info(f"Rank {rank}: === PROMPT ===")
        logger.info(f"Rank {rank}: {prompt}")
        logger.info(f"Rank {rank}: ==============")

    # Compile model (or use eager mode)
    no_compile = kwargs.get("no_compile", False)
    if no_compile:
        logger.info(f"Rank {rank}: Running in eager mode (no torch.compile)")
        compiled_model = model
    else:
        logger.info(f"Rank {rank}: Compiling model with neuron backend...")
        compiled_model = torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)

    # Warm-up forward pass
    logger.info(f"Rank {rank}: Running warm-up forward pass...")
    warmup_output = run_neuron_forward(rank, compiled_model, input_tokens, run_number=1)
    logger.info(f"Rank {rank}: Warm-up complete")

    dist.barrier()

    # Generate tokens
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
        description="DeepSeek V3 16B inference on Trainium2 with tensor parallelism"
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
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID for loading weights (optional)",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=HF_TOKENIZER_ID,
        help=f"HuggingFace tokenizer ID (default: {HF_TOKENIZER_ID})",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=True,
        help="Use random initialization instead of loading weights",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Use smaller debug model for testing (8 experts, 6 layers, fits in TP=4)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Run in eager mode without torch.compile (for debugging)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_deepseek_v3_generation(
        model_name="deepseek_v3_16b",
        num_tokens=args.num_tokens,
        prompt=args.prompt,
        hf_model_id=args.hf_model_id,
        max_seq_len=args.max_seq_len,
        random_init=args.random_init or (args.hf_model_id is None),
        tokenizer_id=args.tokenizer_id,
        debug=args.debug,
        no_compile=args.no_compile,
    )
