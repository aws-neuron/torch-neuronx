# GPT-OSS Model for Neuron torch.compile

This directory contains a GPT-OSS model implementation adapted for the Neuron `torch.compile` backend with tensor parallelism support. We demonstrate an example for forward pass with greedy decoding. We are not using real weights due to lack of MXFP support.

## Quick Start

```bash
export PYTHONPATH=/path/to/examples/torch_compile:$PYTHONPATH

torchrun --nproc_per_node=64 /path/to/gpt_oss_example.py
```
This runs a randomly generated prompt.

## Model Patches

The model files in this directory are adapted from TorchTitan's GPT-OSS implementation. Several patches were required to make the model compatible with the Neuron compilation pipeline:

### 1. RoPE (Rotary Position Embeddings) - `rope_patched.py`

**Problem:** The original RoPE implementation uses complex number operations (`torch.polar`, complex multiplication), which are not supported by Neuron.

**Solution:** Replaced complex number operations with equivalent real number operations:
- `precompute_rope_cache_patched()` returns concatenated cos/sin tensors instead of complex exponentials
- `apply_rotary_emb_patched()` applies rotation using real arithmetic: `x' = x * cos + rotate_half(x) * sin`

### 2. FlexAttention - `attention.py`

**Problem:** FlexAttention's device validation only allows CUDA/CPU/HPU, blocking Neuron devices.

**Solution:**
- Patched `flex_attn_module._validate_device` to skip validation for Neuron devices
- This allows Dynamo to trace FlexAttention calls, which are then translated by the Neuron FX pass into primitive operations (matmuls, softmax, masking)

### 3. KV Repeat for GQA - `rope_patched.py`

**Problem:** The original `repeat_kv` uses `expand()` which creates zero-stride tensors that cause issues in the Neuron compilation pipeline.

**Solution:** `repeat_kv_patched()` uses `repeat()` instead of `expand()` to create proper contiguous tensors.

## Adapting Your Own Model

If you're adapting a model for Neuron torch.compile, watch out for:

1. **FlexAttention** - Patch device validation
2. **Zero-stride tensors** - Use `repeat()` instead of `expand()` where possible
3. **In-place operations** - Some `copy_()` patterns may need functional alternatives

## Files

| File | Description |
|------|-------------|
| `gpt_oss_example.py` | Main example script with TP and token generation |
| `model.py` | GPT-OSS model architecture |
| `attention.py` | FlexAttention with Neuron device patch |
| `rope_patched.py` | RoPE without complex numbers |
| `moe.py` | Static Mixture of Experts implementation |
| `args.py` | Model configuration dataclasses |
| `gpt_oss_utils.py` | Utility functions |
| `gpt_oss_tp.py` | Tensor parallelism utilities |
