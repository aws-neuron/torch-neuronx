# Llama3 Model for Neuron torch.compile

This directory contains a Llama3 model implementation adapted for the Neuron `torch.compile` backend with tensor parallelism support.

## Quick Start

```bash

export PYTHONPATH=/path/to/examples/torch_compile:$PYTHONPATH

torchrun --nproc_per_node=8 /path/to/llama_example.py
```

## Model Patches

The model files in this directory are adapted from TorchTitan's Llama3 implementation. Several patches were required to make the model compatible with the Neuron compilation pipeline:

### 1. RoPE (Rotary Position Embeddings) - `rope_patched.py`

**Problem:** The original RoPE implementation uses complex number operations (`torch.polar`, complex multiplication), which are not supported by torch-mlir.

**Solution:** Replaced complex number operations with equivalent real number operations:
- `precompute_freqs_cis_patched()` returns separate cos/sin tensors instead of a complex tensor
- `apply_rotary_emb_patched()` applies rotation using real arithmetic on even/odd dimension pairs

The patched version is mathematically equivalent (verified with max difference < 5e-7).

### 2. Attention - `attention.py`

**Note:** This example uses standard SDPA (F.scaled_dot_product_attention), not FlexAttention.


## Files

| File | Description |
|------|-------------|
| `llama_example.py` | Main example script with TP and token generation |
| `model.py` | Llama3 model architecture |
| `attention.py` | SDPA-based attention for Neuron |
| `rope_patched.py` | RoPE without complex numbers |
| `args.py` | Model configuration dataclasses |
| `llama3_tp.py` | Tensor parallelism utilities |
