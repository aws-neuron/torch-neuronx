# FLUX.2-klein-4B on Trainium2 — torch.compile Example

End-to-end image generation with [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(4B DiT, distilled for 4-step generation) using `torch.compile(backend="neuron")`.

## Results

| Configuration | $/image | vs H100 ($0.0073) |
|---|---:|---:|
| trn2.48xl × 32 cores, 4 steps | **$0.0061** | **16% CHEAPER** |
| trn2.48xl × 16 cores, 28 steps | $0.0068 | 7% cheaper |
| trn2.3xl batch parallel (2 cores) | $0.024 | 3.3× more expensive |

Per-step NEFF execution: **960 ms** (measured via neuron-profile, 51.8% MFU).

## Quick start

```bash
# On a trn2.3xlarge or trn2.48xlarge with Beta 3 DLC
pip install git+https://github.com/huggingface/diffusers.git@main peft

# Single image (compiled)
HF_TOKEN=<token> python run_flux2_klein_native.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --image input.jpg --prompt "a cat in space" \
    --steps 4 --height 1024 --width 1024

# Batch parallel (2 cores on trn2.3xl)
NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python run_batch_parallel.py --core 0 --image input.jpg &
NEURON_RT_VISIBLE_CORES=2-3 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python run_batch_parallel.py --core 1 --image input.jpg &
wait
```

## Architecture

```
FLUX.2-klein-4B (DiT, distilled)
├── Text encoder (Qwen3)         → CPU
├── VAE encoder + decoder        → CPU
├── Scheduler (FlowMatchEuler)   → patched for Neuron
└── DiT transformer (4B, bf16)   → NEURON (torch.compile)
```

## The 10 Neuron patches

The `NeuronFlux2KleinPipeline` subclass applies 10 patches to make
the diffusers pipeline compatible with `torch.compile(backend="neuron")`:

1. Scheduler `set_timesteps` — CPU-side build + bf16 cast
2. `Timesteps` modules — sin/cos on CPU (avoids arange in compiled graph)
3. `get_1d_rotary_pos_embed` — real-arithmetic (no complex64)
4. `Flux2PosEmbed` swap — CPU+fp32+`use_real=True`
5. `encode_prompt` — Qwen3 on CPU, embeddings moved to Neuron
6. `_encode_vae_image` — VAE on CPU
7. `prepare_image_latents` — move to Neuron after parent
8. `_NeuronTransformerWrapper` — contiguous + device coercion
9. VAE.decode — coerce back to CPU
10. `__getattr__` proxy — forwards diffusers utilities

## Files

| File | Description |
|---|---|
| `neuron_flux2_klein_native.py` | 10-patch pipeline subclass |
| `run_flux2_klein_native.py` | Single-core CLI runner with benchmark |
| `run_batch_parallel.py` | Multi-core batch parallel runner |

## Requirements

- trn2.3xlarge or trn2.48xlarge
- Beta 3 DLC (torch 2.11, torch_neuronx 2.11.3)
- `diffusers` from git main
- `peft` (for LoRA support)
- HF token (model is gated)

## Validated

- trn2.3xlarge (ap-southeast-4): 65.9s / 28 steps, 32.9s / 4 steps
- trn2.48xlarge (us-east-2): 56.1s / 28 steps, multi-core scaling validated

## License

Apache-2.0. Model weights: [FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
