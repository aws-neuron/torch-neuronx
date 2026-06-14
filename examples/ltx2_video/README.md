# LTX-2 18.88B on Trainium2 — Native PyTorch path

Direct `torch.device("neuron")` + `torch.compile(backend="neuron")` serving
of Lightricks' LTX-2 (18.88B audio+video DiT) on AWS Trainium2 / Beta 3.

**Status** — working pipeline shape, with the production correctness recipe
from `aws-neuron/neuronx-distributed-inference/contrib/models/
ltx2-video-audio` ported in. End-to-end generation on `trn2.48xlarge` TP=4
produces a 25-frame 384×512 MP4 at 8 denoising steps. See
[BENCHMARK_VS_H100.md](BENCHMARK_VS_H100.md) for warm latency, $/clip, and
the comparison vs H100.

## Architecture

```
trn2.48xlarge (16 NeuronDevices × 4 cores = 64 cores)

Text encoder (Gemma-3 12B, 24 GB)            → CPU
LTX-2 connectors (3 GB intermediate)         → CPU
LTX-2 video VAE  (1.25 B)                    → CPU
LTX-2 audio VAE + vocoder                    → CPU
LTX-2 DiT transformer (18.88 B / 38 GB bf16) → Neuron, TP=4
    ├─ rank 0  ColShard q/k/v + RowShard out  ~10 GB / core
    ├─ rank 1                                  ~10 GB / core
    ├─ rank 2                                  ~10 GB / core
    └─ rank 3                                  ~10 GB / core
```

CPU components communicate with the TP-sharded transformer via a single
boundary wrapper that handles tensor placement and additive-mask
conversion.

## The ten Neuron correctness fixes

These are needed for the compiled bf16 forward to match CPU output. Without
them, the run produces a washed-out video (std collapse, ±11 outliers in
noise predictions, PSNR ~8 dB vs CPU). Numbered to match the in-code
comments:

1. **Beta 3 device API** — `torch.device("neuron")`, NOT `privateuseone:N`
   (the Beta 2 idiom).
2. **Meta-init build** — `with torch.device("meta"): from_config(...)`
   keeps a 38 GB transformer off any single core during construction.
3. **TP=4 plan covering all six attention paths** — `attn1, audio_attn1,
   attn2, audio_attn2, audio_to_video_attn, video_to_audio_attn`, plus
   both FFNs.
4. **Adaptive QK norm with cross-rank all-reduce** — `qk_norm:
   rms_norm_across_heads` operates on the full inner_dim; under
   ColumnParallel sharding, each rank holds inner_dim/N and the norm
   needs an all-reduce on the local sum-of-squares to compute the
   global RMS, then divides by full_dim.
5. **RoPE rank slicing via `RankTensor`** — a Python-int rank gets baked
   as constant 0 in XLA SPMD tracing, causing all ranks to apply the
   same RoPE shard. `RankTensor` exposes `arange[rank]` as a 0-d int32
   tensor whose value is per-rank-correct under tracing.
6. **`attn.heads` patched to heads/N** — the block forward does
   `query.unflatten(2, (attn.heads, -1))` and needs the local head
   count to compute the right `head_dim`.
7. **CPU↔Neuron transfer wrapper** — single chokepoint at the
   transformer boundary; moves all tensor inputs to Neuron so we don't
   chase device mismatches through the diffusers pipeline internals.
8. **RoPE cos/sin cast to bf16** — the RoPE module returns float32 for
   numerical precision. Crossing the compile boundary as fp32 produces
   intermediates the lazy backend mishandles when paired with bf16
   hidden states. Cast right before return.
9. **BMM-SDPA replacement** — the most critical fix.
   `torch.nn.functional.scaled_dot_product_attention` miscomputes on
   Neuron's compiled bf16 lazy backend for LTX-2: empirically observed
   damped activations (std 0.84 vs CPU 1.06) plus rare ±11 outliers vs
   CPU's clean ±5 envelope. The AWS Neuron contrib (which we follow) ships
   `replace_sdpa_with_bmm()` — explicit `torch.bmm` + softmax + masked
   add. CPU calls fall through to the original SDPA.
10. **Encoder masks → additive `-10000.0` bias** — diffusers passes a
    {0, 1} bool mask. Convert to additive form `(1 - mask) * -10000` BEFORE
    crossing the compile boundary. Crucial: must be `-10000.0`, NOT
    `0.0` for "attend everywhere"; the AWS team's docstring: "Using
    all-zeros causes XLA to constant-fold the mask, dropping it from
    the compiled graph."

Fixes 9 and 10 are NEW in this version; 1-8 mirror the WIP at
`github.com/arminagha1234/Armin-Neuron/ltx2-trainium-wip`.

## Files

| File | Role |
|---|---|
| `src/run_ltx2_native.py` | Main TP=4 runner: meta-init → parallelize → load → install adaptive QK norm → patch RoPE → swap into pipeline → CPU/Neuron patches → generate |
| `src/ltx2_tp_plan.py` | TP plan (1344 entries) + `apply_tp_fixes` (heads patching) + `install_adaptive_qk_norm` (TP RMSNorm) + `patch_rope_rank_slice` (4-RoPE rank slice via RankTensor + bf16 cast) |
| `src/ltx2_meta_loader.py` | Sharded weight loader: regex SHARD_RULES + module-walk resolver |
| `src/neuron_compat.py` | The three Neuron compatibility shims: `install_bmm_sdpa()`, `RankTensor`, `to_additive_mask()` |

## Reproduction

```bash
# Beta 3 DLC pulled, runtime driver installed via runtime_artifacts/*.deb.
# Inside the beta3 container (or its venv on host):

# Required env: torch_neuronx >= 2.11.3, diffusers @ git+main, transformers
#   >= 4.53, huggingface-hub, safetensors

NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
HF_HOME=/path/to/hf/cache \
torchrun --nproc_per_node=4 --rdzv_backend c10d \
         --rdzv_endpoint localhost:29500 \
    src/run_ltx2_native.py \
    --num-steps 8 --num-frames 25 --height 384 --width 512 \
    --output results/ltx2_native_run.png
```

Cold first call includes NEFF compilation (~3-5 minutes for the DiT). Warm
generations reuse the cached NEFFs.

## Known issues

- VAE + Gemma-3 text encoder run on CPU. They're the dominant chunk of warm
  latency (~113 s of 165 s warm wall-clock at canonical shape). Moving them
  onto Neuron is the largest remaining lever — see "Optimization roadmap"
  in [BENCHMARK_VS_H100.md](BENCHMARK_VS_H100.md).
- Output is bit-equivalent to CPU reference within bf16 rounding; the
  small remaining drift comes from BMM-vs-SDPA softmax associativity.
- `torch.compile(backend="neuron")` is enabled by default. Pass
  `--no-compile` for eager mode; useful for debugging.

## License

Apache-2.0. The `replace_sdpa_with_bmm` BMM fallback in `neuron_compat.py`
is adapted from `aws-neuron/neuronx-distributed-inference` (also
Apache-2.0).
