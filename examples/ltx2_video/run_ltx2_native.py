"""LTX-2 18.88B native PyTorch on Trainium2 — Beta 3 + TP=4.

This is the AWS-fix-augmented version of the WIP at
github.com/arminagha1234/Armin-Neuron/ltx2-trainium-wip + the production
fixes from aws-neuron/neuronx-distributed-inference contrib.

Pattern:
  1. install_bmm_sdpa()        — replace SDPA with BMM (CRITICAL)
  2. setup_distributed()        — torchrun, backend='neuron', TP=4
  3. meta-init build of LTX2VideoTransformer3DModel
  4. parallelize_module(...)    — TP plan
  5. apply_tp_fixes()           — heads/N + sharding-aware
  6. load_weights_sharded()     — per-rank weight streaming
  7. install_adaptive_qk_norm() — TP-aware RMSNorm w/ all-reduce
  8. patch_rope_rank_slice()    — RoPE cos/sin sliced by RankTensor +
                                   cast to bf16 for compile boundary
  9. swap into LTX2Pipeline, pin VAE/text_encoder to CPU
 10. encoder masks → to_additive_mask() (-10000.0 bias) at the boundary

Launch:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \\
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \\
        run_ltx2_native.py --num-steps 4 --num-frames 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Install BMM-SDPA replacement BEFORE any forward runs ────────────────────
# Stock F.scaled_dot_product_attention miscomputes on Neuron's compiled bf16
# lazy backend for LTX-2 (damped values, ±11 outliers). The AWS Neuron team's
# contrib documents this and provides the BMM replacement; we use the same.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_compat import install_bmm_sdpa, RankTensor, to_additive_mask
install_bmm_sdpa()

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

from ltx2_tp_plan import (
    ltx2_tp_plan, apply_tp_fixes, install_adaptive_qk_norm,
    patch_rope_rank_slice,
)
from ltx2_meta_loader import load_weights_sharded


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(backend="neuron", timeout=timedelta(minutes=30))
    return rank, world_size, local_rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=25)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=(
        "A golden retriever puppy runs across a sunny green meadow, "
        "its ears flapping in the wind. The camera follows from a low angle. "
        "Birds chirp in the background."
    ))
    p.add_argument("--output", default="results/ltx2_native_run.png")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode)")
    p.add_argument("--guidance-scale", type=float, default=4.0)
    return p.parse_args()


def build_transformer(rank, world_size, local_rank, device, dtype):
    """Construct LTX-2 transformer under meta-init, apply TP, load weights."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from diffusers import LTX2VideoTransformer3DModel
    from huggingface_hub import hf_hub_download, snapshot_download

    if rank == 0:
        print("[run] downloading LTX-2 transformer weights...", flush=True)

    config_path = hf_hub_download("Lightricks/LTX-2", "transformer/config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    snap = snapshot_download("Lightricks/LTX-2", allow_patterns=["transformer/*"])
    weights_dir = os.path.join(snap, "transformer")

    if rank == 0:
        print(f"[run] building meta-init transformer (18.88B params)", flush=True)
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(cfg)

    mesh = init_device_mesh("neuron", (world_size,))
    plan = ltx2_tp_plan(world_size)
    if rank == 0:
        print(f"[run] applying TP plan ({len(plan)} entries) world={world_size}",
              flush=True)
    parallelize_module(model, mesh, plan)
    apply_tp_fixes(model, world_size=world_size, rank=rank)

    if rank == 0:
        print("[run] loading weights sharded...", flush=True)
    t0 = time.time()
    load_weights_sharded(
        model, weights_dir,
        tp_local_rank=rank, world_size=world_size,
        dtype=dtype, device=device,
    )
    if rank == 0:
        print(f"[run] weights loaded in {time.time()-t0:.1f}s", flush=True)

    _materialize_remaining_meta(model, weights_dir, rank, world_size, device, dtype)
    install_adaptive_qk_norm(model, world_size=world_size, rank=rank)
    # Use RankTensor (traced per-rank index) instead of Python int rank
    rank_tensor = RankTensor(world_size, rank).to(device)
    patch_rope_rank_slice(model, world_size=world_size, rank=rank,
                          rank_tensor=rank_tensor)

    model.eval()
    return model, cfg


def _materialize_remaining_meta(model, weights_dir, rank, world_size, device, dtype):
    """Fill any param still on meta with its checkpoint tensor."""
    import json as _json
    from pathlib import Path as _Path
    from safetensors import safe_open as _safe_open

    idx_path = _Path(weights_dir) / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = _json.loads(idx_path.read_text())["weight_map"]

    meta_params = {n: p for n, p in model.named_parameters()
                   if getattr(p, "is_meta", False)}
    if rank == 0:
        print(f"[run] materialize: {len(meta_params)} params still on meta",
              flush=True)
    if not meta_params:
        return

    by_shard = {}
    for name in meta_params:
        sf = weight_map.get(name)
        if sf is None:
            continue
        by_shard.setdefault(sf, []).append(name)

    filled = 0
    for shard_file, names in by_shard.items():
        with _safe_open(_Path(weights_dir) / shard_file, framework="pt", device="cpu") as f:
            for name in names:
                full = f.get_tensor(name).to(dtype)
                parts = name.split(".")
                *parent_path, leaf = parts
                mod = model
                ok = True
                for p in parent_path:
                    if not hasattr(mod, p):
                        ok = False; break
                    mod = getattr(mod, p)
                if not ok:
                    continue
                setattr(mod, leaf,
                        torch.nn.Parameter(full.to(device), requires_grad=False))
                filled += 1
    if rank == 0:
        print(f"[run] materialize: filled {filled} meta params", flush=True)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"[run] torch={torch.__version__}", flush=True)
        print(f"[run] world_size={world_size}, args={vars(args)}", flush=True)

    device = torch.device("neuron")
    cpu = torch.device("cpu")
    dtype = torch.bfloat16

    # Probe the device
    x = torch.randn(4, 4, device=device)
    _ = x @ x.T
    if rank == 0:
        print("[run] neuron device probe ok", flush=True)
        print("[run] BMM-SDPA replacement active", flush=True)

    # Build TP-sharded transformer with all 4 AWS fixes baked in
    transformer, cfg = build_transformer(rank, world_size, local_rank, device, dtype)

    if not args.no_compile:
        if rank == 0:
            print("[run] wrapping transformer with torch.compile(backend='neuron')",
                  flush=True)
        transformer = torch.compile(
            transformer, backend="neuron",
            dynamic=False, fullgraph=False,
        )

    # Build pipeline (CPU) and swap in our sharded transformer
    if rank == 0:
        print("[run] loading LTX2Pipeline (CPU)", flush=True)
    t0 = time.time()
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=dtype)
    if rank == 0:
        print(f"[run] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    del pipe.transformer
    import gc; gc.collect()

    # CPU↔Neuron transfer wrapper at transformer boundary.
    # Also converts encoder_attention_mask / audio_encoder_attention_mask to
    # the additive -10000.0 bias form required by the AWS-team's recipe.
    def _move_to_neuron(obj):
        if torch.is_tensor(obj):
            return obj.to(device) if obj.device != device else obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(_move_to_neuron(o) for o in obj)
        if isinstance(obj, dict):
            return {k: _move_to_neuron(v) for k, v in obj.items()}
        return obj

    MASK_KEYS = ("encoder_attention_mask", "audio_encoder_attention_mask")

    def _process_kwargs(kwargs: dict) -> dict:
        for k in MASK_KEYS:
            v = kwargs.get(k)
            if torch.is_tensor(v) and not v.is_floating_point():
                kwargs[k] = to_additive_mask(v.to(device), dtype=dtype)
        return kwargs

    class _NeuronTransformerWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.config = inner.config if hasattr(inner, "config") else (
                inner._orig_mod.config if hasattr(inner, "_orig_mod") else None
            )

        def forward(self, *args, **kwargs):
            args = tuple(_move_to_neuron(a) for a in args)
            kwargs = {k: _move_to_neuron(v) for k, v in kwargs.items()}
            kwargs = _process_kwargs(kwargs)
            return self.inner(*args, **kwargs)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    pipe.transformer = _NeuronTransformerWrapper(transformer)
    LTX2Pipeline._execution_device = property(lambda self: device)

    # Pin text encoder + VAEs to CPU
    _orig_get = pipe._get_gemma_prompt_embeds
    def _patched_get_gemma(prompt, device=None, dtype=None, **kw):
        return _orig_get(prompt, device=cpu, dtype=dtype, **kw)
    pipe._get_gemma_prompt_embeds = _patched_get_gemma

    if hasattr(pipe, "connectors"):
        try:
            pipe.connectors = pipe.connectors.to(cpu)
        except Exception:
            pass

    if hasattr(pipe, "_encode_vae_image"):
        _orig_vae_enc = pipe._encode_vae_image
        def _patched_vae_enc(image, generator):
            if hasattr(image, "to"):
                image = image.to(cpu)
            return _orig_vae_enc(image, generator).to(device)
        pipe._encode_vae_image = _patched_vae_enc

    try:
        pipe.vae = pipe.vae.to(cpu)
    except Exception:
        pass
    if hasattr(pipe, "audio_vae"):
        try:
            pipe.audio_vae = pipe.audio_vae.to(cpu)
        except Exception:
            pass

    _orig_vae_dec = pipe.vae.decode
    def _patched_vae_dec(*args, **kwargs):
        new_args = [a.to(cpu) if torch.is_tensor(a) else a for a in args]
        new_kwargs = {k: (v.to(cpu) if torch.is_tensor(v) else v)
                      for k, v in kwargs.items()}
        return _orig_vae_dec(*new_args, **new_kwargs)
    pipe.vae.decode = _patched_vae_dec

    if hasattr(pipe, "audio_vae"):
        _orig_avd = pipe.audio_vae.decode
        def _patched_avd(*args, **kwargs):
            new_args = [a.to(cpu) if torch.is_tensor(a) else a for a in args]
            new_kwargs = {k: (v.to(cpu) if torch.is_tensor(v) else v)
                          for k, v in kwargs.items()}
            return _orig_avd(*new_args, **new_kwargs)
        pipe.audio_vae.decode = _patched_avd

    if rank == 0:
        print(f"\n[run] generating: {args.width}×{args.height}, "
              f"{args.num_frames} frames, {args.num_steps} steps", flush=True)
        print("[run] (cold first call includes NEFF compilation; expect 5-15 min)",
              flush=True)

    t0 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=1024,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_type="pil",
            )
    except Exception:
        elapsed = time.time() - t0
        if rank == 0:
            import traceback
            print(f"[run] FAILED after {elapsed:.1f}s", flush=True)
            traceback.print_exc()
        if dist.is_initialized():
            dist.barrier()
        return 1

    elapsed = time.time() - t0
    if rank == 0:
        print(f"[run] generated in {elapsed:.1f}s "
              f"({elapsed/args.num_steps:.2f}s/step avg)", flush=True)
        frames = result.frames[0]
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            frames[0].save(out)
            print(f"[run] WROTE {out} ({len(frames)} frames)", flush=True)
            try:
                from diffusers.utils import export_to_video
                mp4 = out.with_suffix(".mp4")
                export_to_video(frames, str(mp4), fps=24)
                print(f"[run] WROTE {mp4}", flush=True)
            except Exception as e:
                print(f"[run] mp4 export skip: {e}", flush=True)

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
