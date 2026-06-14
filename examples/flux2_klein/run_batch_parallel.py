#!/usr/bin/env python3
"""Run FLUX.2-klein on a single Neuron core. Designed to be launched
as N parallel processes pinned to different cores via
`NEURON_RT_VISIBLE_CORES=N`.

Each process is independent — separate Python interpreter, separate
pipeline, separate Neuron runtime. This is "batch parallelism" or
"data parallelism" — N images served concurrently, no sharing.

For trn2.3xl with LNC=2 (2 logical cores), launch 2 instances:

    NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
        python run_batch_parallel.py --core 0 \\
            --image input.jpg --steps 28 &
    NEURON_RT_VISIBLE_CORES=2-3 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
        python run_batch_parallel.py --core 1 \\
            --image input.jpg --steps 28 &
    wait

Each process produces one image, runs independently, and the
wall-clock for both is roughly the same as a single-core run (since
each core is fully utilized in parallel). Aggregate throughput on
trn2.3xl: 2× single-core, ~38.5 s/image at 1024×1024.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from neuron_flux2_klein_native import NeuronFlux2KleinPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=int, default=0,
                    help="logical core index (used for output filename + per-core seed offset)")
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default=None,
                    help="Optional HF LoRA repo id, e.g. <provider>/flux-2-klein-4B-<adapter>")
    ap.add_argument("--lora-scale", type=float, default=1.1)
    ap.add_argument("--image", required=True,
                    help="Path to input image (jpg/png)")
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None,
                    help="Output path (default: ./flux_batch_core<N>.png)")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    out_path = args.output or f"./flux_batch_core{args.core}.png"

    # Tag prints with the core number for clarity
    def log(msg):
        print(f"[core{args.core}] {msg}", flush=True)

    log(f"loading pipeline (NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES', 'unset')})")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.lora:
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        pipe.unload_lora_weights()
        log(f"loaded base + LoRA fused (scale={args.lora_scale}) in {time.time()-t0:.1f}s")
    else:
        log(f"loaded base in {time.time()-t0:.1f}s (no LoRA)")

    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)
    log("transformer on neuron")

    if not args.no_compile:
        pipe.transformer.inner = torch.compile(
            pipe.transformer.inner, backend="neuron", dynamic=False,
        )
        log("torch.compile applied")

    img = Image.open(args.image).convert("RGB").resize(
        (args.width, args.height), Image.LANCZOS,
    )

    # Different seed per core so we get different outputs
    seed = args.seed + args.core * 7919
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Warmup (uses NEFF cache if available)
    log(f"=== warmup ({args.steps} steps, {args.height}×{args.width}, seed={seed}) ===")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt, image=img, height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=3.5, generator=gen,
    )
    if hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    warmup_s = time.time() - t0
    log(f"warmup: {warmup_s:.1f}s")

    # Timed run
    gen = torch.Generator(device="cpu").manual_seed(seed)
    log(f"=== timed run ===")
    t0 = time.time()
    out2 = pipe(
        prompt=args.prompt, image=img, height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=3.5, generator=gen,
    )
    if hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    timed_s = time.time() - t0
    log(f"timed: {timed_s:.1f}s ({timed_s*1000/args.steps:.0f} ms/step)")

    out2.images[0].save(out_path)
    log(f"saved {out_path}")


if __name__ == "__main__":
    main()
