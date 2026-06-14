#!/usr/bin/env python3
"""FLUX.2-klein-4B + zoom-LoRA on Trainium2 — native PyTorch + Beta 3.

Uses `NeuronFlux2KleinPipeline` (a subclass of diffusers'
`Flux2KleinPipeline`) which:
  * Loads everything on CPU first
  * Installs 8 Neuron-specific patches (scheduler / timesteps / RoPE /
    pos_embed / VAE / latents / generator / DiT wrapper)
  * Moves only the DiT to Neuron via `.to(neuron_device)`

Run inside the Beta 3 container:

    HF_TOKEN=<token> \\
    /opt/torch-neuronx/.venv/bin/python /host/run_flux2_klein_native.py \\
        --base-model black-forest-labs/FLUX.2-klein-4B \\
        --lora <provider>/flux-2-klein-4B-zoom-lora \\
        --image /host/input_with_red_box.png \\
        --prompt "Zoom into the red highlighted area" \\
        --steps 28 \\
        --output /host/zoomed.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image, ImageDraw

# Local module
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline


def neuron_sync() -> None:
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default=None,
                    help="HF repo for the LoRA adapter, e.g. "
                         "<provider>/flux-2-klein-4B-zoom-lora. "
                         "Required unless --no-lora is set.")
    ap.add_argument("--no-lora", action="store_true",
                    help="Skip LoRA load (run base FLUX.2-klein only).")
    ap.add_argument("--lora-scale", type=float, default=1.1)
    ap.add_argument("--image", required=False)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="zoomed.png")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    print(f"[stage] loading {args.base_model} on CPU")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[stage] pipeline loaded in {time.time()-t0:.1f}s on CPU")

    if args.no_lora:
        print("[stage] --no-lora set, skipping LoRA fuse")
    else:
        if not args.lora:
            raise SystemExit(
                "ERROR: --lora is required (e.g. <provider>/flux-2-klein-4B-zoom-lora) "
                "or pass --no-lora to run the base model only."
            )
        print(f"[stage] applying LoRA {args.lora} (scale={args.lora_scale})")
        t0 = time.time()
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        pipe.unload_lora_weights()
        print(f"[stage] LoRA fused in {time.time()-t0:.1f}s")

    device = torch.device("neuron")
    print(f"[stage] applying neuron patches")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)

    print(f"[stage] moving transformer to {device} (single-core eager)")
    t0 = time.time()
    pipe.transformer.to(device)
    neuron_sync()
    print(f"[stage] transformer on neuron in {time.time()-t0:.1f}s")

    if not args.no_compile:
        print(f"[stage] applying torch.compile(backend='neuron') to inner DiT")
        # Compile the INNER DiT, not the wrapper — the wrapper does
        # boundary work (device coerce + .contiguous) that doesn't
        # belong inside the compiled graph.
        pipe.transformer.inner = torch.compile(
            pipe.transformer.inner, backend="neuron", dynamic=False,
        )
        print(f"[stage] compile decorator applied")

    # Build inputs
    if args.bench_only or args.image is None:
        print(f"[stage] no --image given; generating synthetic input "
              f"({args.height}x{args.width}) with red highlight box")
        img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        x0, y0 = args.width // 4, args.height // 4
        x1, y1 = 3 * args.width // 4, 3 * args.height // 4
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
        input_image = img
    else:
        print(f"[stage] loading input image: {args.image}")
        input_image = Image.open(args.image).convert("RGB")
        if input_image.size != (args.width, args.height):
            print(f"[stage] resizing {input_image.size} -> ({args.width}, {args.height})")
            input_image = input_image.resize((args.width, args.height), Image.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"[stage] === first call (compiles {args.steps}-step graph) ===")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        image=input_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    neuron_sync()
    first_call_s = time.time() - t0
    print(f"[time] first call: {first_call_s:.1f} s")

    out.images[0].save(args.output)
    print(f"[result] wrote {args.output} ({out.images[0].size})")

    print(f"[stage] === second call (cached) ===")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    out2 = pipe(
        prompt=args.prompt,
        image=input_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    neuron_sync()
    second_call_s = time.time() - t0
    print(f"[time] cached: {second_call_s:.1f} s")
    per_step_ms = second_call_s * 1000.0 / args.steps
    print(f"[time] per-step (cached): {per_step_ms:.0f} ms")

    if not args.bench_only:
        out2_path = args.output.replace(".png", "_cached.png")
        out2.images[0].save(out2_path)
        print(f"[result] wrote {out2_path}")

    print("\n=== SUMMARY ===")
    print(f"  steps:               {args.steps}")
    print(f"  resolution:          {args.height}x{args.width}")
    print(f"  first call (s):      {first_call_s:.1f}  (includes neuronx-cc compile)")
    print(f"  cached call (s):     {second_call_s:.1f}")
    print(f"  per-step (ms):       {per_step_ms:.0f}")


if __name__ == "__main__":
    main()
